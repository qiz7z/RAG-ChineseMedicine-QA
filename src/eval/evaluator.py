# -*- coding: utf-8 -*-
"""
评估引擎
========
对 RAG 系统进行全量评估。

评估指标：
  ┌────────────────────────────────────────────────────────────┐
  │  检索质量（两档判分口径并存，见文件下方「判分口径」）       │
  │  ├── loose  Hit@1/@3/@5:  药品双向子串匹配（历史报告口径） │
  │  ├── strict Hit@1/@3/@5:  药品精确 + 章节真实出现          │
  │  ├── MRR:                  平均倒数排名（两档各一份）      │
  │  └── 各阶段延迟:           解析/向量/BM25/RRF/重排          │
  ├────────────────────────────────────────────────────────────┤
  │  生成质量                                                  │
  │  ├── 关键词覆盖率:         标准答案关键词在回答中出现比例   │
  │  ├── 一致性问题率:         存在一致性校验问题的回答比例     │
  │  ├── 引用率:               包含引用标注的回答比例           │
  │  └── 安全提醒率:           包含用药安全提醒的比例           │
  ├────────────────────────────────────────────────────────────┤
  │  系统性能                                                  │
  │  ├── 端到端延迟 P50/P95/P99                                │
  │  └── 各组件延迟（检索/生成/后处理）                        │
  └────────────────────────────────────────────────────────────┘

使用方式：
    from eval.evaluator import RetrievalEvaluator, GenerationEvaluator

    # 检索评估
    r_eval = RetrievalEvaluator(retriever)
    r_report = r_eval.evaluate(test_queries)

    # 生成评估
    g_eval = GenerationEvaluator(generator)
    g_report = g_eval.evaluate(test_queries)
"""
import re
import sys
import time
import json
import statistics
import logging
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ============================================================
# 关键词匹配用文本规范化
# ============================================================
# 药典原文用「〜」(U+301C) 表示剂量范围，而测试集的 expected_answer_keywords
# 写成半角「-」（如期望 "6-12g"、原文实际是 "6〜12g"），直接做子串匹配会把
# **答对的题判成未命中**。实测该差异使关键词覆盖率系统性低估约 7~8pp，
# 且 0 覆盖题里近一半是此原因（见 docs/08）。
# 注意：两套引擎刻意各写一份，改这里必须同步改 langchain_app/eval.py。
_DASH_CHARS = "〜～~–—－‑"


def normalize_text(s: str) -> str:
    """关键词匹配前规范化：NFKC 折叠全角/半角 + 统一各类连字符 + 去空白"""
    s = unicodedata.normalize("NFKC", s or "")
    for ch in _DASH_CHARS:
        s = s.replace(ch, "-")
    return re.sub(r"\s+", "", s).lower()


# ============================================================
# 判分口径
# ============================================================
# 检索指标同时输出两档，报告里为 retrieval（loose）与 strict_retrieval（strict）：
#
#   loose  （宽松 / 粗召回）—— 历史报告沿用的口径，保留以对齐旧报告
#           ① drug_name 与 expected_drugs 双向子串匹配
#           ② expected_drugs 为空的题无条件判命中 → **已在 evaluate() 里整体排除**
#              （标 out_of_scope 的题不参与任何口径，见 test_queries.json）
#   strict （严格）
#           ① 药品名精确：drug_name == 期望药品，或该药品的饮片条目（"人参" → "人参-饮片"）
#           ② 期望章节必须在该 chunk 正文里真实出现（【章节】标记）
#           ③ expected_drugs 为空的题不计入分母（不可评测）
#
# loose 口径的放大器（历史数字偏高的根因；保留 loose 是为了让
# 手撕版/标准版/历史三方可比，不是为了报高）：
#   - 子串匹配把「黄连胶囊」「天麻祛风补片」这类含该药的成方制剂算成命中
#
# 2026-09-22 口径收口：原本有 11 道 expected_drugs 为空的题（10 道"方法通则查询"
# 考的是药典**四部**通则正文，而检索语料只有**一部**，源里没有答案；另 1 道是
# 集合型横向查询）。它们在 loose 里被无条件送分、在 strict 里被排除，
# 导致两档分母不一致（loose 120 / strict 109）。现已给这些题打 `out_of_scope` 标记，
# 由 evaluate() 统一剔除 → **两档分母一致，loose 不再有白送分**。
#
# 注意：langchain_app/eval.py 有一份**刻意重复**的等价实现（两套引擎要求零共享
# import，判分公式靠人工对齐）。改这个文件的口径时，必须同步改那一份。

_SECTION_MARK_RE = re.compile(r"【([^】]{1,20})】")

# 期望章节为「来源」时，正文可能不带【来源】标记：
#   - chunk_type=summary（药品概要）把 intro 改写成「概述：」
#   - chunk_type=whole_entry（完整条目）把 intro 直接拼在最前、无标记
# 见 src/etl/chunker.py 的 _make_summary_chunk / _make_whole_entry_chunk
_SOURCE_SECTIONS = ("药品概要", "完整条目")


def drug_match_loose(drug_name: str, expected_drugs: List[str]) -> bool:
    """宽松药品匹配：双向子串；无期望药品按历史口径无条件命中。

    注意 `d in e` 方向（检索到的比期望短，如期望"人参-饮片"、实得"人参"）是
    历史口径的一部分，保留不改。空 drug_name 一律不命中——否则 `"" in "人参"`
    为真，会让任何空名结果白拿一分（历史数据中未出现，属预防性收紧）。
    """
    if not expected_drugs:
        return True
    d = (drug_name or "").strip()
    if not d:
        return False
    return any(e in d or d in e for e in expected_drugs)


def drug_match_strict(drug_name: str, expected_drugs: List[str],
                      is_yinpian: bool = False) -> bool:
    """严格药品匹配：精确同名，或该药品的饮片条目（"人参" → "人参-饮片"）。

    要求检索到的就是该药材/饮片本身的条目，拒绝「含该药的成方制剂/提取物」。
    """
    d = (drug_name or "").strip()
    if not d or not expected_drugs:
        return False
    for e in expected_drugs:
        if d == e:
            return True
        if is_yinpian and d.startswith(e + "-"):
            return True
    return False


def section_match(content: str, section: str, expected_sections: List[str]) -> bool:
    """严格章节匹配：期望章节需在 chunk 正文里真实出现。

    不能直接拿检索引擎的 `section` 字段判：ETL 会把
    【性味与归经】【功能与主治】【用法与用量】【注意】【贮藏】【规格】
    合并成一个 "临床应用" 桶（见 src/etl/chunker.py::_merge_clinical_sections），
    桶名与测试集里的原生章节名根本不是同一套词汇，直接比对会大面积假阴性。
    """
    if not expected_sections or not content:
        return False
    marks = set(_SECTION_MARK_RE.findall(content))
    for e in expected_sections:
        if e in marks or e == (section or ""):
            return True
    if "来源" in expected_sections and section in _SOURCE_SECTIONS:
        return True
    return False


def judge_result(result, expected_drugs: List[str],
                 expected_sections: List[str]) -> Tuple[bool, bool, str]:
    """对单条检索结果同时给出两档判定。

    Args:
        result: SearchResult（需有 drug_name / section / content / is_yinpian）

    Returns:
        (loose_ok, strict_ok, miss_reason)
        miss_reason ∈ {"", "unevaluable", "drug_not_recalled", "drug_not_exact",
                       "section_missing"}；前两项由调用方按上下文补齐。
    """
    drug = getattr(result, "drug_name", "") or ""
    sect = getattr(result, "section", "") or ""
    content = getattr(result, "content", "") or ""
    is_yin = bool(getattr(result, "is_yinpian", False))

    loose = drug_match_loose(drug, expected_drugs)

    if not expected_drugs:
        return loose, False, "unevaluable"
    if not drug_match_strict(drug, expected_drugs, is_yin):
        return loose, False, "drug_not_exact"
    if not section_match(content, sect, expected_sections):
        return loose, False, "section_missing"
    return loose, True, ""


def attribute_strict_miss(results, expected_drugs: List[str], k: int = 5) -> str:
    """strict 未命中时归因（只看前 K 条，与 Hit@K 口径一致）。

    区分三种情况，对症下药：
      drug_not_recalled —— 该药品压根没进前 K（覆盖问题）
      drug_not_exact    —— 召回了含该药的别的东西（子串匹配放大器，判定/排序问题）
      section_missing   —— 药品对了但章节不对（切片或排序问题）
    """
    if not expected_drugs:
        return "unevaluable"
    top = list(results)[:k]
    if not any(drug_match_loose(getattr(r, "drug_name", "") or "", expected_drugs)
               for r in top):
        return "drug_not_recalled"
    if not any(drug_match_strict(getattr(r, "drug_name", "") or "", expected_drugs,
                                 bool(getattr(r, "is_yinpian", False)))
               for r in top):
        return "drug_not_exact"
    return "section_missing"


# ============================================================
# 评估结果数据结构
# ============================================================

@dataclass
class QueryResult:
    """单条测试结果"""
    query_id: str
    query: str
    query_type: str

    # 检索结果
    retrieved_drugs: List[str] = field(default_factory=list)
    retrieved_sections: List[str] = field(default_factory=list)
    hit: bool = False                    # loose 口径是否命中（Hit@K，与历史报告可比）
    first_hit_rank: int = 0              # loose 首次命中排名（1-based，0 表示未命中）
    retrieval_latency: float = 0.0

    # 严格口径（药品精确 + 章节真实出现；仅可评测题有意义）
    evaluable: bool = True               # expected_drugs 非空 → 可评测
    strict_hit: bool = False             # strict 口径是否命中（Hit@K）
    strict_first_hit_rank: int = 0       # strict 首次命中排名（1-based）
    strict_miss_reason: str = ""         # 未命中归因，见 attribute_strict_miss

    # 生成结果
    answer: str = ""
    answer_keywords_hit: int = 0         # 命中关键词数
    answer_keywords_total: int = 0       # 总关键词数
    keyword_coverage: float = 0.0        # 关键词覆盖率
    has_citations: bool = False
    has_consistency_issues: bool = False
    has_medical_disclaimer: bool = False
    generation_latency: float = 0.0
    component_latency: Dict[str, float] = field(default_factory=dict)

    # 有据性判分（LLM-as-judge，见 grounding_judge.py；未开启时为空 dict）
    grounding: Dict = field(default_factory=dict)

    # 第二判官（引文核验式 QuoteVerifiedJudge）的判定，仅 --crosscheck 时有值。
    # 与 grounding **机制不同**：判定建立在可机械核验的逐字引文上，
    # 供与第一判官交叉验证（两者一致性低 = 判分不可信，需人工抽查）。
    grounding_crosscheck: Dict = field(default_factory=dict)

    # 详细检索结果（用于分析）
    top_k_details: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_type": self.query_type,
            "retrieved_drugs": self.retrieved_drugs,
            "retrieved_sections": self.retrieved_sections,
            "hit": self.hit,
            "first_hit_rank": self.first_hit_rank,
            "evaluable": self.evaluable,
            "strict_hit": self.strict_hit,
            "strict_first_hit_rank": self.strict_first_hit_rank,
            "strict_miss_reason": self.strict_miss_reason,
            "retrieval_latency": round(self.retrieval_latency, 3),
            "answer": self.answer,
            "answer_keywords_hit": self.answer_keywords_hit,
            "answer_keywords_total": self.answer_keywords_total,
            "keyword_coverage": round(self.keyword_coverage, 4),
            "has_citations": self.has_citations,
            "has_consistency_issues": self.has_consistency_issues,
            "has_medical_disclaimer": self.has_medical_disclaimer,
            "generation_latency": round(self.generation_latency, 3),
            "component_latency": {k: round(v, 3) for k, v in self.component_latency.items()},
            "grounding": self.grounding,
            "grounding_crosscheck": self.grounding_crosscheck,
            "top_k_details": self.top_k_details,
        }


@dataclass
class EvalReport:
    """评估报告"""
    # 基本信息
    eval_type: str = ""
    total_queries: int = 0
    timestamp: str = ""

    # 检索指标（loose：历史报告口径，保留以保持可比）
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0

    # 检索指标（strict：药品精确 + 章节真实出现，分母仅含可评测题）
    strict_hit_at_1: float = 0.0
    strict_hit_at_3: float = 0.0
    strict_hit_at_5: float = 0.0
    strict_mrr: float = 0.0
    evaluable_queries: int = 0
    unevaluable_queries: int = 0
    out_of_scope_queries: int = 0    # 标了 out_of_scope 的题：两项口径都不计分

    # 生成指标
    avg_keyword_coverage: float = 0.0
    citation_rate: float = 0.0
    consistency_issue_rate: float = 0.0
    medical_disclaimer_rate: float = 0.0
    grounding: Dict[str, Any] = field(default_factory=dict)   # 有据性判分汇总，见 grounding_judge.py
    grounding_crosscheck: Dict[str, Any] = field(default_factory=dict)   # 第二判官（引文核验式）汇总

    # 性能指标
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    avg_latency: float = 0.0
    avg_retrieval_latency: float = 0.0
    avg_llm_latency: float = 0.0

    # 分类型结果
    by_type: Dict[str, Dict] = field(default_factory=dict)

    # 详细结果
    results: List[QueryResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "eval_type": self.eval_type,
            "total_queries": self.total_queries,
            "timestamp": self.timestamp,
            "retrieval": {
                "hit_at_1": round(self.hit_at_1, 4),
                "hit_at_3": round(self.hit_at_3, 4),
                "hit_at_5": round(self.hit_at_5, 4),
                "mrr": round(self.mrr, 4),
                "criteria": "loose：drug_name 双向子串匹配；expected_drugs 为空的题无条件命中"
                            "（历史报告沿用的粗召回口径，保留以保持可比）",
            },
            "strict_retrieval": {
                "hit_at_1": round(self.strict_hit_at_1, 4),
                "hit_at_3": round(self.strict_hit_at_3, 4),
                "hit_at_5": round(self.strict_hit_at_5, 4),
                "mrr": round(self.strict_mrr, 4),
                "evaluable_queries": self.evaluable_queries,
                "unevaluable_queries": self.unevaluable_queries,
                "out_of_scope_queries": self.out_of_scope_queries,
                "criteria": "strict：药品名精确（含该药饮片条目）+ 期望章节在 chunk 正文真实出现；"
                            "分母仅含可评测题（expected_drugs 非空）；"
                            "标了 out_of_scope 的题不计分",
            },
            "generation": {
                "avg_keyword_coverage": round(self.avg_keyword_coverage, 4),
                "citation_rate": round(self.citation_rate, 4),
                "consistency_issue_rate": round(self.consistency_issue_rate, 4),
                "medical_disclaimer_rate": round(self.medical_disclaimer_rate, 4),
                # 有据性判分（LLM-as-judge）——**下表中的对外数字应优先用这一块**
                "grounding": self.grounding,
                # 第二判官（引文核验式）汇总，仅 --crosscheck 时非空。
                # 空字典表示未启用，不占报告体积。
                "grounding_crosscheck": self.grounding_crosscheck,
            },
            "performance": {
                "latency_p50": round(self.latency_p50, 3),
                "latency_p95": round(self.latency_p95, 3),
                "latency_p99": round(self.latency_p99, 3),
                "avg_latency": round(self.avg_latency, 3),
                "avg_retrieval_latency": round(self.avg_retrieval_latency, 3),
                "avg_llm_latency": round(self.avg_llm_latency, 3),
            },
            "by_type": self.by_type,
            "results": [r.to_dict() for r in self.results],
        }


# ============================================================
# 检索评估器
# ============================================================

class RetrievalEvaluator:
    """
    检索质量评估器。

    对每条测试查询执行检索，同时按两档口径计算 Hit@K 和 MRR
    （口径定义见本模块顶部「判分口径」）：

      loose  （历史口径）
        - 有 expected_drugs：drug_name 与期望药品双向子串匹配即命中
        - 无 expected_drugs：检索到任何结果即命中
      strict （严格口径，仅统计 expected_drugs 非空的题）
        - 药品名必须精确（或为该药品的饮片条目）
        - 且期望章节必须在该 chunk 正文里真实出现

    strict 未命中的题会写入 strict_miss_reason 归因，便于定位是覆盖问题、
    判定问题还是章节/切片问题。
    """

    def __init__(self, retriever, k_values: List[int] = None):
        """
        Args:
            retriever: Retriever 实例
            k_values: 评估的 K 值列表，默认 [1, 3, 5]
        """
        self.retriever = retriever
        self.k_values = k_values or [1, 3, 5]

    def _check_hit(self, result, expected_drugs: List[str], expected_sections: List[str]) -> bool:
        """loose 口径判定（保留原签名，语义与历史报告一致）。

        规则：有 expected_drugs → 药品名双向子串匹配；无 → 无条件命中。
        expected_sections 在此档不参与判定（见模块顶部说明）。
        """
        return drug_match_loose(getattr(result, "drug_name", "") or "", expected_drugs)

    def _check_hit_strict(self, result, expected_drugs: List[str],
                          expected_sections: List[str]) -> bool:
        """strict 口径判定：药品精确（含饮片）+ 章节真实出现。"""
        _loose, strict, _reason = judge_result(result, expected_drugs, expected_sections)
        return strict

    def evaluate_single(self, test_case: Dict) -> QueryResult:
        """评估单条查询"""
        qr = QueryResult(
            query_id=test_case["id"],
            query=test_case["query"],
            query_type=test_case["type"],
        )

        expected_drugs = test_case.get("expected_drugs") or []
        expected_sections = test_case.get("expected_sections") or []

        # 可评测性：expected_drugs 为空（方法通则/横向条件）无法做药品级判分
        qr.evaluable = bool(expected_drugs)

        # 执行检索
        t0 = time.time()
        response = self.retriever.search(test_case["query"])
        qr.retrieval_latency = time.time() - t0

        results = response.results

        # 记录检索结果的药品和章节
        qr.retrieved_drugs = [r.drug_name for r in results[:10]]
        qr.retrieved_sections = [r.section for r in results[:10]]

        # 记录 Top-K 详情（附逐条两档判定，便于离线审计，无需再反查 chunks.json）
        qr.top_k_details = []
        for i, r in enumerate(results[:10]):
            _loose, strict_ok, _reason = judge_result(r, expected_drugs, expected_sections)
            qr.top_k_details.append({
                "rank": i + 1,
                "drug_name": r.drug_name,
                "section": r.section,
                "score": round(r.rerank_score or r.score, 4),
                "content_preview": r.content[:100].replace('\n', ' '),
                "drug_exact": drug_match_strict(
                    r.drug_name, expected_drugs, bool(getattr(r, "is_yinpian", False))),
                "section_ok": section_match(
                    getattr(r, "content", "") or "", r.section, expected_sections),
                "strict": strict_ok,
            })

        # 计算两档 Hit@K 与首次命中排名（1-based）
        qr.first_hit_rank = 0
        qr.strict_first_hit_rank = 0
        for i, r in enumerate(results, 1):
            loose, strict, _reason = judge_result(r, expected_drugs, expected_sections)
            if loose and qr.first_hit_rank == 0:
                qr.first_hit_rank = i
            if strict and qr.strict_first_hit_rank == 0:
                qr.strict_first_hit_rank = i
            # 两档都已定位（不可评测题无需等 strict）即可提前结束
            if qr.first_hit_rank and (qr.strict_first_hit_rank or not qr.evaluable):
                break

        qr.hit = qr.first_hit_rank > 0
        qr.strict_hit = qr.strict_first_hit_rank > 0

        # strict 未命中归因：区分"没召回 / 召回的不是该药材 / 章节不对"
        if not qr.evaluable:
            qr.strict_miss_reason = "unevaluable"
        elif not qr.strict_hit:
            qr.strict_miss_reason = attribute_strict_miss(
                results, expected_drugs, k=max(self.k_values))

        return qr

    def evaluate(self, test_queries: List[Dict], verbose: bool = True) -> EvalReport:
        """
        评估全部测试查询。

        Args:
            test_queries: 测试查询列表
            verbose: 是否打印进度

        Returns:
            EvalReport 评估报告
        """
        # 剔除 out_of_scope 题：两项口径都不计分（见 test_queries.json 的题级说明）
        n_all = len(test_queries)
        test_queries = [tc for tc in test_queries if not tc.get("out_of_scope")]
        out_of_scope = n_all - len(test_queries)

        report = EvalReport(
            eval_type="retrieval",
            total_queries=len(test_queries),
            out_of_scope_queries=out_of_scope,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if out_of_scope and verbose:
            print(f"  已剔除 {out_of_scope} 道 out_of_scope 题（不计分，见 test_queries.json）")

        results = []
        for i, tc in enumerate(test_queries):
            if verbose:
                print(f"  [{i+1}/{len(test_queries)}] {tc['id']} {tc['query'][:40]}...", end="", flush=True)

            qr = self.evaluate_single(tc)
            results.append(qr)

            if verbose:
                hit_str = f"rank={qr.first_hit_rank}" if qr.hit else "MISS"
                print(f" -> {hit_str} ({qr.retrieval_latency:.2f}s)")

        report.results = results

        # 计算汇总指标
        self._compute_metrics(report)

        return report

    def _compute_metrics(self, report: EvalReport):
        """计算汇总指标"""
        results = report.results
        n = len(results)
        if n == 0:
            return

        # Hit@K（loose，历史口径，全量题）
        for k in self.k_values:
            hits = sum(1 for r in results if 0 < r.first_hit_rank <= k)
            hit_rate = hits / n
            if k == 1:
                report.hit_at_1 = hit_rate
            elif k == 3:
                report.hit_at_3 = hit_rate
            elif k == 5:
                report.hit_at_5 = hit_rate

        # MRR
        rr_sum = sum(1.0 / r.first_hit_rank for r in results if r.first_hit_rank > 0)
        report.mrr = rr_sum / n

        # ---- strict 口径：只在可评测题上统计（expected_drugs 非空）----
        evaluable = [r for r in results if r.evaluable]
        report.evaluable_queries = len(evaluable)
        report.unevaluable_queries = n - len(evaluable)
        m = len(evaluable)
        if m:
            for k in self.k_values:
                strict_rate = sum(1 for r in evaluable
                                  if 0 < r.strict_first_hit_rank <= k) / m
                if k == 1:
                    report.strict_hit_at_1 = strict_rate
                elif k == 3:
                    report.strict_hit_at_3 = strict_rate
                elif k == 5:
                    report.strict_hit_at_5 = strict_rate
            report.strict_mrr = sum(
                1.0 / r.strict_first_hit_rank for r in evaluable
                if r.strict_first_hit_rank > 0) / m

        # 延迟统计
        latencies = [r.retrieval_latency for r in results]
        report.avg_retrieval_latency = statistics.mean(latencies)
        sorted_lat = sorted(latencies)
        report.latency_p50 = self._percentile(sorted_lat, 50)
        report.latency_p95 = self._percentile(sorted_lat, 95)
        report.latency_p99 = self._percentile(sorted_lat, 99)
        report.avg_latency = report.avg_retrieval_latency

        # 分类型统计（两档并列）
        by_type = {}
        for r in results:
            t = r.query_type
            if t not in by_type:
                by_type[t] = {"count": 0, "evaluable": 0, "hits": 0, "strict_hits": 0,
                              "mrr_sum": 0.0, "strict_mrr_sum": 0.0, "latency_sum": 0.0}
            by_type[t]["count"] += 1
            if r.hit:
                by_type[t]["hits"] += 1
            if r.first_hit_rank > 0:
                by_type[t]["mrr_sum"] += 1.0 / r.first_hit_rank
            if r.evaluable:
                by_type[t]["evaluable"] += 1
                if r.strict_hit:
                    by_type[t]["strict_hits"] += 1
                if r.strict_first_hit_rank > 0:
                    by_type[t]["strict_mrr_sum"] += 1.0 / r.strict_first_hit_rank
            by_type[t]["latency_sum"] += r.retrieval_latency

        for t, v in by_type.items():
            c = v["count"]
            ev = v["evaluable"]
            v["hit_at_5"] = round(v["hits"] / c, 4) if c > 0 else 0
            v["mrr"] = round(v["mrr_sum"] / c, 4) if c > 0 else 0
            v["strict_hit_at_5"] = round(v["strict_hits"] / ev, 4) if ev > 0 else None
            v["strict_mrr"] = round(v["strict_mrr_sum"] / ev, 4) if ev > 0 else None
            v["avg_latency"] = round(v["latency_sum"] / c, 3) if c > 0 else 0
            del v["hits"]
            del v["strict_hits"]
            del v["mrr_sum"]
            del v["strict_mrr_sum"]
            del v["latency_sum"]

        report.by_type = by_type

    @staticmethod
    def _percentile(sorted_list: List[float], p: float) -> float:
        """计算百分位数"""
        if not sorted_list:
            return 0.0
        idx = int(len(sorted_list) * p / 100)
        idx = min(idx, len(sorted_list) - 1)
        return sorted_list[idx]


# ============================================================
# 生成评估器
# ============================================================

class GenerationEvaluator:
    """
    生成质量评估器。

    对每条测试查询执行完整的 RAG 流程（检索 → 生成 → 后处理），
    评估回答的关键词覆盖率、引用率、一致性问题和延迟。
    """

    def __init__(self, generator, grounding_judge=None, crosscheck_judge=None):
        """
        Args:
            generator: Generator 实例
            grounding_judge: 可选，`eval.grounding_judge.GroundingJudge` 实例。
                传入则额外做**逐论断有据性判定**，产出真实的幻觉率与「回答有据率」
                （替代恒真的 citation_rate 与只查数值的 consistency_issue_rate）。
                不传则跳过（不产生额外 LLM 调用）。
            crosscheck_judge: 可选，`QuoteVerifiedJudge` 实例（引文核验式**第二判官**）。
                与 grounding_judge 的机制不同（要求逐字抄出支撑原文，代码再核验引文），
                用于**交叉验证**第一判官的结论——两判官不一致率高说明判分不可信。
                传入则每题额外多一次 LLM 调用，只在 `--crosscheck` 时启用。
        """
        self.generator = generator
        self.grounding_judge = grounding_judge
        self.crosscheck_judge = crosscheck_judge

    def evaluate_single(self, test_case: Dict) -> QueryResult:
        """评估单条查询"""
        qr = QueryResult(
            query_id=test_case["id"],
            query=test_case["query"],
            query_type=test_case["type"],
        )

        expected_keywords = test_case.get("expected_answer_keywords", [])
        expected_drugs = test_case.get("expected_drugs") or []
        expected_sections = test_case.get("expected_sections") or []
        qr.evaluable = bool(expected_drugs)

        # 执行端到端问答
        t0 = time.time()
        response = self.generator.answer(test_case["query"])
        total_latency = time.time() - t0

        # 填充结果
        qr.answer = response.answer
        qr.generation_latency = total_latency
        qr.component_latency = response.component_latency
        qr.retrieval_latency = response.retrieval.latency if response.retrieval else 0

        # 检索结果信息 + 两档检索判定（用于把"检索错"和"生成错"分开归因）
        if response.retrieval and response.retrieval.results:
            results = response.retrieval.results
            qr.retrieved_drugs = [r.drug_name for r in results[:5]]
            qr.retrieved_sections = [r.section for r in results[:10]]
            for i, r in enumerate(results, 1):
                loose, strict, _reason = judge_result(r, expected_drugs, expected_sections)
                if loose and qr.first_hit_rank == 0:
                    qr.first_hit_rank = i
                if strict and qr.strict_first_hit_rank == 0:
                    qr.strict_first_hit_rank = i
            qr.hit = qr.first_hit_rank > 0
            qr.strict_hit = qr.strict_first_hit_rank > 0
            if qr.evaluable and not qr.strict_hit:
                qr.strict_miss_reason = attribute_strict_miss(
                    results, expected_drugs, k=5)
        elif not qr.evaluable:
            qr.strict_miss_reason = "unevaluable"

        # 关键词覆盖率（先做规范化，避免全角/波浪线差异造成假阴性，见 normalize_text）
        if expected_keywords:
            norm_answer = normalize_text(response.answer)
            hits = sum(1 for kw in expected_keywords
                       if kw and normalize_text(kw) in norm_answer)
            qr.answer_keywords_hit = hits
            qr.answer_keywords_total = len(expected_keywords)
            qr.keyword_coverage = hits / len(expected_keywords)

        # 引用、一致性、安全提醒
        qr.has_citations = len(response.citations) > 0
        qr.has_consistency_issues = len(response.consistency_issues) > 0
        qr.has_medical_disclaimer = "遵医嘱" in response.answer or "具体用药" in response.answer

        # 有据性判定（LLM-as-judge）
        # 注意：用的是**生成时实际喂给 LLM 的那批来源**（response.retrieval.results），
        # 不是事后重检索的结果——否则索引/配置一变就会拿新来源去判旧回答。
        srcs = []
        if response.retrieval and response.retrieval.results:
            srcs = [{"drug_name": r.drug_name, "section": r.section,
                     "content": r.content} for r in response.retrieval.results]
        # 有据性判定（LLM-as-judge）
        # 注意：用的是**生成时实际喂给 LLM 的那批来源**（response.retrieval.results），
        # 不是事后重检索的结果——否则索引/配置一变就会拿新来源去判旧回答。
        if self.grounding_judge is not None:
            qr.grounding = self.grounding_judge.judge(
                test_case["query"], response.answer, srcs).to_dict()
        # 第二判官（引文核验式，--crosscheck 时启用）：同一批来源，机制不同
        if self.crosscheck_judge is not None:
            qr.grounding_crosscheck = self.crosscheck_judge.judge(
                test_case["query"], response.answer, srcs).to_dict()

        return qr

    def evaluate(self, test_queries: List[Dict], verbose: bool = True) -> EvalReport:
        """
        评估全部测试查询。

        Args:
            test_queries: 测试查询列表
            verbose: 是否打印进度

        Returns:
            EvalReport 评估报告
        """
        # 剔除 out_of_scope 题（与检索侧同口径）
        n_all = len(test_queries)
        test_queries = [tc for tc in test_queries if not tc.get("out_of_scope")]
        out_of_scope = n_all - len(test_queries)

        report = EvalReport(
            eval_type="generation",
            total_queries=len(test_queries),
            out_of_scope_queries=out_of_scope,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        results = []
        for i, tc in enumerate(test_queries):
            if verbose:
                print(f"  [{i+1}/{len(test_queries)}] {tc['id']} {tc['query'][:40]}...", end="", flush=True)

            try:
                qr = self.evaluate_single(tc)
            except Exception as e:
                print(f" ERROR: {e}")
                qr = QueryResult(
                    query_id=tc["id"],
                    query=tc["query"],
                    query_type=tc["type"],
                    answer=f"[ERROR] {e}",
                )
            results.append(qr)

            if verbose:
                print(f" -> cov={qr.keyword_coverage:.2f} ({qr.generation_latency:.2f}s)")

        report.results = results
        self._compute_metrics(report)

        return report

    def _compute_metrics(self, report: EvalReport):
        """计算汇总指标"""
        results = report.results
        n = len(results)
        if n == 0:
            return

        # 生成质量
        coverages = [r.keyword_coverage for r in results]
        report.avg_keyword_coverage = statistics.mean(coverages) if coverages else 0
        report.citation_rate = sum(1 for r in results if r.has_citations) / n
        report.consistency_issue_rate = sum(1 for r in results if r.has_consistency_issues) / n

        # 有据性判分汇总（只有开了 grounding 才有数据）
        from eval.grounding_judge import GroundingResult          # 局部导入，避免无谓耦合

        def _aggregate_field(attr):
            gres = []
            for r in results:
                g = getattr(r, attr) or {}
                if not g:
                    continue
                # 按字段名重建（而非手工白名单）：新增字段会自动带上，
                # 不会再出现"逐题算对、聚合恒为 0"的静默丢失（缺陷 20b）
                gres.append(GroundingResult.from_dict(g))
            if gres:
                from eval.grounding_judge import aggregate as _agg
                return _agg(gres)
            return {}

        report.grounding = _aggregate_field("grounding")
        report.grounding_crosscheck = _aggregate_field("grounding_crosscheck")
        report.medical_disclaimer_rate = sum(1 for r in results if r.has_medical_disclaimer) / n

        # 性能指标
        latencies = [r.generation_latency for r in results]
        sorted_lat = sorted(latencies)
        report.avg_latency = statistics.mean(latencies)
        report.latency_p50 = RetrievalEvaluator._percentile(sorted_lat, 50)
        report.latency_p95 = RetrievalEvaluator._percentile(sorted_lat, 95)
        report.latency_p99 = RetrievalEvaluator._percentile(sorted_lat, 99)

        # 检索和 LLM 分别延迟
        retrieval_lats = [r.retrieval_latency for r in results if r.retrieval_latency > 0]
        report.avg_retrieval_latency = statistics.mean(retrieval_lats) if retrieval_lats else 0

        llm_lats = [r.component_latency.get("llm_generation", 0) for r in results]
        report.avg_llm_latency = statistics.mean(llm_lats) if llm_lats else 0

        # 分类型统计
        by_type = {}
        for r in results:
            t = r.query_type
            if t not in by_type:
                by_type[t] = {
                    "count": 0,
                    "cov_sum": 0.0,
                    "citation_count": 0,
                    "issue_count": 0,
                    "latency_sum": 0.0,
                }
            bt = by_type[t]
            bt["count"] += 1
            bt["cov_sum"] += r.keyword_coverage
            if r.has_citations:
                bt["citation_count"] += 1
            if r.has_consistency_issues:
                bt["issue_count"] += 1
            bt["latency_sum"] += r.generation_latency

        for t, v in by_type.items():
            c = v["count"]
            v["avg_keyword_coverage"] = round(v["cov_sum"] / c, 4) if c > 0 else 0
            v["citation_rate"] = round(v["citation_count"] / c, 4) if c > 0 else 0
            v["consistency_issue_rate"] = round(v["issue_count"] / c, 4) if c > 0 else 0
            v["avg_latency"] = round(v["latency_sum"] / c, 3) if c > 0 else 0
            del v["cov_sum"]
            del v["citation_count"]
            del v["issue_count"]
            del v["latency_sum"]

        report.by_type = by_type
