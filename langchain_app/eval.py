# -*- coding: utf-8 -*-
"""
独立评估器（标准版）
====================
判分公式与主项目 src/eval/evaluator.py **刻意重复实现**（两套引擎要求零共享
import，靠人工对齐口径）；改其中一份必须同步改另一份。

两档口径并存：

  loose （历史报告口径，保留以保持三方可比）
    - 有 expected_drugs：检索结果的 drug_name 与期望药品做双向子串匹配
    - 无 expected_drugs：检索到任何结果即判命中（会无条件送分，见下）
  strict （严格）
    - 药品名精确：drug_name == 期望药品，或该药品的饮片条目（"人参" → "人参-饮片"）
    - 且期望章节必须在该 chunk 正文里真实出现（【章节】标记；不能拿 metadata 里的
      section 字段判，ETL 把多个章节合并成了 "临床应用" 桶）
    - expected_drugs 为空的题不计入分母（不可评测）

loose 口径偏高的两个根因（保留它只为可比，不是为报高）：
  1. 子串匹配会把「黄连胶囊」「天麻祛风补片」这类含该药的成方制剂算成命中
  2. 11 道 expected_drugs 为空的题无条件送分，其中 10 道"方法通则查询"考的是
     药典四部通则正文，而检索语料只有一部 —— 源里没有答案

指标：Hit@1/3/5、MRR（两档各一份）、延迟 P50/P95/P99、分类型聚合。
"""
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple

from config import TEST_SET_PATH


# ============================================================
# 关键词匹配用文本规范化
# ============================================================
# 药典原文用「〜」(U+301C) 表示剂量范围，测试集的 expected_answer_keywords 却写成
# 半角「-」（期望 "6-12g"、原文 "6〜12g"），直接子串匹配会把答对的题判成未命中。
# 与 src/eval/evaluator.py::normalize_text **逐字对齐**（两份刻意重复的实现）。
_DASH_CHARS = "〜～~–—－‑"


def normalize_text(s: str) -> str:
    """关键词匹配前规范化：NFKC 折叠全角/半角 + 统一各类连字符 + 去空白"""
    s = unicodedata.normalize("NFKC", s or "")
    for ch in _DASH_CHARS:
        s = s.replace(ch, "-")
    return re.sub(r"\s+", "", s).lower()


@dataclass
class QueryResult:
    query_id: str
    query: str
    query_type: str
    retrieval_latency: float = 0.0
    doc_count: int = 0                   # 检索器实际返回的候选条数
    retrieved_drugs: List[str] = field(default_factory=list)
    retrieved_sections: List[str] = field(default_factory=list)
    top_k_details: List[dict] = field(default_factory=list)
    first_hit_rank: int = 0              # loose 首次命中排名（1-based）
    hit: bool = False                    # loose 是否命中
    evaluable: bool = True               # expected_drugs 非空 → 可评测
    strict_first_hit_rank: int = 0       # strict 首次命中排名（1-based）
    strict_hit: bool = False             # strict 是否命中
    strict_miss_reason: str = ""         # drug_not_recalled / drug_not_exact / section_missing


@dataclass
class EvalReport:
    engine: str
    total_queries: int
    timestamp: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    avg_latency: float
    avg_doc_count: float = 0.0           # 平均召回候选条数（诊断配置用）
    # strict 口径（分母仅含可评测题）
    strict_hit_at_1: float = 0.0
    strict_hit_at_3: float = 0.0
    strict_hit_at_5: float = 0.0
    strict_mrr: float = 0.0
    evaluable_queries: int = 0
    unevaluable_queries: int = 0
    out_of_scope_queries: int = 0    # 标了 out_of_scope 的题：两项口径都不计分
    # 召回率（项目书口径，定义与 src/eval/evaluator.py 完全一致）
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    full_recall_at_5: float = 0.0
    recall_queries: int = 0
    by_type: Dict[str, dict] = field(default_factory=dict)
    results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["criteria"] = {
            "loose": "drug_name 双向子串匹配；expected_drugs 为空的题无条件命中"
                     "（历史报告口径，保留以保持可比）",
            "strict": "药品名精确（含该药饮片条目）+ 期望章节在 chunk 正文真实出现；"
                      "分母仅含可评测题（expected_drugs 非空）",
        }
        return d


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(len(values) * p / 100), len(values) - 1)
    return values[idx]


# ------------------------------------------------------------
# 判分：与 src/eval/evaluator.py 的 drug_match_loose / drug_match_strict /
#       section_match / judge_result 语义必须逐字对齐
# ------------------------------------------------------------

_SECTION_MARK_RE = re.compile(r"【([^】]{1,20})】")

# 「来源」正文在 summary/whole_entry 两种 chunk 里不带【】标记
# （summary 改写成「概述：」，whole_entry 直接拼在最前）
_SOURCE_SECTIONS = ("药品概要", "完整条目")


def _drug_match_loose(drug_name: str, expected_drugs: List[str]) -> bool:
    """宽松：双向子串；无期望药品按历史口径无条件命中。

    空 drug_name 一律不命中（`"" in "人参"` 为真，会让空名结果白拿一分）。
    与 src/eval/evaluator.py::drug_match_loose 逐字对齐。
    """
    if not expected_drugs:
        return True
    d = (drug_name or "").strip()
    if not d:
        return False
    return any(e in d or d in e for e in expected_drugs)


def _drug_match_strict(drug_name: str, expected_drugs: List[str],
                       is_yinpian: bool = False) -> bool:
    """严格：精确同名，或该药品的饮片条目"""
    d = (drug_name or "").strip()
    if not d or not expected_drugs:
        return False
    for e in expected_drugs:
        if d == e:
            return True
        if is_yinpian and d.startswith(e + "-"):
            return True
    return False


def _section_match(content: str, section: str, expected_sections: List[str]) -> bool:
    """严格：期望章节需在 chunk 正文里真实出现"""
    if not expected_sections or not content:
        return False
    marks = set(_SECTION_MARK_RE.findall(content))
    for e in expected_sections:
        if e in marks or e == (section or ""):
            return True
    if "来源" in expected_sections and section in _SOURCE_SECTIONS:
        return True
    return False


def _judge(drug_name: str, section: str, content: str, is_yinpian: bool,
           expected_drugs: List[str], expected_sections: List[str]) -> Tuple[bool, bool, str]:
    """返回 (loose_ok, strict_ok, miss_reason)"""
    loose = _drug_match_loose(drug_name, expected_drugs)
    if not expected_drugs:
        return loose, False, "unevaluable"
    if not _drug_match_strict(drug_name, expected_drugs, is_yinpian):
        return loose, False, "drug_not_exact"
    if not _section_match(content, section, expected_sections):
        return loose, False, "section_missing"
    return loose, True, ""


def _attribute_miss(docs, expected_drugs: List[str], k: int = 5) -> str:
    """strict 未命中归因（只看前 K 条）：没召回 / 召回的不是该药材 / 章节不对"""
    if not expected_drugs:
        return "unevaluable"
    top = list(docs)[:k]
    if not any(_drug_match_loose(d.metadata.get("drug_name", ""), expected_drugs)
               for d in top):
        return "drug_not_recalled"
    if not any(_drug_match_strict(d.metadata.get("drug_name", ""), expected_drugs,
                                  bool(d.metadata.get("is_yinpian")))
               for d in top):
        return "drug_not_exact"
    return "section_missing"


def _check_hit(drug_name: str, expected_drugs: List[str]) -> bool:
    """loose 口径（保留原签名与语义，供既有测试/脚本调用）"""
    return _drug_match_loose(drug_name, expected_drugs)


def evaluate_retrieval(retriever, test_queries: List[Dict], engine: str = "lc-standard",
                       verbose: bool = True) -> EvalReport:
    # 剔除 out_of_scope 题：两项口径都不计分（与主项目 src/eval/evaluator.py 同口径）
    n_all = len(test_queries)
    test_queries = [tc for tc in test_queries if not tc.get("out_of_scope")]
    out_of_scope = n_all - len(test_queries)

    results: List[QueryResult] = []
    for i, tc in enumerate(test_queries, 1):
        expected = tc.get("expected_drugs") or []
        expected_sections = tc.get("expected_sections") or []
        t0 = time.time()
        docs = retriever.invoke(tc["query"])
        latency = time.time() - t0

        qr = QueryResult(
            query_id=tc["id"], query=tc["query"], query_type=tc["type"],
            retrieval_latency=latency, evaluable=bool(expected),
            doc_count=len(docs),
            retrieved_drugs=[d.metadata.get("drug_name", "") for d in docs[:10]],
            retrieved_sections=[d.metadata.get("section", "") for d in docs[:10]],
        )
        for rank, d in enumerate(docs, 1):
            loose, strict, _r = _judge(
                d.metadata.get("drug_name", ""), d.metadata.get("section", ""),
                d.page_content or "", bool(d.metadata.get("is_yinpian")),
                expected, expected_sections)
            if loose and qr.first_hit_rank == 0:
                qr.first_hit_rank = rank
            if strict and qr.strict_first_hit_rank == 0:
                qr.strict_first_hit_rank = rank
            if qr.first_hit_rank and (qr.strict_first_hit_rank or not qr.evaluable):
                break
        qr.hit = qr.first_hit_rank > 0
        qr.strict_hit = qr.strict_first_hit_rank > 0
        if not qr.evaluable:
            qr.strict_miss_reason = "unevaluable"
        elif not qr.strict_hit:
            qr.strict_miss_reason = _attribute_miss(docs, expected, k=5)

        # Top-K 明细（与主项目 evaluator 对齐，便于两版逐题对拍与人工复核）
        for rk, d in enumerate(docs[:10], 1):
            meta = d.metadata
            sc = meta.get("relevance_score", meta.get("score"))
            qr.top_k_details.append({
                "rank": rk,
                "drug_name": meta.get("drug_name", ""),
                "section": meta.get("section", ""),
                "score": round(float(sc), 4) if isinstance(sc, (int, float)) else None,
                "content_preview": (d.page_content or "")[:100].replace("\n", " "),
                "drug_exact": _drug_match_strict(
                    meta.get("drug_name", ""), expected, bool(meta.get("is_yinpian"))),
                "section_ok": _section_match(
                    d.page_content or "", meta.get("section", ""), expected_sections),
            })
        results.append(qr)

        if verbose and i % 10 == 0:
            print(f"  进度: {i}/{len(test_queries)}")

    n = len(results)
    hit1 = sum(r.first_hit_rank == 1 for r in results) / n
    hit3 = sum(0 < r.first_hit_rank <= 3 for r in results) / n
    hit5 = sum(0 < r.first_hit_rank <= 5 for r in results) / n
    mrr = sum(1 / r.first_hit_rank for r in results if r.hit) / n
    lats = [r.retrieval_latency for r in results]

    ev = [r for r in results if r.evaluable]
    m = len(ev)
    if m:
        s_hit1 = sum(r.strict_first_hit_rank == 1 for r in ev) / m
        s_hit3 = sum(0 < r.strict_first_hit_rank <= 3 for r in ev) / m
        s_hit5 = sum(0 < r.strict_first_hit_rank <= 5 for r in ev) / m
        s_mrr = sum(1 / r.strict_first_hit_rank for r in ev if r.strict_hit) / m
    else:
        s_hit1 = s_hit3 = s_hit5 = s_mrr = 0.0

    # 召回率（项目书口径）：期望药集合被 top-K 覆盖的比例，按题均值；不判章节
    _recall_rows = []
    for _r, _q in zip(results, test_queries):
        _exp = _q.get("expected_drugs") or []
        if not (_r.evaluable and _exp):
            continue
        _got = list(_r.retrieved_drugs or [])

        def _cov(_k: int) -> float:
            return sum(1 for _e in _exp
                       if any(_drug_match_strict(_g, [_e], _g.endswith("-饮片"))
                              for _g in _got[:_k])) / len(_exp)

        _recall_rows.append((_cov(1), _cov(3), _cov(5)))
    if _recall_rows:
        _nr = len(_recall_rows)
        recall_at_1 = sum(a for a, _, _ in _recall_rows) / _nr
        recall_at_3 = sum(b for _, b, _ in _recall_rows) / _nr
        recall_at_5 = sum(c for _, _, c in _recall_rows) / _nr
        full_recall_at_5 = sum(1 for _, _, c in _recall_rows if c >= 1.0) / _nr
        recall_queries = _nr
    else:
        recall_at_1 = recall_at_3 = recall_at_5 = full_recall_at_5 = 0.0
        recall_queries = 0

    by_type: Dict[str, dict] = {}
    for r in results:
        b = by_type.setdefault(r.query_type, {
            "count": 0, "evaluable": 0, "hits": 0, "strict_hits": 0,
            "rr": 0.0, "strict_rr": 0.0, "lat": 0.0})
        b["count"] += 1
        b["hits"] += 1 if r.hit else 0
        b["rr"] += 1 / r.first_hit_rank if r.hit else 0
        if r.evaluable:
            b["evaluable"] += 1
            b["strict_hits"] += 1 if r.strict_hit else 0
            b["strict_rr"] += 1 / r.strict_first_hit_rank if r.strict_hit else 0
        b["lat"] += r.retrieval_latency
        # 召回率（项目书口径，按题型）
        _exp = dict((q.get("id"), q.get("expected_drugs") or []) for q in test_queries).get(r.query_id) or []
        if r.evaluable and _exp:
            _got = [d or "" for d in (r.retrieved_drugs or [])][:5]
            b.setdefault("recall_sum", 0.0)
            b.setdefault("recall_n", 0)
            b["recall_sum"] += sum(1 for e in _exp
                                   if any(_drug_match_strict(g, [e], g.endswith("-饮片")) for g in _got)) / len(_exp)
            b["recall_n"] += 1
    for t, b in by_type.items():
        b["hit_at_5"] = b["hits"] / b["count"]
        b["mrr"] = b["rr"] / b["count"]
        b["strict_hit_at_5"] = (b["strict_hits"] / b["evaluable"]
                                if b["evaluable"] else None)
        b["strict_mrr"] = (b["strict_rr"] / b["evaluable"]
                           if b["evaluable"] else None)
        b["avg_latency"] = b["lat"] / b["count"]
        _rn = b.pop("recall_n", 0)
        b["recall_at_5"] = round(b.pop("recall_sum", 0.0) / _rn, 4) if _rn else None

    report = EvalReport(
        engine=engine,
        total_queries=n,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        hit_at_1=hit1, hit_at_3=hit3, hit_at_5=hit5, mrr=mrr,
        strict_hit_at_1=s_hit1, strict_hit_at_3=s_hit3, strict_hit_at_5=s_hit5,
        strict_mrr=s_mrr, evaluable_queries=m, unevaluable_queries=n - m,
        out_of_scope_queries=out_of_scope,
        recall_at_1=recall_at_1, recall_at_3=recall_at_3, recall_at_5=recall_at_5,
        full_recall_at_5=full_recall_at_5, recall_queries=recall_queries,
        latency_p50=_percentile(lats, 50),
        latency_p95=_percentile(lats, 95),
        latency_p99=_percentile(lats, 99),
        avg_latency=sum(lats) / n if n else 0,
        avg_doc_count=sum(r.doc_count for r in results) / n if n else 0,
        by_type=by_type,
        results=[asdict(r) for r in results],
    )
    return report


def load_test_queries(limit: int = None, query_type: str = None) -> List[Dict]:
    data = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    queries = data["queries"]
    if query_type:
        queries = [q for q in queries if q["type"] == query_type]
    if limit:
        queries = queries[:limit]
    return queries


# ============================================================
# 生成评估（口径与主项目 GenerationEvaluator 一致：
#   关键词覆盖率 = 命中的期望关键词占比；引用率；延迟分位）
# ============================================================

@dataclass
class GenQueryResult:
    query_id: str
    query: str
    query_type: str
    answer: str = ""
    answer_keywords_hit: int = 0
    answer_keywords_total: int = 0
    keyword_coverage: float = 0.0
    has_citations: bool = False
    has_consistency_issues: bool = False
    consistency_issue_count: int = 0
    latency: float = 0.0
    retrieval_latency: float = 0.0


@dataclass
class GenEvalReport:
    engine: str
    total_queries: int
    timestamp: str
    avg_keyword_coverage: float
    citation_rate: float
    consistency_issue_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    avg_latency: float
    by_type: Dict[str, dict] = field(default_factory=dict)
    results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_generation(service, test_queries: List[Dict], engine: str = "lc-standard",
                        verbose: bool = True) -> GenEvalReport:
    """逐题调用 service.answer（每题新会话，无跨题污染），计算生成质量指标"""
    results: List[GenQueryResult] = []
    for i, tc in enumerate(test_queries, 1):
        keywords = tc.get("expected_answer_keywords", []) or []
        t0 = time.time()
        try:
            out = service.answer(tc["query"])  # 不传 session_id → 每题独立会话
        except Exception as e:
            # 单题失败（通道超时/限流等）不报废整场评估：记 0 分继续
            import logging
            logging.getLogger(__name__).warning(f"题目 {tc.get('id')} 生成失败: {e}")
            out = {"answer": "", "citations": [], "consistency_issues": [],
                   "retrieval_latency": 0.0}
        latency = time.time() - t0

        answer = out.get("answer", "")
        # 先规范化再匹配，避免全角/波浪线差异造成假阴性（见 normalize_text）
        norm_answer = normalize_text(answer)
        hit = sum(1 for kw in keywords if kw and normalize_text(kw) in norm_answer)
        issues = out.get("consistency_issues", []) or []
        qr = GenQueryResult(
            query_id=tc["id"], query=tc["query"], query_type=tc["type"],
            answer=answer[:800],
            answer_keywords_hit=hit,
            answer_keywords_total=len(keywords),
            keyword_coverage=hit / len(keywords) if keywords else 0.0,
            has_citations=bool(out.get("citations")),
            has_consistency_issues=len(issues) > 0,
            consistency_issue_count=len(issues),
            latency=latency,
            retrieval_latency=out.get("retrieval_latency", 0.0),
        )
        results.append(qr)
        if verbose and i % 10 == 0:
            print(f"  进度: {i}/{len(test_queries)}")

    n = len(results)
    lats = [r.latency for r in results]
    by_type: Dict[str, dict] = {}
    for r in results:
        b = by_type.setdefault(r.query_type, {
            "count": 0, "cov": 0.0, "cite": 0, "issue": 0, "lat": 0.0,
        })
        b["count"] += 1
        b["cov"] += r.keyword_coverage
        b["cite"] += 1 if r.has_citations else 0
        b["issue"] += 1 if r.has_consistency_issues else 0
        b["lat"] += r.latency
    for t, b in by_type.items():
        b["avg_keyword_coverage"] = b["cov"] / b["count"]
        b["citation_rate"] = b["cite"] / b["count"]
        b["consistency_issue_rate"] = b["issue"] / b["count"]
        b["avg_latency"] = b["lat"] / b["count"]

    return GenEvalReport(
        engine=engine,
        total_queries=n,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        avg_keyword_coverage=sum(r.keyword_coverage for r in results) / n,
        citation_rate=sum(1 for r in results if r.has_citations) / n,
        consistency_issue_rate=sum(1 for r in results if r.has_consistency_issues) / n,
        latency_p50=_percentile(lats, 50),
        latency_p95=_percentile(lats, 95),
        latency_p99=_percentile(lats, 99),
        avg_latency=sum(lats) / n if n else 0,
        by_type=by_type,
        results=[asdict(r) for r in results],
    )
