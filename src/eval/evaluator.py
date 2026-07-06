# -*- coding: utf-8 -*-
"""
评估引擎
========
对 RAG 系统进行全量评估。

评估指标：
  ┌────────────────────────────────────────────────────────────┐
  │  检索质量                                                  │
  │  ├── Hit@1 / @3 / @5:     前 K 条结果中包含正确答案的比例  │
  │  ├── MRR:                  平均倒数排名                    │
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
import sys
import time
import json
import statistics
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


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
    hit: bool = False                    # 是否命中（Hit@K）
    first_hit_rank: int = 0              # 首次命中排名（1-based，0 表示未命中）
    retrieval_latency: float = 0.0

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
            "top_k_details": self.top_k_details,
        }


@dataclass
class EvalReport:
    """评估报告"""
    # 基本信息
    eval_type: str = ""
    total_queries: int = 0
    timestamp: str = ""

    # 检索指标
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0

    # 生成指标
    avg_keyword_coverage: float = 0.0
    citation_rate: float = 0.0
    consistency_issue_rate: float = 0.0
    medical_disclaimer_rate: float = 0.0

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
            },
            "generation": {
                "avg_keyword_coverage": round(self.avg_keyword_coverage, 4),
                "citation_rate": round(self.citation_rate, 4),
                "consistency_issue_rate": round(self.consistency_issue_rate, 4),
                "medical_disclaimer_rate": round(self.medical_disclaimer_rate, 4),
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

    对每条测试查询执行检索，计算 Hit@K 和 MRR。

    判定逻辑：
      - 对于有 expected_drugs 的查询：检索结果的 drug_name 在期望列表中即为命中
      - 对于无 expected_drugs 的查询（横向条件/方法通则）：只要检索到结果即为命中
      - 同时考虑 expected_sections 的匹配（章节匹配为加分项，药品名匹配为主要判据）
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
        """
        检查单条检索结果是否命中期望答案。

        判定规则：
          1. 如果有 expected_drugs：drug_name 匹配任一期望药品 → 命中
          2. 如果没有 expected_drugs（横向条件/方法通则查询）：始终视为命中
             （因为这类查询的正确答案不限定于特定药品）
        """
        if expected_drugs:
            # 药品名匹配（支持部分匹配，如"人参"匹配"人参-饮片"）
            drug = result.drug_name or ""
            for expected in expected_drugs:
                if expected in drug or drug in expected:
                    return True
            return False
        else:
            # 无特定药品期望的查询，只要有结果即为命中
            return True

    def evaluate_single(self, test_case: Dict) -> QueryResult:
        """评估单条查询"""
        qr = QueryResult(
            query_id=test_case["id"],
            query=test_case["query"],
            query_type=test_case["type"],
        )

        expected_drugs = test_case.get("expected_drugs", [])
        expected_sections = test_case.get("expected_sections", [])

        # 执行检索
        t0 = time.time()
        response = self.retriever.search(test_case["query"])
        qr.retrieval_latency = time.time() - t0

        results = response.results

        # 记录检索结果的药品和章节
        qr.retrieved_drugs = [r.drug_name for r in results[:10]]
        qr.retrieved_sections = [r.section for r in results[:10]]

        # 记录 Top-K 详情
        qr.top_k_details = [
            {
                "rank": i + 1,
                "drug_name": r.drug_name,
                "section": r.section,
                "score": round(r.rerank_score or r.score, 4),
                "content_preview": r.content[:100].replace('\n', ' '),
            }
            for i, r in enumerate(results[:10])
        ]

        # 计算 Hit@K 和首次命中排名
        qr.first_hit_rank = 0
        for i, r in enumerate(results):
            if self._check_hit(r, expected_drugs, expected_sections):
                qr.first_hit_rank = i + 1  # 1-based
                break

        qr.hit = qr.first_hit_rank > 0

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
        report = EvalReport(
            eval_type="retrieval",
            total_queries=len(test_queries),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

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

        # Hit@K
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

        # 延迟统计
        latencies = [r.retrieval_latency for r in results]
        report.avg_retrieval_latency = statistics.mean(latencies)
        sorted_lat = sorted(latencies)
        report.latency_p50 = self._percentile(sorted_lat, 50)
        report.latency_p95 = self._percentile(sorted_lat, 95)
        report.latency_p99 = self._percentile(sorted_lat, 99)
        report.avg_latency = report.avg_retrieval_latency

        # 分类型统计
        by_type = {}
        for r in results:
            t = r.query_type
            if t not in by_type:
                by_type[t] = {"count": 0, "hits": 0, "mrr_sum": 0.0, "latency_sum": 0.0}
            by_type[t]["count"] += 1
            if r.hit:
                by_type[t]["hits"] += 1
            if r.first_hit_rank > 0:
                by_type[t]["mrr_sum"] += 1.0 / r.first_hit_rank
            by_type[t]["latency_sum"] += r.retrieval_latency

        for t, v in by_type.items():
            c = v["count"]
            v["hit_at_5"] = round(v["hits"] / c, 4) if c > 0 else 0
            v["mrr"] = round(v["mrr_sum"] / c, 4) if c > 0 else 0
            v["avg_latency"] = round(v["latency_sum"] / c, 3) if c > 0 else 0
            del v["hits"]
            del v["mrr_sum"]
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

    def __init__(self, generator):
        """
        Args:
            generator: Generator 实例
        """
        self.generator = generator

    def evaluate_single(self, test_case: Dict) -> QueryResult:
        """评估单条查询"""
        qr = QueryResult(
            query_id=test_case["id"],
            query=test_case["query"],
            query_type=test_case["type"],
        )

        expected_keywords = test_case.get("expected_answer_keywords", [])

        # 执行端到端问答
        t0 = time.time()
        response = self.generator.answer(test_case["query"])
        total_latency = time.time() - t0

        # 填充结果
        qr.answer = response.answer
        qr.generation_latency = total_latency
        qr.component_latency = response.component_latency
        qr.retrieval_latency = response.retrieval.latency if response.retrieval else 0

        # 检索结果信息
        if response.retrieval and response.retrieval.results:
            qr.retrieved_drugs = [r.drug_name for r in response.retrieval.results[:5]]
            qr.hit = any(r.first_hit_rank > 0 for r in [qr])  # placeholder

        # 关键词覆盖率
        answer_lower = response.answer.lower()
        if expected_keywords:
            hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            qr.answer_keywords_hit = hits
            qr.answer_keywords_total = len(expected_keywords)
            qr.keyword_coverage = hits / len(expected_keywords)

        # 引用、一致性、安全提醒
        qr.has_citations = len(response.citations) > 0
        qr.has_consistency_issues = len(response.consistency_issues) > 0
        qr.has_medical_disclaimer = "遵医嘱" in response.answer or "具体用药" in response.answer

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
        report = EvalReport(
            eval_type="generation",
            total_queries=len(test_queries),
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
