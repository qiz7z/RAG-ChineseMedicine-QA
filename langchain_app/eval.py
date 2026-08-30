# -*- coding: utf-8 -*-
"""
独立评估器（标准版）
====================
判分公式与主项目 evaluator.py 完全一致（保证横向可比）：
  - 有 expected_drugs：检索结果的 drug_name 与期望药品做双向子串匹配
  - 无 expected_drugs：检索到任何结果即判命中（已知口径偏宽松，
    三方评估共用同一公式时相对可比）
指标：Hit@1/3/5、MRR、延迟 P50/P95/P99、分类型聚合。
"""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict

from config import TEST_SET_PATH


@dataclass
class QueryResult:
    query_id: str
    query: str
    query_type: str
    retrieval_latency: float = 0.0
    retrieved_drugs: List[str] = field(default_factory=list)
    first_hit_rank: int = 0
    hit: bool = False


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
    by_type: Dict[str, dict] = field(default_factory=dict)
    results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(len(values) * p / 100), len(values) - 1)
    return values[idx]


def _check_hit(drug_name: str, expected_drugs: List[str]) -> bool:
    """与主项目 _check_hit 相同的双向子串匹配；无期望药品自动命中"""
    if expected_drugs:
        drug = drug_name or ""
        return any(e in drug or drug in e for e in expected_drugs)
    return True


def evaluate_retrieval(retriever, test_queries: List[Dict], engine: str = "lc-standard",
                       verbose: bool = True) -> EvalReport:
    results: List[QueryResult] = []
    for i, tc in enumerate(test_queries, 1):
        expected = tc.get("expected_drugs", []) or []
        t0 = time.time()
        docs = retriever.invoke(tc["query"])
        latency = time.time() - t0

        qr = QueryResult(
            query_id=tc["id"], query=tc["query"], query_type=tc["type"],
            retrieval_latency=latency,
            retrieved_drugs=[d.metadata.get("drug_name", "") for d in docs[:10]],
        )
        qr.first_hit_rank = 0
        for rank, d in enumerate(docs, 1):
            if _check_hit(d.metadata.get("drug_name", ""), expected):
                qr.first_hit_rank = rank
                break
        qr.hit = qr.first_hit_rank > 0
        results.append(qr)

        if verbose and i % 10 == 0:
            print(f"  进度: {i}/{len(test_queries)}")

    n = len(results)
    hit1 = sum(r.first_hit_rank == 1 for r in results) / n
    hit3 = sum(0 < r.first_hit_rank <= 3 for r in results) / n
    hit5 = sum(0 < r.first_hit_rank <= 5 for r in results) / n
    mrr = sum(1 / r.first_hit_rank for r in results if r.hit) / n
    lats = [r.retrieval_latency for r in results]

    by_type: Dict[str, dict] = {}
    for r in results:
        bucket = by_type.setdefault(r.query_type, {"count": 0, "hits": 0, "rr": 0.0, "lat": 0.0})
        bucket["count"] += 1
        bucket["hits"] += 1 if r.hit else 0
        bucket["rr"] += 1 / r.first_hit_rank if r.hit else 0
        bucket["lat"] += r.retrieval_latency
    for t, b in by_type.items():
        b["hit_at_5"] = b["hits"] / b["count"]
        b["mrr"] = b["rr"] / b["count"]
        b["avg_latency"] = b["lat"] / b["count"]

    report = EvalReport(
        engine=engine,
        total_queries=n,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        hit_at_1=hit1, hit_at_3=hit3, hit_at_5=hit5, mrr=mrr,
        latency_p50=_percentile(lats, 50),
        latency_p95=_percentile(lats, 95),
        latency_p99=_percentile(lats, 99),
        avg_latency=sum(lats) / n if n else 0,
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
