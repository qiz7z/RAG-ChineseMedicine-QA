# -*- coding: utf-8 -*-
"""
LangChain 标准版评估运行脚本
=============================
用法:
  # 完整混合检索评估（不需要 LLM 额度）
  python scripts/run/run_eval_lc.py

  # 消融实验
  python scripts/run/run_eval_lc.py --no-rerank     # 去掉 CrossEncoder 重排
  python scripts/run/run_eval_lc.py --no-bm25       # 纯向量（对照 baseline）

  # 子集 / 按类型
  python scripts/run/run_eval_lc.py --limit 10 --type "横向条件查询"
"""
import sys
import io
import json
import time
import argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'langchain_app'))


def main():
    parser = argparse.ArgumentParser(description="LangChain 标准版检索评估")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 CrossEncoder 重排（消融）")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25 路（纯向量对照）")
    parser.add_argument("--limit", type=int, default=0, help="测试题数量限制（0=全部）")
    parser.add_argument("--type", type=str, default="", help="按问题类型筛选")
    args = parser.parse_args()

    from retrievers import build_hybrid_retriever
    from eval import evaluate_retrieval, load_test_queries
    from config import REPORT_DIR

    engine_name = "lc-std"
    if args.no_rerank:
        engine_name += "-noRerank"
    if args.no_bm25:
        engine_name += "-noBM25"

    test_queries = load_test_queries(
        limit=args.limit if args.limit > 0 else None,
        query_type=args.type if args.type else None,
    )

    print("=" * 70)
    print(f"  LangChain 标准版检索评估 | 引擎: {engine_name} | 题数: {len(test_queries)}")
    print("=" * 70)

    retriever = build_hybrid_retriever(
        enable_reranker=not args.no_rerank,
        enable_bm25=not args.no_bm25,
    )
    report = evaluate_retrieval(retriever, test_queries, engine=engine_name)

    print()
    print(f"  Hit@1: {report.hit_at_1:.2%} | Hit@3: {report.hit_at_3:.2%} | "
          f"Hit@5: {report.hit_at_5:.2%} | MRR: {report.mrr:.4f}")
    print(f"  延迟 P50: {report.latency_p50:.3f}s | P95: {report.latency_p95:.3f}s")
    print("  分类型:")
    for t, v in sorted(report.by_type.items(), key=lambda x: -x[1]["count"]):
        print(f"    {t}: n={v['count']} hit@5={v['hit_at_5']:.2%} mrr={v['mrr']:.4f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"eval_{engine_name}_{ts}.json"
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已保存: {out}")


if __name__ == "__main__":
    main()
