# -*- coding: utf-8 -*-
"""
评估运行脚本
=============
对 RAG 系统进行全量评估，生成评估报告。

用法:
  # 检索评估（不需要 LLM API Key）
  python scripts/run/run_eval.py --mode retrieval

  # 生成评估（需要 LLM API Key）
  python scripts/run/run_eval.py --mode generation

  # 全量评估（检索 + 生成）
  python scripts/run/run_eval.py --mode all

  # 指定测试集子集（前 10 题，快速验证）
  python scripts/run/run_eval.py --mode retrieval --limit 10

  # 指定问题类型
  python scripts/run/run_eval.py --mode retrieval --type "单药品单属性查询"

环境变量:
  LONGCAT_API_KEY   - 美团 LongCat API Key（生成评估必需）
"""
import sys
import io
import os
import json
import time
import argparse
from pathlib import Path

# 修复 Windows 控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# 路径配置
TEST_SET_PATH = PROJECT_ROOT / 'data' / 'eval' / 'test_queries.json'
REPORT_DIR = PROJECT_ROOT / 'data' / 'eval' / 'reports'


def load_test_queries(limit: int = None, query_type: str = None) -> list:
    """加载测试集"""
    with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = data['queries']

    # 按类型筛选
    if query_type:
        queries = [q for q in queries if q['type'] == query_type]
        if not queries:
            print(f"[ERROR] 未找到类型为 '{query_type}' 的测试题")
            print(f"  可用类型: {set(q['type'] for q in data['queries'])}")
            sys.exit(1)

    # 数量限制
    if limit and limit > 0:
        queries = queries[:limit]

    return queries


def print_retrieval_report(report):
    """打印检索评估报告"""
    print()
    print("=" * 70)
    print("  检索质量评估报告")
    print("=" * 70)
    print()
    print(f"  测试题数:  {report.total_queries}")
    print(f"  评估时间:  {report.timestamp}")
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                    检索质量指标                          │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Hit@1:  {report.hit_at_1:>8.2%}   {'✓' if report.hit_at_1 >= 0.70 else '✗'} (目标: ≥70%)          │")
    print(f"  │  Hit@3:  {report.hit_at_3:>8.2%}   {'✓' if report.hit_at_3 >= 0.80 else '✗'} (目标: ≥80%)          │")
    print(f"  │  Hit@5:  {report.hit_at_5:>8.2%}   {'✓' if report.hit_at_5 >= 0.90 else '✗'} (目标: ≥90%)          │")
    print(f"  │  MRR:       {report.mrr:>8.4f}   {'✓' if report.mrr >= 0.80 else '✗'} (目标: ≥0.80)        │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print("  │                    性能指标                              │")
    print(f"  │  平均延迟:  {report.avg_latency:.3f}s                           │")
    print(f"  │  P50 延迟:  {report.latency_p50:.3f}s                           │")
    print(f"  │  P95 延迟:  {report.latency_p95:.3f}s                           │")
    print(f"  │  P99 延迟:  {report.latency_p99:.3f}s                           │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()

    # 分类型结果
    if report.by_type:
        print("  分类型检索结果:")
        print(f"  {'类型':<20s} {'数量':>4s} {'Hit@5':>10s} {'MRR':>8s} {'延迟':>8s}")
        print(f"  {'─'*20} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")
        for t, v in sorted(report.by_type.items(), key=lambda x: -x[1]['count']):
            print(f"  {t:<20s} {v['count']:>4d} {v['hit_at_5']:>10.2%} {v['mrr']:>8.4f} {v['avg_latency']:>7.3f}s")
    print()

    # 未命中查询
    missed = [r for r in report.results if not r.hit]
    if missed:
        print(f"  未命中查询 ({len(missed)} 条):")
        for r in missed:
            print(f"    {r.query_id} [{r.query_type}] {r.query}")
            if r.retrieved_drugs:
                print(f"      实际检索到: {r.retrieved_drugs[:3]}")
        print()


def print_generation_report(report):
    """打印生成评估报告"""
    print()
    print("=" * 70)
    print("  生成质量评估报告")
    print("=" * 70)
    print()
    print(f"  测试题数:  {report.total_queries}")
    print(f"  评估时间:  {report.timestamp}")
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                    生成质量指标                          │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  关键词覆盖率:       {report.avg_keyword_coverage:>8.2%}   {'✓' if report.avg_keyword_coverage >= 0.70 else '✗'} (目标: ≥70%)    │")
    print(f"  │  引用率:             {report.citation_rate:>8.2%}   {'✓' if report.citation_rate >= 0.80 else '✗'} (目标: ≥80%)    │")
    print(f"  │  一致性问题率:       {report.consistency_issue_rate:>8.2%}   {'✓' if report.consistency_issue_rate <= 0.10 else '✗'} (目标: ≤10%)   │")
    print(f"  │  安全提醒率:         {report.medical_disclaimer_rate:>8.2%}                      │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print("  │                    性能指标                              │")
    print(f"  │  端到端平均延迟:     {report.avg_latency:.3f}s                        │")
    print(f"  │  P50 延迟:           {report.latency_p50:.3f}s                        │")
    print(f"  │  P95 延迟:           {report.latency_p95:.3f}s   {'✓' if report.latency_p95 <= 5.0 else '✗'} (目标: ≤5s)      │")
    print(f"  │  P99 延迟:           {report.latency_p99:.3f}s                        │")
    print(f"  │  平均检索延迟:       {report.avg_retrieval_latency:.3f}s                        │")
    print(f"  │  平均 LLM 延迟:      {report.avg_llm_latency:.3f}s                        │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()

    # 分类型结果
    if report.by_type:
        print("  分类型生成结果:")
        print(f"  {'类型':<20s} {'数量':>4s} {'关键词覆盖':>10s} {'引用率':>8s} {'延迟':>8s}")
        print(f"  {'─'*20} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")
        for t, v in sorted(report.by_type.items(), key=lambda x: -x[1]['count']):
            print(f"  {t:<20s} {v['count']:>4d} {v['avg_keyword_coverage']:>10.2%} {v['citation_rate']:>8.2%} {v['avg_latency']:>7.3f}s")
    print()

    # 低覆盖率查询
    low_cov = [r for r in report.results if r.keyword_coverage < 0.3]
    if low_cov:
        print(f"  低覆盖率查询 ({len(low_cov)} 条, coverage < 30%):")
        for r in low_cov[:10]:
            print(f"    {r.query_id} [{r.query_type}] cov={r.keyword_coverage:.2f} {r.query[:40]}")
        if len(low_cov) > 10:
            print(f"    ... 还有 {len(low_cov) - 10} 条")
    print()

    # 一致性问题
    issues = [r for r in report.results if r.has_consistency_issues]
    if issues:
        print(f"  存在一致性问题的回答 ({len(issues)} 条):")
        for r in issues[:5]:
            print(f"    {r.query_id} {r.query[:40]}")
    print()


def save_report(report, mode: str):
    """保存评估报告到 JSON 文件"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{mode}_{timestamp}.json"
    filepath = REPORT_DIR / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"  报告已保存: {filepath}")
    return filepath


def run_retrieval_eval(test_queries):
    """运行检索评估"""
    from retrieval.retriever import Retriever
    from eval.evaluator import RetrievalEvaluator

    print()
    print("=" * 70)
    print("  初始化检索引擎...")
    print("=" * 70)

    retriever = Retriever()

    print()
    print(f"  开始检索评估 ({len(test_queries)} 题)...")
    print()

    evaluator = RetrievalEvaluator(retriever)
    report = evaluator.evaluate(test_queries, verbose=True)

    print_retrieval_report(report)
    save_report(report, "retrieval")

    retriever.close()
    return report


def run_generation_eval(test_queries):
    """运行生成评估"""
    from generation.generator import Generator
    from eval.evaluator import GenerationEvaluator

    # 检查 API Key
    api_key = os.environ.get("LONGCAT_API_KEY", "")
    if not api_key:
        print("[ERROR] 生成评估需要 LONGCAT_API_KEY 环境变量！")
        print("  PowerShell: $env:LONGCAT_API_KEY=\"your_api_key\"")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  初始化生成引擎（检索 + LLM）...")
    print("=" * 70)

    generator = Generator()

    print()
    print(f"  开始生成评估 ({len(test_queries)} 题)...")
    print(f"  注意: 每题约需 3-8 秒，总计约需 {len(test_queries) * 5 // 60} 分钟")
    print()

    evaluator = GenerationEvaluator(generator)
    report = evaluator.evaluate(test_queries, verbose=True)

    print_generation_report(report)
    save_report(report, "generation")

    return report


def main():
    parser = argparse.ArgumentParser(description="药典 RAG 系统评估工具")
    parser.add_argument(
        "--mode", choices=["retrieval", "generation", "all"],
        default="retrieval",
        help="评估模式: retrieval=仅检索, generation=仅生成, all=全量 (默认: retrieval)",
    )
    parser.add_argument("--limit", type=int, default=0, help="测试题数量限制（0=全部）")
    parser.add_argument("--type", type=str, default="", help="按问题类型筛选")
    args = parser.parse_args()

    print("=" * 70)
    print("  中国药典智能问答系统 - 评估工具")
    print("=" * 70)

    # 加载测试集
    test_queries = load_test_queries(
        limit=args.limit if args.limit > 0 else None,
        query_type=args.type if args.type else None,
    )
    print(f"  测试集: {TEST_SET_PATH}")
    print(f"  测试题数: {len(test_queries)}")

    # 运行评估
    if args.mode in ("retrieval", "all"):
        run_retrieval_eval(test_queries)

    if args.mode in ("generation", "all"):
        run_generation_eval(test_queries)

    print()
    print("=" * 70)
    print("  评估完成！")
    print(f"  报告目录: {REPORT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
