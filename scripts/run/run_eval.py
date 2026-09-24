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
    print(f"  │  [loose 粗召回口径，全 {report.total_queries:>3d} 题]                    │")
    print(f"  │    Hit@1:  {report.hit_at_1:>8.2%}   {'✓' if report.hit_at_1 >= 0.70 else '✗'} (目标: ≥70%)          │")
    print(f"  │    Hit@3:  {report.hit_at_3:>8.2%}   {'✓' if report.hit_at_3 >= 0.80 else '✗'} (目标: ≥80%)          │")
    print(f"  │    Hit@5:  {report.hit_at_5:>8.2%}   {'✓' if report.hit_at_5 >= 0.90 else '✗'} (目标: ≥90%)          │")
    print(f"  │    MRR:       {report.mrr:>8.4f}   {'✓' if report.mrr >= 0.80 else '✗'} (目标: ≥0.80)        │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  [strict 严格口径，可评测 {report.evaluable_queries:>3d} 题]                  │")
    print(f"  │    Hit@1:  {report.strict_hit_at_1:>8.2%}   (药品名精确 + 章节真实出现)      │")
    print(f"  │    Hit@3:  {report.strict_hit_at_3:>8.2%}                                    │")
    print(f"  │    Hit@5:  {report.strict_hit_at_5:>8.2%}   ← 对外报数用这一行              │")
    print(f"  │    MRR:       {report.strict_mrr:>8.4f}                                  │")
    print(f"  │    不可评测题: {report.unevaluable_queries:>2d} 道（expected_drugs 为空，已剔出分母）     │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print("  │                    性能指标                              │")
    print(f"  │  平均延迟:  {report.avg_latency:.3f}s                           │")
    print(f"  │  P50 延迟:  {report.latency_p50:.3f}s                           │")
    print(f"  │  P95 延迟:  {report.latency_p95:.3f}s                           │")
    print(f"  │  P99 延迟:  {report.latency_p99:.3f}s                           │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()

    # 召回率（项目书口径：Recall@5 ≥ 90%）
    print(f"  召回率（项目书口径）: @1 {report.recall_at_1:.2%} | @3 {report.recall_at_3:.2%} "
          f"| @5 {report.recall_at_5:.2%} {'✓' if report.recall_at_5 >= 0.90 else '✗'}"
          f"   (目标 ≥90%)")
    print(f"    ├ 全召回@5: {report.full_recall_at_5:.2%}（期望药**全部**进 top-5 的题占比）"
          f" | 可评测 {report.recall_queries} 题 | 药名精确匹配、**不判章节**")
    print()

    # 分类型结果
    if report.by_type:
        print("  分类型检索结果（strict 列只统计该类型的可评测题）:")
        print(f"  {'类型':<20s} {'数量':>4s} {'loose@5':>9s} {'strict@5':>9s} {'MRR':>8s} {'延迟':>8s}")
        print(f"  {'─'*20} {'─'*4} {'─'*9} {'─'*9} {'─'*8} {'─'*8}")
        for t, v in sorted(report.by_type.items(), key=lambda x: -x[1]['count']):
            s = v.get('strict_hit_at_5')
            s_txt = f"{s:>9.2%}" if s is not None else f"{'n/a':>9s}"
            print(f"  {t:<20s} {v['count']:>4d} {v['hit_at_5']:>9.2%} {s_txt} {v['mrr']:>8.4f} {v['avg_latency']:>7.3f}s")
    print()

    # strict 未命中归因（对症下药：覆盖 / 判定 / 切片）
    strict_missed = [r for r in report.results if r.evaluable and not r.strict_hit]
    if strict_missed:
        from collections import Counter
        reason_cn = {
            "drug_not_recalled": "药品未进前5（覆盖问题）",
            "drug_not_exact": "召回的是含该药的别条目（子串放大器）",
            "section_missing": "药品对了但章节不符（切片/排序）",
        }
        cnt = Counter(r.strict_miss_reason for r in strict_missed)
        print(f"  strict 未命中 {len(strict_missed)} 道，归因:")
        for k, c in cnt.most_common():
            print(f"    {c:>3d}  {k}  {reason_cn.get(k, '')}")
        for r in strict_missed:
            print(f"    {r.query_id} [{r.query_type}] {r.query}")
            print(f"       期望药品/章节 → 实际 top1: {r.retrieved_drugs[:1]} / {r.retrieved_sections[:1]}"
                  f"   归因={r.strict_miss_reason}")
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
    print(f"  │  引用率⚠️恒真:       {report.citation_rate:>8.2%}   （引用由检索结果机械拼出，非质量指标）  │")
    print(f"  │  数值一致性检出:     {report.consistency_issue_rate:>8.2%}   （只查带单位数值，是幻觉率下界）  │")
    print(f"  │  安全提醒率:         {report.medical_disclaimer_rate:>8.2%}                      │")
    g = getattr(report, "grounding", None) or {}
    if g:
        print("  ├─────────────────────────────────────────────────────────┤")
        print("  │        ★ 有据性判分（LLM-as-judge，对外应引用这一组）     │")
        print(f"  │  幻觉率:             {g.get('hallucination_rate', 0):>8.2%}   (无据+矛盾)/论断数            │")
        print(f"  │    ├ 无据(编造):     {g.get('unsupported_rate', 0):>8.2%}   {g.get('unsupported', 0):>4d}/{g.get('claims_total', 0):<4d} 条论断            │")
        print(f"  │    └ 与来源矛盾:     {g.get('contradiction_rate', 0):>8.2%}   {g.get('contradicted', 0):>4d}/{g.get('claims_total', 0):<4d} 条论断            │")
        print(f"  │  回答有据率:         {g.get('grounded_answer_rate', 0):>8.2%}   {g.get('judged_answers', 0)} 条回答中完全无编造/矛盾   │")
        print(f"  │  论断有据率:         {g.get('citation_support_rate', 0):>8.2%}   {g.get('supported', 0):>4d}/{g.get('claims_total', 0):<4d} 条论断（替代恒真引用率）│")
        if g.get("excluded_judge_errors"):
            print(f"  │  ⚠️ 判定失败被排除:  {g['excluded_judge_errors']:>4d} 题（不计入以上比例）            │")
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


def _rerank_tag(explicit: bool = None) -> str:
    """当前重排配置的标签，写进报告文件名。

    为什么需要：src 的**开重排与关重排报告此前同名**（`eval_retrieval_2026…`），
    事后只能靠时间戳猜，极易把消融数字当成主结果（2026-09-24 就误读了一次，
    差点把「开重排 93.58%」当成 README 采用的「关重排 91.74%」）。
    lc 侧早有 `noRerank`/`std` 区分，src 侧补齐。
    """
    from config import ENABLE_RERANKER
    eff = ENABLE_RERANKER if explicit is None else explicit
    return "rerank" if eff else "noRerank"


def save_report(report, mode: str, config_note: str = ""):
    """保存评估报告到 JSON 文件（`config_note` 会写进文件名，如 `noRerank`）"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{config_note}" if config_note else ""
    filename = f"eval_{mode}{tag}_{timestamp}.json"
    filepath = REPORT_DIR / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"  报告已保存: {filepath}")
    return filepath


def run_retrieval_eval(test_queries, enable_reranker=None):
    """运行检索评估

    Args:
        enable_reranker: None=用 config 默认值；False=关重排（消融对照）
    """
    from retrieval.retriever import Retriever
    from eval.evaluator import RetrievalEvaluator

    print()
    print("=" * 70)
    print("  初始化检索引擎...")
    print("=" * 70)

    retriever = Retriever(enable_reranker=enable_reranker)
    if enable_reranker is not None:
        print(f"  [消融] 重排开关 = {enable_reranker}")

    print()
    print(f"  开始检索评估 ({len(test_queries)} 题)...")
    print()

    evaluator = RetrievalEvaluator(retriever)
    report = evaluator.evaluate(test_queries, verbose=True)

    print_retrieval_report(report)
    save_report(report, "retrieval", _rerank_tag(enable_reranker))

    retriever.close()
    return report


def run_generation_eval(test_queries, grounding: bool = True, crosscheck: bool = False):
    """运行生成评估

    Args:
        grounding: 是否额外做**有据性判分**（LLM-as-judge，每题多一次 LLM 调用）。
                   产出真实的幻觉率 / 回答有据率，替代恒真的 citation_rate
                   与只查数值的 consistency_issue_rate。见 src/eval/grounding_judge.py。
    """
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

    judge = None
    cc_judge = None
    if grounding:
        try:
            from generation.llm_client import LLMClient
            from eval.grounding_judge import GroundingJudge
            judge = GroundingJudge(LLMClient())
            print(f"  有据性判分: 已开启（判官模型 {judge.model_name}，每题额外 1 次 LLM 调用）")
            print("              产出 hallucination_rate / grounded_answer_rate / citation_support_rate")
        except Exception as e:                     # noqa: BLE001
            print(f"  [WARN] 有据性判分初始化失败，已跳过: {type(e).__name__}: {e}")

    if crosscheck and grounding:
        from eval.grounding_judge import QuoteVerifiedJudge
        cc_judge = QuoteVerifiedJudge(LLMClient())
        print(f"  第二判官(引文核验): 已开启（每题再 +1 次 LLM 调用）——用于交叉验证第一判官")
    evaluator = GenerationEvaluator(generator, grounding_judge=judge,
                                    crosscheck_judge=cc_judge)
    report = evaluator.evaluate(test_queries, verbose=True)

    print_generation_report(report)
    save_report(report, "generation", _rerank_tag())

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
    parser.add_argument("--no-rerank", action="store_true",
                        help="关闭 CrossEncoder 重排（消融对照；默认沿用 config.ENABLE_RERANKER）")
    parser.add_argument("--crosscheck", action="store_true",
                        help="生成评估时加开**引文核验式第二判官**（每题 +1 次 LLM 调用）。"
                             "机制与第一判官不同：要求判官逐字抄出支撑原文，代码再核验引文真伪，"
                             "用于交叉验证「回答有据率」结论是否可靠。")
    parser.add_argument("--no-grounding", action="store_true",
                        help="生成评估时关闭有据性判分（LLM-as-judge）。"
                             "默认开启——它产出真实的幻觉率与回答有据率，"
                             "而老的 citation_rate 是恒真指标、consistency_issue_rate 只是数值型下界")
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
        run_retrieval_eval(
            test_queries,
            enable_reranker=False if args.no_rerank else None,
        )

    if args.mode in ("generation", "all"):
        run_generation_eval(test_queries, grounding=not args.no_grounding,
                            crosscheck=args.crosscheck)

    print()
    print("=" * 70)
    print("  评估完成！")
    print(f"  报告目录: {REPORT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
