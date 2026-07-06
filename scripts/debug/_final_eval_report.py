import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

# 修复后（100题）
r_after = json.load(open('data/eval/reports/eval_retrieval_20260706_094233.json', 'r', encoding='utf-8'))

print("=== 修复后评估结果（100题）===")
print(f'Hit@1: {r_after["retrieval"]["hit_at_1"]:.2%}')
print(f'Hit@3: {r_after["retrieval"]["hit_at_3"]:.2%}')
print(f'Hit@5: {r_after["retrieval"]["hit_at_5"]:.2%}')
print(f'MRR:   {r_after["retrieval"]["mrr"]:.4f}')
print(f'平均延迟: {r_after["performance"]["avg_latency"]:.3f}s')

print("\n=== 分类型表现 ===")
for t, v in sorted(r_after["by_type"].items(), key=lambda x: -x[1]["count"]):
    print(f'{t:20s} Hit@5={v["hit_at_5"]:>6.2%} MRR={v["mrr"]:>6.4f} (n={v["count"]})')

print("\n=== 未命中查询示例 ===")
missed = [r for r in r_after["results"] if not r["hit"]]
print(f'未命中: {len(missed)}/100 = {len(missed)/100:.0%}')
for r in missed[:15]:
    print(f'  {r["query_id"]:6s} {r["query_type"]:20s} {r["query"][:50]}...')
    if r.get("retrieved_drugs"):
        print(f'         实际检索到: {r["retrieved_drugs"][:3]}')