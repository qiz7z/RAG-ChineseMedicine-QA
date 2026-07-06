import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
r = json.load(open('data/eval/reports/eval_retrieval_20260706_094233.json', 'r', encoding='utf-8'))
print("=== 修复后评估结果 ===")
print(f'Hit@1: {r["retrieval"]["hit_at_1"]:.2%}')
print(f'Hit@3: {r["retrieval"]["hit_at_3"]:.2%}')
print(f'Hit@5: {r["retrieval"]["hit_at_5"]:.2%}')
print(f'MRR: {r["retrieval"]["mrr"]:.4f}')

print("\n=== 修复前评估结果（从历史记录读取）===")
r_before = json.load(open('data/eval/reports/eval_retrieval_20260706_085538.json', 'r', encoding='utf-8'))
print(f'Hit@1: {r_before["retrieval"]["hit_at_1"]:.2%}')
print(f'Hit@3: {r_before["retrieval"]["hit_at_3"]:.2%}')
print(f'Hit@5: {r_before["retrieval"]["hit_at_5"]:.2%}')
print(f'MRR: {r_before["retrieval"]["mrr"]:.4f}')

print("\n=== 提升幅度 ===")
hit5_improvement = r["retrieval"]["hit_at_5"] - r_before["retrieval"]["hit_at_5"]
print(f'Hit@5: +{hit5_improvement:+.2%}')