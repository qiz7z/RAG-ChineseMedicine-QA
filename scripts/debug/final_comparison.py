import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))
print('=== 最终评估结果 (BM25过滤) ===')
print(f'Hit@1: {r["retrieval"]["hit_at_1"]:.2%}')
print(f'Hit@3: {r["retrieval"]["hit_at_3"]:.2%}')
print(f'Hit@5: {r["retrieval"]["hit_at_5"]:.2%}')
print(f'MRR:   {r["retrieval"]["mrr"]:.4f}')
print()
print('=== 对比修复前 ===')
r_before = json.load(open('data/eval/reports/eval_retrieval_20260706_101008.json', 'r', encoding='utf-8'))
print(f'Hit@1: {r_before["retrieval"]["hit_at_1"]:.2%}')
print(f'Hit@3: {r_before["retrieval"]["hit_at_3"]:.2%}')
print(f'Hit@5: {r_before["retrieval"]["hit_at_5"]:.2%}')
print(f'MRR:   {r_before["retrieval"]["mrr"]:.4f}')
print()
print('=== 对比第一轮修复前 ===')
r_first = json.load(open('data/eval/reports/eval_retrieval_20260706_085538.json', 'r', encoding='utf-8'))
print(f'Hit@5: {r_first["retrieval"].get("recall_at_5", 0):.2%} (仅10题参考)')