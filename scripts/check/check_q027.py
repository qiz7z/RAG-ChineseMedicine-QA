import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))

q027 = [q for q in r['results'] if q['query_id'] == 'Q027'][0]

print("Q027:")
print(f"  检索药品: {q027['retrieved_drugs']}")
print(f"  hit: {q027['hit']}")
print(f"  first_hit_rank: {q027['first_hit_rank']}")
print()

# 检查 expected_drugs 是否正确匹配
top1_drug = q027['top_k_details'][0]['drug_name']
print(f"Top 1 drug: {top1_drug}")
print(f"命中判定: {top1_drug == '人参'}")