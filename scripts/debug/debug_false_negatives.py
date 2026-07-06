import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))

# Q027
q027 = [q for q in r['results'] if q['query_id'] == 'Q027'][0]
print("Q027 分析:")
print(f"  查询: {q027['query']}")
print(f"  期望: {q027['top_k_details'][0]['drug_name']}")
print(f"  检索药品: {q027['retrieved_drugs']}")
print(f"  hit: {q027['hit']}")
print(f"  first_hit_rank: {q027['first_hit_rank']}")
print()

# 检查期望药品
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))
q027_test = [q for q in t['queries'] if q['id'] == 'Q027'][0]
print(f"  期望药品: {q027_test[0]['expected_drugs']}")
print()

# Q067
q067 = [q for q in r['results'] if q['query_id'] == 'Q067'][0]
print("Q067 分析:")
print(f"  查询: {q067['query']}")
print(f"  检索药品: {q067['retrieved_drugs']}")
print(f"  hit: {q067['hit']}")
print(f"  first_hit_rank: {q067['first_hit_rank']}")
print()

# 检查期望药品
q067_test = [q for q in t['queries'] if q['id'] == 'Q067'][0]
print(f"  期望药品: {q067_test[0]['expected_drugs']}")
print()

# Q002
q002 = [q for q in r['results'] if q['query_id'] == 'Q002'][0]
print("Q002 分析:")
print(f"  查询: {q002['query']}")
print(f"  期望药品: {q002_test[0]['expected_drugs']}")
print(f"  数据中是否有: False")
print(f"  检索结果为空: {len(q002['retrieved_drugs']) == 0}")
print()