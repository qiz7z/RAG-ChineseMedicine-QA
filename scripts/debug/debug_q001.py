import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))

# 查看Q001
q001 = [q for q in r['results'] if q['query_id'] == 'Q001'][0]
print(f"查询: {q001['query']}")
print(f"检索到的药品: {q001['retrieved_drugs'][:5]}")
print(f"命中: {q001['hit']}")
print(f"首次命中排名: {q001['first_hit_rank']}")
print()

# 查看期望药品
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))
q002_test = [q for q in t['queries'] if q['id'] == 'Q002'][0]
print(f"Q002 期望药品: {q002_test['expected_drugs']}")