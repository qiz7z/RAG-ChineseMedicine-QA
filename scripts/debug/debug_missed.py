import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))

missed = [q for q in r['results'] if not q['hit']]
print(f"未命中: {len(missed)}\n")

for m in missed[:10]:
    query_id = m['query_id']
    # 找到对应的测试查询
    test_q = [q for q in t['queries'] if q['id'] == query_id]
    if test_q:
        print(f"{query_id} {m['query'][:50]}")
        print(f"  期望: {test_q[0]['expected_drugs']}")
        print(f"  检索: {m['retrieved_drugs'][:3]}")
        print()