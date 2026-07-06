import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))
m = [q for q in r['results'] if not q['hit']]
print('未命中查询示例:')
for q in m[:5]:
    print(f'{q["query_id"]} {q["query"]}')
    print(f'  期望: {q.get("expected_drugs", "无")}')
    print(f'  检索: {q.get("retrieved_drugs", [])}')
    print()