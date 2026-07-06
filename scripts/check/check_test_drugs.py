import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))
print('有expected_drugs的查询:')
for q in t['queries']:
    if q.get('expected_drugs'):
        print(f'{q["id"]} {q["query"][:50]}...')
        print(f'  期望: {q["expected_drugs"]}')
        print()