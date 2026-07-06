import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf_8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))

missed = [q for q in r['results'] if not q['hit']]

# Q002
q002_eval = [q for q in r['results'] if q['query_id'] == 'Q002'][0]
q002_test = [q for q in t['queries'] if q['id'] == 'Q002'][0]

print(f"Q002")
print(f"  查询: {q002_eval['query']}")
print(f"  期望: {q002_test[0]['expected_drugs']}")
print(f"  检索: {q002_eval['retrieved_drugs']}")
print()

# 检查数据中是否有黄芩
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
has_huangqi = [c for c in chunks if '黄芩' in c['drug_name']]
print(f"数据中含'黄芩'的药品数: {len(has_huangqi)}")
if has_huangqi:
    print(f"  示例: {[c['drug_name'] for c in has_huangqi[:5]]}")