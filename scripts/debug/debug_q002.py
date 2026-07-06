import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

# 从评估结果和测试集中获取Q002的信息
r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))

q002_eval = [q for q in r['results'] if q['query_id'] == 'Q002'][0]
q002_test = [q for q in t['queries'] if q['id'] == 'Q002'][0]

print(f"查询: {q002_eval['query']}")
print(f"期望药品: {q002_test[0]['expected_drugs']}")
print(f"检索药品: {q002_eval['retrieved_drugs']}")
print(f"命中: {q002_eval['hit']}")
print()

# 检查数据中是否有"傲"开头的药品
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
starts_with_ao = [c['drug_name'] for c in chunks if c['drug_name'].startswith('傲')]
print(f"以'傲'开头的药品: {starts_with_ao}")
print()

# 看看该查询的 Top 3 详情
details = q002_eval.get('top_k_details', [])[:3]
for d in details:
    print(f"  Rank {d['rank']}: {d['drug_name']} > {d['section']}")
    print(f"    {d['content_preview']}...")