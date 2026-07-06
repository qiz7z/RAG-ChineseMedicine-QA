import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
all_drugs = set(c['drug_name'] for c in chunks)

# 分析几个具体失败案例
failed_cases = ['Q002', 'Q031', 'Q026', 'Q027', 'Q042', 'Q067', 'Q075']

print("=== 失败案例分析 ===")
for case_id in failed_cases:
    # 从测试集获取期望
    test_q = [q for q in t['queries'] if q['id'] == case_id][0]
    # 从评估结果获取
    eval_q = [q for q in r['results'] if q['query_id'] == case_id][0]
    
    print(f"\n{case_id} {test_q['query'][:60]}")
    print(f"  期望: {test_q['expected_drugs']}")
    print(f"  检索: {eval_q['retrieved_drugs'][:3]}")
    print(f"  数据中存在: {test_q['expected_drugs'][0] in all_drugs}")
    
    if test_q['expected_drugs'][0] not in all_drugs:
        # 查找相似的药品名
        similar = [d for d in sorted(all_drugs) if test_q['expected_drugs'][0] in d or d in test_q['expected_drugs'][0]]
        print(f"  相似药品: {similar[:5]}")