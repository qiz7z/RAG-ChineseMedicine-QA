import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))
missed = [q for q in r['results'] if not q['hit']]

print(f"剩余未命中: {len(missed)}/100")
print()

# 统计失败类型
fail_types = {}
for q in missed:
    expected = q.get('expected_drugs', [])
    retrieved = q.get('retrieved_drugs', [])
    
    # 检查期望药品是否在检索结果中
    found = False
    for exp in expected:
        for ret in retrieved:
            if exp in ret or ret in exp:
                found = True
                break
        if found:
            break
    
    if expected:
        if found:
            fail_type = "药品找到但答案未匹配"
        else:
            fail_type = "检索未命中药品"
    else:
        fail_type = "横向/通用查询（无需特定药品）"
    
    fail_types[fail_type] = fail_types.get(fail_type, 0) + 1

from collections import Counter
print("失败类型分布:")
for t, c in Counter(fail_types).most_common():
    print(f"  {t}: {c} ({c/len(missed)*100:.1f}%)")
print()

# 详细分析"检索未命中药品"
print("=== 检索未命中药品的查询示例 ===")
count = 0
for q in missed:
    if q.get('expected_drugs') and any(d in ''.join(q.get('retrieved_drugs', [])) for d in q['expected_drugs']):
        # 这个查询的期望药品在检索结果中，但没有命中
        print(f"  {q['query_id']:6s} {q['query']}")
        print(f"    期望: {q['expected_drugs']}")
        print(f"    检索: {q['retrieved_drugs'][:3]}")
        count += 1
        if count >= 10:
            break

print(f"\n共 {count} 个查询是药品找到了但答案未匹配")

# 检查药品名识别失败
print("\n=== 检查药品名识别 ===")
no_drug_parsed = []
for q in missed:
    if q['query_type'] in ['单药品单属性查询', '单药品多属性查询']:
        if q['parsed'].get('drug_names', []):
            no_drug_parsed.append(f"{q['query_id']}: 识别到 {q['parsed']['drug_names']}")
        else:
            no_drug_parsed.append(f"{q['query_id']}: 未识别到药品名")

print(f"单药品查询中，{len(no_drug_parsed)} 个未识别药品名:")
for line in no_drug_parsed[:15]:
    print(f"  {line}")