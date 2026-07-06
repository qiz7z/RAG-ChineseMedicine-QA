"""深入分析修复后仍失败的查询"""
import json, re

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

r = json.load(open('data/eval/reports/eval_retrieval_20260706_101008.json', 'r', encoding='utf-8'))
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))

all_drugs = sorted(set(c['drug_name'] for c in chunks))
all_drug_set = set(all_drugs)

# 所有未命中查询
missed = [q for q in r['results'] if not q['hit']]
print(f"未命中: {len(missed)}/100\n")

# 分析每个失败查询
for q in missed:
    query = q['query']
    retrieved = q.get('retrieved_drugs', [])
    
    # 尝试从查询中提取药品名
    # 简单方法：在所有药品名中搜索
    matched_drugs = []
    for drug in all_drugs:
        base = drug.replace('-饮片', '')
        if base and len(base) >= 2 and base in query:
            matched_drugs.append(drug)
    
    # 判断失败类型
    if matched_drugs:
        # 药品名在查询中存在，但检索没找到
        # 检查检索结果中是否有正确的药品
        correct_in_retrieved = any(d in matched_drugs for d in retrieved)
        if correct_in_retrieved:
            # 药品找到了但答案没匹配 - 可能是章节不匹配
            fail_type = "章节不匹配"
        else:
            # 药品完全没找到
            fail_type = "检索未命中药品"
    else:
        # 查询中的药品名不在数据中
        fail_type = "药品不在数据中"
    
    print(f"{q['query_id']:6s} [{fail_type}] {query[:50]}")
    print(f"       期望药品: {matched_drugs[:3] if matched_drugs else '未找到'}")
    print(f"       实际检索: {retrieved[:3]}")
    print()

# 统计失败类型
from collections import Counter
fail_types = []
for q in missed:
    query = q['query']
    retrieved = q.get('retrieved_drugs', [])
    matched_drugs = []
    for drug in all_drugs:
        base = drug.replace('-饮片', '')
        if base and len(base) >= 2 and base in query:
            matched_drugs.append(drug)
    
    if matched_drugs:
        correct_in_retrieved = any(d in matched_drugs for d in retrieved)
        if correct_in_retrieved:
            fail_types.append("章节不匹配")
        else:
            fail_types.append("检索未命中药品")
    else:
        fail_types.append("药品不在数据中")

print("\n=== 失败类型统计 ===")
for t, c in Counter(fail_types).most_common():
    print(f"  {t}: {c} ({c/len(missed)*100:.1f}%)")
