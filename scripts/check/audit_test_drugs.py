# -*- coding: utf-8 -*-
"""全面排查测试集 expected_drugs 与实际数据的匹配情况"""
import json, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

# 加载数据
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
all_drugs = sorted(set(c['drug_name'] for c in chunks))
all_drug_set = set(all_drugs)
all_base_drugs = set(d.replace('-饮片', '') for d in all_drugs)

# 加载测试集
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))

print("=== 测试集 expected_drugs 与数据匹配情况 ===\n")

fixes = []
for q in t['queries']:
    expected = q.get('expected_drugs', [])
    query = q['query']
    
    for drug in expected:
        if drug in all_drug_set or drug in all_base_drugs:
            continue  # 药品存在，无需修正
        
        # 药品不存在，查找正确的药品名
        # 策略1：在查询中搜索含该药品名的子串
        # 策略2：在数据中搜索含该药品名的药品
        candidates = []
        
        # 搜索数据中包含期望药品名的药品（可能是子串）
        for d in all_drugs:
            base = d.replace('-饮片', '')
            if drug in base or base in drug:
                candidates.append(d)
        
        # 搜索查询中是否包含其他药品名
        query_matches = []
        for d in all_base_drugs:
            if len(d) >= 2 and d in query:
                query_matches.append(d)
        
        print(f"{q['id']} {query[:50]}")
        print(f"  期望: {drug}  [不存在于数据]")
        if candidates:
            print(f"  数据中相似: {candidates[:5]}")
        if query_matches:
            print(f"  查询中包含的药品: {query_matches}")
        
        # 自动建议修正
        suggested = None
        if query_matches:
            # 优先选择查询中出现的最短匹配（更精确）
            suggested = query_matches[0]
        elif candidates:
            suggested = candidates[0].replace('-饮片', '')
        
        if suggested:
            print(f"  建议修正: {drug} → {suggested}")
            fixes.append({
                'id': q['id'],
                'old': drug,
                'new': suggested,
            })
        else:
            print(f"  无法自动修正，需要手动检查")
            fixes.append({
                'id': q['id'],
                'old': drug,
                'new': None,
            })
        print()

print(f"\n=== 共需修正 {len(fixes)} 处 ===")
print(json.dumps(fixes, ensure_ascii=False, indent=2))
