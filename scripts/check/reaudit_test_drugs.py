# -*- coding: utf-8 -*-
"""重新检查测试集 expected_drugs 与修复后数据的匹配情况"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
all_drugs = sorted(set(c['drug_name'] for c in chunks))
all_drug_set = set(all_drugs)
all_base_drugs = set(d.replace('-饮片', '') for d in all_drugs)

t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))

print("=== 修复后测试集 expected_drugs 匹配情况 ===\n")

still_missing = []
for q in t['queries']:
    expected = q.get('expected_drugs', [])
    for drug in expected:
        if drug in all_drug_set or drug in all_base_drugs:
            continue
        
        # 药品仍不存在
        # 查找数据中包含该药品名的药品
        candidates = [d for d in all_drugs if drug in d or d.replace('-饮片','') in drug]
        # 查找查询中包含的药品
        query = q['query']
        query_matches = [d for d in all_base_drugs if len(d) >= 2 and d in query]
        
        print(f"{q['id']} {query[:50]}")
        print(f"  期望: {drug}  [仍不存在]")
        if candidates:
            print(f"  数据中相似: {candidates[:5]}")
        if query_matches:
            print(f"  查询中包含: {query_matches}")
        
        # 智能建议
        suggested = None
        if query_matches:
            # 如果期望药品不在查询中但查询中有其他存在的药品，选择查询中的
            if drug not in query:
                suggested = query_matches[0]
            else:
                # 期望药品在查询中但不存在于数据，查找最相似的
                if candidates:
                    suggested = candidates[0].replace('-饮片', '')
        elif candidates:
            suggested = candidates[0].replace('-饮片', '')
        
        if suggested:
            print(f"  建议修正: {drug} → {suggested}")
        else:
            print(f"  无法自动修正")
        
        still_missing.append({
            'id': q['id'],
            'query': query,
            'old': drug,
            'new': suggested,
            'candidates': candidates[:3],
            'query_matches': query_matches,
        })
        print()

print(f"\n=== 仍需修正 {len(still_missing)} 处 ===")
print(json.dumps(still_missing, ensure_ascii=False, indent=2))
