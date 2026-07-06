# -*- coding: utf-8 -*-
"""验证修复后的数据：检查关键药品名是否正确"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

with open('data/processed/drugs.json', 'r', encoding='utf-8') as f:
    entries = json.load(f)

with open('data/processed/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"总药品条目: {len(entries)}")
print(f"总切片数: {len(chunks)}")

# 检查关键药品
key_drugs = ['人参', '人 参', '黄芪', '黄 芪', '当归', '当 归', '丁香', '丁 香', 
             '干姜', '干 姜', '一捻金', '银黄口服液', '人工牛黄', '五味子']

for drug in key_drugs:
    drug_entries = [e for e in entries if e['drug_name'] == drug]
    drug_chunks = [c for c in chunks if c['drug_name'] == drug]
    if drug_entries or drug_chunks:
        print(f"\n  '{drug}':")
        print(f"    条目数: {len(drug_entries)}, 切片数: {len(drug_chunks)}")
        if drug_entries:
            e = drug_entries[0]
            print(f"    拼音: {e.get('pinyin_name', '')}")
            print(f"    拉丁: {e.get('latin_name', '')}")
            print(f"    章节: {[s['section_name'] for s in e.get('sections', [])[:5]]}")
            if e.get('intro_text'):
                print(f"    概述: {e['intro_text'][:80]}...")
    else:
        print(f"\n  '{drug}': 未找到 ❌")

# 检查之前错位的情况：人工牛黄是否还包含人参内容
print("\n=== 检查人工牛黄是否还包含人参内容 ===")
rg_nh_chunks = [c for c in chunks if c['drug_name'] == '人工牛黄']
for c in rg_nh_chunks:
    if '人参' in c['content']:
        print(f"  ❌ 人工牛黄仍包含人参内容: {c['content'][:80]}...")
    else:
        pass  # 正常
if not any('人参' in c['content'] for c in rg_nh_chunks):
    print("  ✅ 人工牛黄不再包含人参内容")

# 统计药品名中含空格的条目
spaced_names = [e for e in entries if ' ' in e['drug_name']]
print(f"\n含空格的药品名: {len(spaced_names)} 个")
for e in spaced_names[:10]:
    print(f"  {e['drug_name']} (拼音: {e.get('pinyin_name', '')})")

# 检查切片药品名分布
from collections import Counter
drug_chunk_counts = Counter(c['drug_name'] for c in chunks)
print(f"\n切片涉及药品数: {len(drug_chunk_counts)}")
print(f"切片最多的10个药品:")
for name, cnt in drug_chunk_counts.most_common(10):
    print(f"  {name}: {cnt} 个切片")
