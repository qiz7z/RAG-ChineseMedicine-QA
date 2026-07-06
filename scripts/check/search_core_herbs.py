# -*- coding: utf-8 -*-
"""精确搜索核心药材在数据中的存在情况"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
all_drugs = sorted(set(c['drug_name'] for c in chunks))

# 核心药材列表
core_herbs = ['黄芪', '甘草', '川芎', '茯苓', '麻黄', '半夏', '柴胡', '地黄', '山药', '泽泻', '薏苡仁', '附子', '黄芩', '丹参', '白术', '桂枝', '生地', '熟地']

print("=== 核心药材在数据中的搜索结果 ===\n")
for herb in core_herbs:
    # 精确匹配
    exact = [d for d in all_drugs if d == herb or d.replace('-饮片', '') == herb]
    # 包含匹配
    contains = [d for d in all_drugs if herb in d]
    # 检查 chunk 内容中是否包含该药材名作为标题
    title_matches = [c for c in chunks if c.get('drug_name', '') == herb]
    
    print(f"【{herb}】")
    if exact:
        print(f"  精确匹配: {exact}")
    else:
        print(f"  精确匹配: 无")
    if contains:
        print(f"  包含匹配: {contains[:10]}")
    
    # 在所有 chunk 的文本中搜索该药材名（作为独立词出现）
    text_hits = 0
    for c in chunks[:100]:  # 只搜前100个chunk节省时间
        if herb in c.get('text', ''):
            text_hits += 1
    if text_hits > 0:
        print(f"  在前100个chunk文本中出现: {text_hits}次")
    print()

# 额外检查：数据中有多少以"黄"开头的药品
print("\n=== 以'黄'开头的药品 ===")
huang_drugs = [d for d in all_drugs if d.startswith('黄')]
for d in huang_drugs:
    print(f"  {d}")

print("\n=== 以'甘'开头的药品 ===")
gan_drugs = [d for d in all_drugs if d.startswith('甘')]
for d in gan_drugs:
    print(f"  {d}")

print("\n=== 以'川'开头的药品 ===")
chuan_drugs = [d for d in all_drugs if d.startswith('川')]
for d in chuan_drugs:
    print(f"  {d}")

print("\n=== 以'茯'开头的药品 ===")
fu_drugs = [d for d in all_drugs if d.startswith('茯')]
for d in fu_drugs:
    print(f"  {d}")

print("\n=== 以'麻'开头的药品 ===")
ma_drugs = [d for d in all_drugs if d.startswith('麻')]
for d in ma_drugs:
    print(f"  {d}")

print("\n=== 以'柴'开头的药品 ===")
chai_drugs = [d for d in all_drugs if d.startswith('柴')]
for d in chai_drugs:
    print(f"  {d}")

print("\n=== 以'半'开头的药品 ===")
ban_drugs = [d for d in all_drugs if d.startswith('半')]
for d in ban_drugs:
    print(f"  {d}")

print("\n=== 以'地'开头的药品 ===")
di_drugs = [d for d in all_drugs if d.startswith('地')]
for d in di_drugs:
    print(f"  {d}")

print("\n=== 以'山'开头的药品 ===")
shan_drugs = [d for d in all_drugs if d.startswith('山')]
for d in shan_drugs:
    print(f"  {d}")

print("\n=== 以'泽'开头的药品 ===")
ze_drugs = [d for d in all_drugs if d.startswith('泽')]
for d in ze_drugs:
    print(f"  {d}")

print("\n=== 以'附'开头的药品 ===")
fu2_drugs = [d for d in all_drugs if d.startswith('附')]
for d in fu2_drugs:
    print(f"  {d}")

print("\n=== 以'丹'开头的药品 ===")
dan_drugs = [d for d in all_drugs if d.startswith('丹')]
for d in dan_drugs:
    print(f"  {d}")

print("\n=== 以'白'开头的药品 ===")
bai_drugs = [d for d in all_drugs if d.startswith('白')]
for d in bai_drugs:
    print(f"  {d}")

print("\n=== 以'桂'开头的药品 ===")
gui_drugs = [d for d in all_drugs if d.startswith('桂')]
for d in gui_drugs:
    print(f"  {d}")

print("\n=== 以'薏'开头的药品 ===")
yi_drugs = [d for d in all_drugs if d.startswith('薏')]
for d in yi_drugs:
    print(f"  {d}")
