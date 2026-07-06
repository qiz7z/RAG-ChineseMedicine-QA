# -*- coding: utf-8 -*-
"""验证修复后的数据是否包含核心药材"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
all_drugs = sorted(set(c['drug_name'] for c in chunks))

core_herbs = ['黄芪', '甘草', '川芎', '茯苓', '麻黄', '半夏', '柴胡', '地黄', '山药', '泽泻', '薏苡仁', '附子', '黄芩', '丹参', '白术', '桂枝', '土茯苓']

print("=== 核心药材验证 ===\n")
found = 0
for herb in core_herbs:
    exact = [d for d in all_drugs if d == herb or d.replace('-饮片', '') == herb]
    if exact:
        print(f"  ✓ {herb}: {exact}")
        found += 1
    else:
        # 模糊搜索
        contains = [d for d in all_drugs if herb in d]
        if contains:
            print(f"  ~ {herb}: 精确匹配无，包含: {contains[:5]}")
        else:
            print(f"  ✗ {herb}: 不存在")

print(f"\n找到: {found}/{len(core_herbs)}")

# 检查是否还有 OCR 错误的药品名
print("\n=== 检查 OCR 错误残留 ===")
ocr_errors = ['黄茂', '黄苓', '黄苔', '川弯', '川夸', '川菖', '川萼', '主茯苓']
for err in ocr_errors:
    matches = [d for d in all_drugs if err in d]
    if matches:
        print(f"  ✗ 仍有 OCR 错误 '{err}': {matches}")
    else:
        print(f"  ✓ '{err}' 已纠正")

print(f"\n总药品数: {len(all_drugs)}")
