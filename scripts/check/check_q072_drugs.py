# -*- coding: utf-8 -*-
"""检查 Q072 和 Q073 中的其他药品是否存在"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
all_drugs = sorted(set(c['drug_name'] for c in chunks))
all_base = set(d.replace('-饮片', '') for d in all_drugs)

check = ['红花', '天南星', '西洋参', '连翘', '板蓝根', '酸枣仁', '远志', '柏子仁', '赤芍', '青皮', '薏苡仁']
for herb in check:
    exists = herb in all_base or herb in set(all_drugs)
    if exists:
        matches = [d for d in all_drugs if d == herb or d.replace('-饮片','') == herb]
        print(f"  ✓ {herb}: {matches}")
    else:
        contains = [d for d in all_drugs if herb in d]
        print(f"  ✗ {herb}: 不存在" + (f" (相似: {contains[:3]})" if contains else ""))
