import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
drugs = sorted(set(c['drug_name'] for c in chunks))

# 查找可能的正确药品名
queries = {
    '傲槖': ['云芝', '薏苡仁', '虎杖'],
    '芫梃': ['川楝子', '毛诃子'],
    '猫爪草': [],
    '大枣': ['大枣'],
    '连翘': ['连翘'],
    '黄芩': ['黄芩'],
    '黄芪': [],  # 检查是否存在
    '黄连': ['黄连'],
}

print("=== 检查失败查询的期望药品是否在数据中 ===")
for wrong_drug, possible in queries.items():
    print(f"{wrong_drug} → 在数据中: {wrong_drug in drugs}")
    if wrong_drug in drugs:
        print(f"  ✓ 找到")
    else:
        print(f"  ✗ 未找到")
        print(f"  可能是: {[d for d in drugs if wrong_drug in d or d in wrong_drug][:3]}")