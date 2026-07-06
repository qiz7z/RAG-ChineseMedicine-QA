import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
drugs = sorted(set(c['drug_name'] for c in chunks))

queries = {
    '傲槖': [],
    '芫梃': [],
    '猫爪草': [],
    '大枣': ['大枣'],
    '连翘': ['连翘'],
    '黄芩': ['黄芩'],
    '黄芪': [],  # 检查是否存在
    '黄连': ['黄连'],
}

print("=== 检查失败查询的期望药品是否在数据中 ===")
for wrong_drug, possible in queries.items():
    in_data = wrong_drug in drugs
    print(f"{wrong_drug} 在数据中: {in_data}")
    if in_data:
        print(f"  找到")
    else:
        print(f"  未找到")
    if possible:
        print(f"  可能是: {possible}")