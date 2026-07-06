"""分析检索失败的根本原因"""
import json

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

# 加载评估结果
r = json.load(open('data/eval/reports/eval_retrieval_20260706_094233.json', 'r', encoding='utf-8'))
missed = [q for q in r['results'] if not q['hit']]

print("=== 检索失败的根本原因分析 ===")
print(f"\n未命中: {len(missed)}/100 = {len(missed)}%\n")

# 统计失败模式
patterns = {
    '药品名含空格不匹配': 0,
    '药品名/别名不匹配': 0,
    '药品不在文档中': 0,
    '其他': 0
}

# 加载所有药品名
import json
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
all_drugs = sorted(set(c['drug_name'] for c in chunks))

for q in missed:
    query = q['query']
    # 提取药品名（简单规则：找到第一个2-4字的中文词）
    import re
    drug_match = re.search(r'([^\s,，？?！!]{2,4}[药味苷])', query)
    if not drug_match:
        drug_match = re.search(r'([^\s,，？?！!]{2,4})', query)

    if drug_match:
        query_drug = drug_match.group(1)
        # 检查带空格和不带空格的版本
        spaced = f"{query_drug[0]} {query_drug[1:]}" if len(query_drug) >= 2 else query_drug
        spaced2 = ' '.join(query_drug)

        # 检查检索到的药品
        retrieved = q.get('retrieved_drugs', [])

        # 模式判断
        if spaced in all_drugs or spaced2 in all_drugs:
            patterns['药品名含空格不匹配'] += 1
        elif any(query_drug in d or d in query_drug for d in all_drugs):
            # 药品存在但别名不匹配
            patterns['药品名/别名不匹配'] += 1
        else:
            patterns['药品不在文档中'] += 1
    else:
        patterns['其他'] += 1

print("失败模式分布:")
for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
    print(f"  {pattern}: {count} ({count/len(missed)*100:.1f}%)")

# 具体示例
print("\n=== 失败示例分析 ===")
print("\n1. 药品名含空格不匹配:")
examples = [q for q in missed if any(d in q['query'] for d in ['人参', '当归', '黄芪'])]
for q in examples[:3]:
    print(f"  问题: {q['query']}")
    print(f"  检索到: {q.get('retrieved_drugs', [])[:3]}")

print("\n2. 检查数据中的实际药品名（前20个）:")
print(f"  {all_drugs[:20]}")

# 统计含空格的药品名比例
spaced_drugs = [d for d in all_drugs if ' ' in d and not d.endswith('-饮片')]
print(f"\n含空格的药品名: {len(spaced_drugs)}/{len(all_drugs)} = {len(spaced_drugs)/len(all_drugs)*100:.1f}%")
if spaced_drugs:
    print(f"  示例: {spaced_drugs[:10]}")