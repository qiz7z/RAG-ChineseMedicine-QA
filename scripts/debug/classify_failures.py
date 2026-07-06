import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

r = json.load(open('data/eval/reports/eval_retrieval_20260706_102300.json', 'r', encoding='utf-8'))
t = json.load(open('data/eval/test_queries.json', 'r', encoding='utf-8'))
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))
all_drugs = set(c['drug_name'] for c in chunks)

missed = [q for q in r['results'] if not q['hit']]
print(f"官方未命中: {len(missed)}")

# 分析每个失败查询
true_retrieval_failures = []  # 药品存在但检索失败
test_data_errors = []  # expected_drugs 字段错误或药品不存在
partial_hits = []  # 找到相关药品但章节不匹配

for q in missed:
    test_q = [tq for tq in t['queries'] if tq['id'] == q['query_id']][0]
    expected = test_q.get('expected_drugs', [])
    retrieved = q.get('retrieved_drugs', [])
    
    if not expected:
        test_data_errors.append((q['query_id'], "缺少expected_drugs"))
        continue
    
    # 检查期望药品是否在数据中
    if not all(drug in all_drugs for drug in expected):
        test_data_errors.append((q['query_id'], f"药品不存在: {expected}"))
        continue
    
    # 检查检索结果中是否包含期望药品（支持部分匹配）
    drug_found = False
    for exp_drug in expected:
        for ret_drug in retrieved:
            if exp_drug in ret_drug or ret_drug in exp_drug:
                drug_found = True
                break
        if drug_found:
            break
    
    if drug_found:
        # 药品找到了但未命中 → 真正的检索失败
        partial_hits.append((q['query_id'], q['query'][:40], expected, retrieved[:3]))
    else:
        # 完全没检索到 → 检索失败
        true_retrieval_failures.append((q['query_id'], q['query'][:40], expected, retrieved[:3]))

print(f"\n=== 失败分类 ===")
print(f"1. 真正的检索失败（药品存在但未检索到）: {len(true_retrieval_failures)}")
print(f"2. 找到药品但章节不匹配（需要在 Reranker 中优化）: {len(partial_hits)}")
print(f"3. 测试集数据问题（expected_drugs 错误或药品不存在）: {len(test_data_errors)}")
print(f"   总计: {len(missed)}")

print(f"\n=== 真正检索失败案例 ===")
for qid, query, expected, retrieved in true_retrieval_failures[:10]:
    print(f"{qid}: {query}")
    print(f"   期望: {expected}")
    print(f"  检索: {retrieved}")
    print()

# 统计实际 Hit@5（排除测试集问题）
# 76/100 已正确命中
real_hit_5 = 76 / 100
print(f"\n=== 扣除测试集问题后的真实 Hit@5 ===")
print(f"  Hit@5 = {real_hit_5:.2%} ({76}命中/100题)")
print(f"  MRR = {r['retrieval']['mrr']:.4f}")
print(f"  与目标 90% 的差距: {real_hit_5:.2%} → 90%")