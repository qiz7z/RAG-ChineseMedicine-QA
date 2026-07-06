# -*- coding: utf-8 -*-
"""
检索引擎测试脚本
=================
验证完整的检索流程：查询解析 → 向量检索 + BM25 → RRF融合 → 重排 → 上下文组装

用法:
  python scripts/test/test_retrieval.py
"""
import sys
import io
import time
from pathlib import Path

# 修复 Windows 控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from retrieval.retriever import Retriever


def main():
    print("=" * 60)
    print("  药典 RAG 系统 - 检索引擎测试（阶段三）")
    print("=" * 60)

    # 初始化检索引擎
    print("\n初始化检索引擎...\n")
    retriever = Retriever()

    # 测试查询集（覆盖不同查询类型）
    test_cases = [
        {
            "query": "人参的性味归经是什么？",
            "description": "属性查询 - 精确药品名 + 明确章节",
            "expected_drug": "人参",
        },
        {
            "query": "双黄连口服液的含量测定方法",
            "description": "检测查询 - 复方制剂 + 含量测定",
            "expected_drug": "双黄连口服液",
        },
        {
            "query": "哪些药材含有挥发油成分？",
            "description": "成分查询 - 无具体药品名，全库检索",
            "expected_drug": None,
        },
        {
            "query": "高效液相色谱法的操作步骤",
            "description": "方法查询 - 检测方法类问题",
            "expected_drug": None,
        },
        {
            "query": "黄芪和当归一起用的功效",
            "description": "功效查询 - 多药品名",
            "expected_drug": None,
        },
        {
            "query": "六味地黄丸怎么服用？",
            "description": "用法查询 - 成方制剂 + 用法用量",
            "expected_drug": "六味地黄丸",
        },
    ]

    total_start = time.time()

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_cases)}: {case['description']}")
        print(f"查询: '{query}'")
        print("-" * 60)

        # 执行检索
        response = retriever.search(query)

        # 输出查询解析结果
        parsed = response.parsed
        print(f"\n📋 查询解析:")
        print(f"   药品名: {parsed['drug_names']}")
        print(f"   章节:   {parsed['sections']}")
        print(f"   意图:   {parsed['intent']}")
        print(f"   关键词: {parsed['keywords']}")
        print(f"   过滤:   {parsed['filter']}")

        # 输出检索结果
        print(f"\n📊 检索结果 ({len(response.results)} 条):")
        for j, r in enumerate(response.results, 1):
            score_type = "rerank" if r.rerank_score is not None else "rrf"
            score = r.rerank_score if r.rerank_score is not None else r.score
            sources_str = "+".join(r.sources) if r.sources else "?"
            print(f"   {j}. [{score_type}={score:.4f}] [{sources_str}]")
            print(f"      药品: {r.drug_name} | 章节: {r.section} | 分类: {r.category}")
            print(f"      内容: {r.content[:100].replace(chr(10), ' ')}...")

        # 输出耗时
        print(f"\n⏱ 耗时: {response.latency:.3f}s")
        for component, latency in response.component_latency.items():
            print(f"   {component}: {latency:.3f}s")

        # 输出上下文预览
        print(f"\n📝 上下文预览 (前300字):")
        context_preview = response.context[:300].replace('\n', ' ')
        print(f"   {context_preview}...")

    # 总结
    print(f"\n{'='*60}")
    print(f"  测试完成!")
    print(f"  总耗时: {time.time() - total_start:.1f}s")
    print(f"  测试用例: {len(test_cases)} 个")
    print(f"{'='*60}")

    retriever.close()


if __name__ == '__main__':
    main()
