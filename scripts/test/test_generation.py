# -*- coding: utf-8 -*-
"""
生成引擎测试脚本
=================
端到端验证完整 RAG 流程：检索 → LLM 生成 → 后处理

测试内容：
  1. 单轮问答（属性查询、检测方法查询等）
  2. 多轮对话（指代消解）
  3. 流式输出
  4. 各阶段耗时分析

用法:
  # 先设置 API Key 环境变量
  set LONGCAT_API_KEY=your_api_key_here
  python scripts/test/test_generation.py

  # 或在 PowerShell 中
  $env:LONGCAT_API_KEY="your_api_key_here"
  python scripts/test/test_generation.py
"""
import sys
import io
import time
import os
from pathlib import Path

# 修复 Windows 控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def check_api_key():
    """检查 API Key 是否设置"""
    api_key = os.environ.get("LONGCAT_API_KEY", "")
    if not api_key:
        print("=" * 60)
        print("❌ 错误: 未设置 LONGCAT_API_KEY 环境变量！")
        print("=" * 60)
        print("\n请设置美团 LongCat API Key 后再运行：")
        print("\n  CMD:")
        print('    set LONGCAT_API_KEY=your_api_key_here')
        print("\n  PowerShell:")
        print('    $env:LONGCAT_API_KEY="your_api_key_here"')
        print("\n获取 API Key: https://longcat.chat/platform/api_keys")
        print()
        sys.exit(1)
    return api_key


def main():
    check_api_key()

    from generation.generator import Generator

    print("=" * 60)
    print("  药典 RAG 系统 - 生成引擎测试（阶段四）")
    print("  LLM: 美团 LongCat 2.0")
    print("=" * 60)

    # 初始化生成引擎
    print("\n初始化生成引擎...\n")
    generator = Generator()

    # ============================================================
    # 测试一：单轮问答
    # ============================================================
    print("\n" + "━" * 60)
    print("  测试一：单轮问答")
    print("━" * 60)

    single_turn_queries = [
        {
            "query": "人参的性味归经是什么？",
            "description": "属性查询 - 精确药品 + 明确章节",
        },
        {
            "query": "双黄连口服液的含量测定方法是什么？",
            "description": "检测查询 - 复方制剂 + 含量测定",
        },
        {
            "query": "黄芪有什么功效？",
            "description": "功效查询 - 单味药材",
        },
        {
            "query": "六味地黄丸怎么服用？",
            "description": "用法查询 - 成方制剂",
        },
    ]

    for i, case in enumerate(single_turn_queries, 1):
        query = case["query"]
        print(f"\n{'─' * 60}")
        print(f"问题 {i}: {case['description']}")
        print(f"用户: {query}")
        print(f"{'─' * 60}")

        response = generator.answer(query)

        print(f"\n🤖 回答:")
        print(response.answer)

        print(f"\n📎 引用来源:")
        for j, citation in enumerate(response.citations, 1):
            print(f"   [{j}] {citation}")

        if response.consistency_issues:
            print(f"\n⚠️ 一致性检查:")
            for issue in response.consistency_issues:
                print(f"   - {issue}")

        print(f"\n⏱ 耗时分析:")
        print(f"   总耗时: {response.latency:.3f}s")
        for component, latency in response.component_latency.items():
            print(f"   {component}: {latency:.3f}s")
        if response.retrieval:
            print(f"   检索结果数: {len(response.retrieval.results)}")

    # ============================================================
    # 测试二：多轮对话（指代消解）
    # ============================================================
    print("\n" + "━" * 60)
    print("  测试二：多轮对话（指代消解）")
    print("━" * 60)

    # 清空对话历史，重新开始
    generator.clear_dialogue()

    dialogue_turns = [
        "人参的性味归经是什么？",
        "那它的功能主治呢？",           # "它" 应消解为 "人参"
        "用法用量是怎样的？",           # 上下文延续
    ]

    for i, query in enumerate(dialogue_turns, 1):
        print(f"\n{'─' * 60}")
        print(f"第 {i} 轮对话")
        print(f"👤 用户: {query}")
        print(f"{'─' * 60}")

        response = generator.answer(query)

        print(f"📝 消解后查询: {response.resolved_query}")
        if response.resolved_query != query:
            print(f"   (指代消解: '{query}' → '{response.resolved_query}')")

        print(f"\n🤖 回答:")
        print(response.answer)

        print(f"\n⏱ 耗时: {response.latency:.3f}s | 对话轮次: {response.dialogue_turn}")

    # ============================================================
    # 测试三：流式输出
    # ============================================================
    print("\n" + "━" * 60)
    print("  测试三：流式输出")
    print("━" * 60)

    stream_query = "当归的性味和功效是什么？"
    print(f"\n👤 用户: {stream_query}")
    print(f"\n🤖 回答 (流式):")

    generator.clear_dialogue()
    print()
    for chunk in generator.answer_stream(stream_query):
        print(chunk, end="", flush=True)
    print("\n")

    # ============================================================
    # 总结
    # ============================================================
    print("━" * 60)
    print("  ✅ 生成引擎测试完成!")
    print(f"  LLM 模型: {generator.llm_client.model}")
    print("━" * 60)

    generator.close()


if __name__ == '__main__':
    main()
