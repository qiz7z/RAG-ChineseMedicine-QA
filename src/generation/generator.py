# -*- coding: utf-8 -*-
"""
生成引擎（主模块）
===================
整合检索引擎 + LLM 生成 + 后处理，提供端到端的问答能力。

完整生成流程：
  ┌──────────────────────────────────────────────────────────────┐
  │  用户查询："人参的性味归经是什么？"                            │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  1. 指代消解（DialogueManager）                               │
  │     多轮对话时：将"它"→"人参"                                 │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  2. 检索（Retriever）                                         │
  │     查询解析 → 向量+BM25 → RRF → 重排 → 上下文组装            │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  3. Prompt 构建（Prompts）                                    │
  │     System Prompt + 药典参考资料 + 对话历史 + 用户问题         │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  4. LLM 生成（LLMClient → LongCat 2.0）                      │
  │     基于检索上下文生成专业回答                                 │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  5. 后处理（PostProcessor）                                   │
  │     引用标注 + 一致性校验 + 格式美化 + 安全提醒                │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  6. 更新对话历史（DialogueManager）                           │
  └──────────────────────────────────────────────────────────────┘

使用方式：
    from generation.generator import Generator

    generator = Generator()
    response = generator.answer("人参的性味归经是什么？")
    print(response.answer)
    print(response.citations)
"""
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    ENABLE_CITATION,
    ENABLE_CONSISTENCY_CHECK,
    ENABLE_FORMAT_BEAUTIFY,
)
from retrieval.retriever import Retriever, RetrievalResponse
from generation.llm_client import LLMClient
from generation.prompts import build_prompt, build_no_result_prompt
from generation.postprocessor import PostProcessor, PostProcessResult
from generation.dialogue import DialogueManager

logger = logging.getLogger(__name__)


# ============================================================
# 响应数据结构
# ============================================================

@dataclass
class GenerationResponse:
    """
    生成引擎完整响应。

    包含从查询到生成的全部信息。
    """
    # 核心结果
    query: str                                    # 原始用户查询
    resolved_query: str                           # 指代消解后的查询
    answer: str                                   # 最终回答（经后处理）

    # 检索信息
    retrieval: Optional[RetrievalResponse] = None # 检索响应

    # 生成信息
    raw_answer: str = ""                          # LLM 原始回答（未后处理）
    citations: List[str] = field(default_factory=list)         # 引用来源
    consistency_issues: List[str] = field(default_factory=list) # 一致性问题

    # 元信息
    latency: float = 0.0                          # 总耗时（秒）
    component_latency: Dict[str, float] = field(default_factory=dict)  # 各阶段耗时
    llm_model: str = ""                           # 使用的 LLM 模型
    dialogue_turn: int = 0                        # 对话轮次

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "resolved_query": self.resolved_query,
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "citations": self.citations,
            "consistency_issues": self.consistency_issues,
            "latency": self.latency,
            "component_latency": self.component_latency,
            "llm_model": self.llm_model,
            "dialogue_turn": self.dialogue_turn,
            "retrieval_results": (
                [r.to_dict() for r in self.retrieval.results]
                if self.retrieval else []
            ),
            "retrieval_latency": self.retrieval.latency if self.retrieval else 0,
        }


# ============================================================
# 生成引擎
# ============================================================

class Generator:
    """
    生成引擎：整合检索 + LLM 生成 + 后处理。

    使用方式：
        generator = Generator()
        response = generator.answer("人参的性味归经是什么？")
        print(response.answer)

    多轮对话：
        generator = Generator()
        generator.answer("人参的性味归经是什么？")
        response = generator.answer("那它的功能主治呢？")  # "它"自动消解为"人参"
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[Retriever] = None,
        enable_dialogue: bool = True,
        enable_postprocess: bool = True,
    ):
        """
        Args:
            llm_client: LLM 客户端实例（如不提供则自动创建）
            retriever: 检索引擎实例（如不提供则自动创建）
            enable_dialogue: 是否启用多轮对话管理
            enable_postprocess: 是否启用回答后处理
        """
        print("=" * 60)
        print("初始化生成引擎")
        print("=" * 60)

        self.enable_dialogue = enable_dialogue
        self.enable_postprocess = enable_postprocess

        # 1. 初始化检索引擎
        if retriever:
            self.retriever = retriever
        else:
            t0 = time.time()
            self.retriever = Retriever()
            print(f"  检索引擎初始化完成 ({time.time() - t0:.1f}s)")

        # 2. 初始化 LLM 客户端
        if llm_client:
            self.llm_client = llm_client
        else:
            t0 = time.time()
            self.llm_client = LLMClient()
            print(f"  LLM 客户端初始化完成 ({time.time() - t0:.1f}s)")

        # 3. 初始化后处理器
        self.postprocessor = PostProcessor()

        # 4. 初始化对话管理器
        self.dialogue_manager = DialogueManager(
            llm_client=self.llm_client if enable_dialogue else None,
        )

        print("=" * 60)
        print("生成引擎初始化完成!")
        print("=" * 60)

    # ----------------------------------------------------------
    # 核心方法：问答
    # ----------------------------------------------------------

    def answer(
        self,
        query: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResponse:
        """
        端到端问答：查询 → 检索 → 生成 → 后处理。

        Args:
            query: 用户查询
            temperature: LLM 温度参数（默认从 config 读取）
            max_tokens: LLM 最大输出 token 数

        Returns:
            GenerationResponse 对象
        """
        total_start = time.time()
        component_latency = {}

        # ============ 1. 指代消解 ============
        t0 = time.time()
        resolved_query = query
        if self.enable_dialogue and self.dialogue_manager.turn_count > 0:
            resolved_query = self.dialogue_manager.resolve_query(query)
        component_latency["coreference_resolution"] = time.time() - t0

        # ============ 2. 检索 ============
        t0 = time.time()
        retrieval_response = self.retriever.search(resolved_query)
        component_latency["retrieval"] = time.time() - t0

        # ============ 3. Prompt 构建 ============
        t0 = time.time()
        retrieval_results = [r.to_dict() for r in retrieval_response.results]

        if retrieval_results:
            chat_history = (
                self.dialogue_manager.get_history_messages()
                if self.enable_dialogue else None
            )
            messages = build_prompt(
                user_query=resolved_query,
                retrieval_results=retrieval_results,
                chat_history=chat_history,
            )
        else:
            # 无检索结果时使用空结果 Prompt
            messages = build_no_result_prompt(resolved_query)
        component_latency["prompt_building"] = time.time() - t0

        # ============ 4. LLM 生成 ============
        t0 = time.time()
        raw_answer = self.llm_client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        component_latency["llm_generation"] = time.time() - t0

        # ============ 5. 后处理 ============
        t0 = time.time()
        if self.enable_postprocess and retrieval_results:
            pp_result = self.postprocessor.process(raw_answer, retrieval_results)
            final_answer = pp_result.answer
            citations = pp_result.citations
            consistency_issues = pp_result.consistency_issues
        else:
            final_answer = raw_answer
            citations = []
            consistency_issues = []
        component_latency["post_processing"] = time.time() - t0

        # ============ 6. 更新对话历史 ============
        if self.enable_dialogue:
            self.dialogue_manager.add_user_message(query, resolved_query)
            self.dialogue_manager.add_assistant_message(final_answer)

        total_latency = time.time() - total_start

        return GenerationResponse(
            query=query,
            resolved_query=resolved_query,
            answer=final_answer,
            retrieval=retrieval_response,
            raw_answer=raw_answer,
            citations=citations,
            consistency_issues=consistency_issues,
            latency=total_latency,
            component_latency=component_latency,
            llm_model=self.llm_client.model,
            dialogue_turn=self.dialogue_manager.turn_count,
        )

    # ----------------------------------------------------------
    # 流式生成
    # ----------------------------------------------------------

    def answer_stream(
        self,
        query: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        流式问答：逐 token 返回 LLM 生成内容。

        注意：流式模式下不进行后处理和对话历史更新。
        后处理需要调用方在获取完整回答后手动执行。

        Args:
            query: 用户查询
            temperature: LLM 温度参数
            max_tokens: LLM 最大输出 token 数

        Yields:
            LLM 生成的文本片段
        """
        # 1. 指代消解
        resolved_query = query
        if self.enable_dialogue and self.dialogue_manager.turn_count > 0:
            resolved_query = self.dialogue_manager.resolve_query(query)

        # 2. 检索
        retrieval_response = self.retriever.search(resolved_query)
        retrieval_results = [r.to_dict() for r in retrieval_response.results]

        # 3. Prompt 构建
        if retrieval_results:
            chat_history = (
                self.dialogue_manager.get_history_messages()
                if self.enable_dialogue else None
            )
            messages = build_prompt(
                user_query=resolved_query,
                retrieval_results=retrieval_results,
                chat_history=chat_history,
            )
        else:
            messages = build_no_result_prompt(resolved_query)

        # 4. 流式生成
        yield from self.llm_client.chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    def simple_answer(self, query: str) -> str:
        """
        简化版问答接口，直接返回回答文本。

        Args:
            query: 用户查询

        Returns:
            回答文本
        """
        response = self.answer(query)
        return response.answer

    def clear_dialogue(self):
        """清空对话历史"""
        self.dialogue_manager.clear()

    def close(self):
        """释放资源"""
        if self.retriever:
            self.retriever.close()

    def __repr__(self) -> str:
        return (
            f"Generator("
            f"llm={self.llm_client.model}, "
            f"dialogue_turns={self.dialogue_manager.turn_count})"
        )


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("\n" + "=" * 60)
    print("生成引擎测试")
    print("=" * 60)

    generator = Generator()

    test_queries = [
        "人参的性味归经是什么？",
        "那它的功能主治呢？",       # 多轮对话：指代消解
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"用户: {query}")
        print("-" * 60)

        response = generator.answer(query)

        print(f"消解后查询: {response.resolved_query}")
        print(f"\n回答:\n{response.answer}")
        print(f"\n引用: {response.citations}")
        print(f"耗时: {response.latency:.3f}s")
        print(f"各阶段: {response.component_latency}")

    generator.close()
    print(f"\n{'='*60}")
    print("✅ 生成引擎测试完成!")
