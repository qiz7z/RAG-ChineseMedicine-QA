# -*- coding: utf-8 -*-
"""
多轮对话管理模块
=================
管理多轮对话上下文，支持：

1. 对话历史保持：维护最近 N 轮对话
2. 指代消解：通过 LLM 将"它"、"这个药"等指代词替换为具体药品名
3. 上下文窗口控制：防止对话历史超出 token 限制

使用方式：
    dm = DialogueManager(llm_client)
    dm.add_user_message("人参的性味归经是什么？")
    dm.add_assistant_message("人参的性味为甘、微苦...")
    resolved = dm.resolve_query("那它的功能主治呢？")
    # resolved => "那人参的功能主治呢？"
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from config import DIALOGUE_MAX_HISTORY, DIALOGUE_CONTEXT_WINDOW

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """单轮对话"""
    role: str           # "user" 或 "assistant"
    content: str        # 消息内容
    query: str = ""     # 原始查询（仅 user 角色有）


class DialogueManager:
    """
    多轮对话管理器。

    负责维护对话历史、执行指代消解、控制上下文窗口。

    使用方式：
        dm = DialogueManager(llm_client)
        # 第一轮
        resolved_q = dm.resolve_query("人参的性味归经是什么？")
        dm.add_user_message(resolved_q)
        dm.add_assistant_message(answer)
        # 第二轮
        resolved_q = dm.resolve_query("那它的功能主治呢？")
        # resolved_q => "那人参的功能主治呢？"
    """

    def __init__(
        self,
        llm_client=None,
        max_history: int = None,
    ):
        """
        Args:
            llm_client: LLM 客户端实例（用于指代消解）。如不提供则跳过指代消解。
            max_history: 保留的最大对话轮数
        """
        self.llm_client = llm_client
        self.max_history = max_history or DIALOGUE_MAX_HISTORY
        self.history: List[DialogueTurn] = []

    # ----------------------------------------------------------
    # 对话历史管理
    # ----------------------------------------------------------

    def add_user_message(self, content: str, original_query: str = ""):
        """添加用户消息到对话历史"""
        self.history.append(DialogueTurn(
            role="user",
            content=content,
            query=original_query or content,
        ))
        self._trim_history()

    def add_assistant_message(self, content: str):
        """添加助手消息到对话历史"""
        self.history.append(DialogueTurn(
            role="assistant",
            content=content,
        ))
        self._trim_history()

    def clear(self):
        """清空对话历史"""
        self.history.clear()

    def _trim_history(self):
        """裁剪对话历史，保留最近 max_history 轮"""
        max_turns = self.max_history * 2  # 每轮 = user + assistant
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

    # ----------------------------------------------------------
    # 获取对话历史（OpenAI 格式）
    # ----------------------------------------------------------

    def get_history_messages(self) -> List[Dict]:
        """
        获取对话历史（OpenAI messages 格式）。

        用于注入到 LLM 的 messages 列表中。

        Returns:
            OpenAI 格式的消息列表
        """
        # 控制上下文窗口大小
        messages = []
        total_chars = 0

        # 从最近的对话开始，向前累加，直到达到窗口限制
        for turn in reversed(self.history):
            if total_chars + len(turn.content) > DIALOGUE_CONTEXT_WINDOW:
                break
            messages.insert(0, {
                "role": turn.role,
                "content": turn.content,
            })
            total_chars += len(turn.content)

        return messages

    # ----------------------------------------------------------
    # 指代消解
    # ----------------------------------------------------------

    def resolve_query(self, current_query: str) -> str:
        """
        对当前查询执行指代消解。

        如果对话历史为空或 LLM 客户端未设置，直接返回原查询。

        Args:
            current_query: 用户当前查询

        Returns:
            消解后的查询（指代词替换为具体药品名）
        """
        # 无对话历史或无 LLM 客户端时，跳过指代消解
        if not self.history or not self.llm_client:
            return current_query

        # 检查是否包含可能的指代词
        if not self._has_coreference(current_query):
            return current_query

        try:
            from generation.prompts import build_coreference_prompt

            history_messages = self.get_history_messages()
            messages = build_coreference_prompt(current_query, history_messages)

            resolved = self.llm_client.chat(
                messages,
                temperature=0.0,   # 指代消解需要确定性输出
                max_tokens=256,
            )

            resolved = resolved.strip()

            # 简单校验：消解结果不应为空，且长度不应超过原查询的3倍
            if resolved and len(resolved) < len(current_query) * 3:
                logger.info(f"指代消解: '{current_query}' -> '{resolved}'")
                return resolved
            else:
                logger.warning("指代消解结果异常，使用原始查询")
                return current_query

        except Exception as e:
            logger.warning(f"指代消解失败: {e}，使用原始查询")
            return current_query

    def _has_coreference(self, query: str) -> bool:
        """
        快速检查查询中是否包含可能的指代词。

        Args:
            query: 用户查询

        Returns:
            是否包含指代词
        """
        # 中文常见指代词
        coreference_words = [
            "它", "它们", "这个药", "该药", "该药材", "这种药",
            "这种药材", "上述", "前面提到的", "刚才说的",
        ]
        return any(word in query for word in coreference_words)

    # ----------------------------------------------------------
    # 状态查询
    # ----------------------------------------------------------

    def get_last_drug_name(self) -> Optional[str]:
        """
        从对话历史中提取最近提到的药品名。

        用于辅助检索过滤。

        Returns:
            最近的药品名（如无则返回 None）
        """
        # 从最近的用户查询中查找药品名
        for turn in reversed(self.history):
            if turn.role == "user":
                # 简单策略：从查询文本中提取引号内的内容或已知药品名
                # 更精确的药品名提取由 QueryParser 完成
                return turn.query
        return None

    @property
    def turn_count(self) -> int:
        """当前对话轮数"""
        return len(self.history) // 2

    def __repr__(self) -> str:
        return f"DialogueManager(turns={self.turn_count}, max_history={self.max_history})"
