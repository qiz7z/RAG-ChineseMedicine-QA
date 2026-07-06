# -*- coding: utf-8 -*-
"""
LLM 客户端
===========
封装美团 LongCat 2.0 API（OpenAI 兼容接口）的调用逻辑。

功能：
  - 统一的 chat completion 接口
  - 自动重试（指数退避）
  - 流式输出支持
  - 超时控制
  - API Key 校验

LongCat 2.0 API 信息：
  - 接入端点: https://api.longcat.chat/openai
  - 模型名称: LongCat-2.0
  - 兼容 OpenAI API 规范
  - 上下文长度: 1M tokens，最大输出 128K tokens
"""
import time
import logging
from typing import List, Dict, Optional, Generator

from config import (
    LONGCAT_API_KEY,
    LONGCAT_BASE_URL,
    LONGCAT_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LongCat 2.0 LLM 客户端。

    使用 OpenAI SDK 调用 LongCat API，兼容 OpenAI 接口格式。

    使用方式：
        client = LLMClient()
        response = client.chat(messages)
        print(response)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Args:
            api_key: API Key，默认从 config 读取（环境变量 LONGCAT_API_KEY）
            base_url: API 端点，默认从 config 读取
            model: 模型名称，默认从 config 读取
        """
        self.api_key = api_key or LONGCAT_API_KEY
        self.base_url = base_url or LONGCAT_BASE_URL
        self.model = model or LONGCAT_MODEL

        if not self.api_key:
            raise ValueError(
                "LongCat API Key 未设置！请设置环境变量 LONGCAT_API_KEY，\n"
                "或通过 LLMClient(api_key='your_key') 传入。\n"
                "获取 API Key: https://longcat.chat/platform/api_keys"
            )

        # 延迟导入 OpenAI SDK（避免未安装时影响其他模块）
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "未安装 openai 包，请运行: pip install openai"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        print(f"  LLM 客户端初始化完成:")
        print(f"    模型: {self.model}")
        print(f"    端点: {self.base_url}")

    # ----------------------------------------------------------
    # 核心方法：Chat Completion
    # ----------------------------------------------------------

    def chat(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        timeout: int = 60,
    ) -> str:
        """
        调用 LLM 生成回答（非流式）。

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 温度参数，默认从 config 读取
            max_tokens: 最大输出 token 数，默认从 config 读取
            top_p: 核采样参数，默认从 config 读取
            timeout: 请求超时时间（秒）

        Returns:
            LLM 生成的回答文本
        """
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        max_tokens = max_tokens or LLM_MAX_TOKENS
        top_p = top_p if top_p is not None else LLM_TOP_P

        last_error = None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    timeout=timeout,
                )
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                if attempt < LLM_MAX_RETRIES:
                    delay = LLM_RETRY_DELAY ** attempt
                    logger.warning(
                        f"LLM 调用失败 (第{attempt}次): {e}，{delay}秒后重试..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM 调用失败（已重试{LLM_MAX_RETRIES}次）: {e}")

        raise RuntimeError(f"LLM 调用失败: {last_error}")

    # ----------------------------------------------------------
    # 流式输出
    # ----------------------------------------------------------

    def chat_stream(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        timeout: int = 60,
    ) -> Generator[str, None, None]:
        """
        调用 LLM 生成回答（流式输出）。

        逐 token 返回生成内容，适用于实时显示场景。

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            top_p: 核采样参数
            timeout: 请求超时时间（秒）

        Yields:
            每次生成的文本片段
        """
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        max_tokens = max_tokens or LLM_MAX_TOKENS
        top_p = top_p if top_p is not None else LLM_TOP_P

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    def simple_chat(self, user_message: str, system_message: str = "") -> str:
        """
        简化版调用接口（单轮对话）。

        Args:
            user_message: 用户消息
            system_message: 系统消息（可选）

        Returns:
            LLM 回答
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.chat(messages)

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model}, base_url={self.base_url})"
