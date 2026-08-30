# -*- coding: utf-8 -*-
"""LLM 客户端（标准版）：langchain_openai.ChatOpenAI，OpenAI 兼容接口"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_openai import ChatOpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


def build_chat_model(temperature: float = None, max_tokens: int = None) -> ChatOpenAI:
    """构建 LLM 实例（LongCat / Agnes 等 OpenAI 兼容端点）"""
    if not LLM_API_KEY:
        raise RuntimeError(
            "LLM API Key 未设置！请设置环境变量 LONGCAT_API_KEY "
            "或在项目根目录 .env 中配置。"
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        timeout=120,
        max_retries=1,  # 4xx 类错误（额度/鉴权）重试无意义
    )
