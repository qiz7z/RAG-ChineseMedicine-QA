# -*- coding: utf-8 -*-
"""
生成引擎模块（阶段四）
======================
负责将检索结果通过 LLM 转化为高质量的自然语言回答。

模块组成：
  - prompts:        Prompt 模板设计
  - llm_client:     LLM API 客户端（LongCat 2.0 / OpenAI 兼容）
  - postprocessor:  回答后处理（引用标注、一致性校验、格式美化）
  - dialogue:       多轮对话管理（上下文保持、指代消解）
  - generator:      生成引擎主模块（整合检索+生成）
"""

from generation.generator import Generator, GenerationResponse
from generation.llm_client import LLMClient
from generation.dialogue import DialogueManager
