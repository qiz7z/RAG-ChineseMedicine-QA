# -*- coding: utf-8 -*-
"""
BGE Embedding（标准版）
=======================
基于 langchain-huggingface.HuggingFaceEmbeddings；唯一的自定义点是查询端
附加 BGE 官方检索指令（模型卡片建议），文档端不加——这是模型用法约定，
不是框架定制。
"""
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL_PATH,
    EMBEDDING_BATCH_SIZE,
    BGE_QUERY_INSTRUCTION,
)


class BGEQueryEmbeddings(HuggingFaceEmbeddings):
    """查询编码自动附加 BGE 检索指令（文档编码不变）"""

    query_instruction: str = ""

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"{self.query_instruction}{text}")


def build_embeddings(device: str = None) -> BGEQueryEmbeddings:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return BGEQueryEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": EMBEDDING_BATCH_SIZE,
            "normalize_embeddings": True,
        },
        query_instruction=BGE_QUERY_INSTRUCTION,
    )
