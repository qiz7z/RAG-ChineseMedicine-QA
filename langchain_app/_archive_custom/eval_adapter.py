# -*- coding: utf-8 -*-
"""
评估引擎适配
=============
为对比实验组装不同引擎变体。所有变体共享同一评估器
（src/eval/evaluator.py，duck-typed）与测试集，保证口径一致：

  hybrid   — LangChain 混合检索引擎（LC FAISS + BM25 + RRF + 重排，
             复用手撕版解析/融合/重排/后处理组件）
  standard — 纯标准组件对照组（仅 LC FAISS 向量检索，无混合/过滤/重排）
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent / "src"))

from chain import LCGenerator


def build_engine(variant: str = "hybrid", enable_reranker: bool = None) -> LCGenerator:
    """按变体构建评估引擎。

    Args:
        variant: "hybrid" | "standard"
        enable_reranker: 是否启用重排（仅 hybrid 变体有效）
    """
    if variant == "hybrid":
        return LCGenerator(enable_reranker=enable_reranker)
    if variant == "standard":
        from standard_baseline import StandardBaselineRetriever
        return LCGenerator(retriever=StandardBaselineRetriever())
    raise ValueError(f"未知变体: {variant}（可选 hybrid / standard）")
