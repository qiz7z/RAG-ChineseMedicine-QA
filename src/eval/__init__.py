# -*- coding: utf-8 -*-
"""
评估引擎模块（阶段五）
=====================
对 RAG 系统进行全量评估，包括检索质量和生成质量。

评估维度：
  1. 检索质量：Hit@K、MRR、各阶段延迟
  2. 生成质量：关键词覆盖率、一致性校验、引用准确率
  3. 系统性能：端到端延迟（P50/P95/P99）、各组件延迟
"""
from eval.evaluator import RetrievalEvaluator, GenerationEvaluator, EvalReport

__all__ = ["RetrievalEvaluator", "GenerationEvaluator", "EvalReport"]
