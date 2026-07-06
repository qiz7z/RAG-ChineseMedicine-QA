# -*- coding: utf-8 -*-
"""
检索引擎模块（阶段三）
====================
将向量检索、BM25 检索、RRF 融合、重排模型整合为统一的检索接口。

核心组件：
  1. QueryParser    — 查询解析：提取药品名、章节名、识别查询意图
  2. Reranker       — BGE-Reranker-v2-m3 重排模型封装
  3. Retriever      — 主检索引擎：协调多路召回 → RRF 融合 → 重排 → 上下文组装

检索流程：
  用户查询
    │
    ├─ QueryParser 解析（提取药品名、章节名、意图识别）
    │
    ├─ 向量检索（Chroma, top_k=15）──┐
    │                                ├─ RRF 融合 → top_n=30 候选
    ├─ BM25 检索（top_k=15）─────────┘
    │
    ├─ Reranker 重排（BGE-reranker-v2-m3）
    │
    └─ 上下文组装 → 返回 Top-5 结果
"""
