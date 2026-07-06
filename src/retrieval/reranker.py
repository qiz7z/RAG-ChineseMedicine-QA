# -*- coding: utf-8 -*-
"""
BGE Reranker 重排模型封装
=========================
使用 BAAI/bge-reranker-v2-m3 对召回的候选结果进行精排。

为什么需要重排？
  - 向量检索（双塔模型）速度快但精度有限，召回的 Top-30 中可能有噪声
  - BM25 基于词频匹配，无法理解语义
  - RRF 融合只看排名，不评估真正的相关性
  - 重排模型（交叉编码器）将 query 和 document 拼接输入，能精确评估相关性

模型架构差异：
  ┌─────────────┬──────────────────┬──────────────────┐
  │             │  Embedding(双塔)  │  Reranker(交叉)  │
  ├─────────────┼──────────────────┼──────────────────┤
  │  query处理  │  独立编码         │  与doc拼接编码    │
  │  doc处理    │  独立编码         │  与query拼接编码  │
  │  交互方式   │  向量点积(无交互)  │  全注意力交互     │
  │  速度       │  快(可离线编码)   │  慢(必须在线计算) │
  │  精度       │  中              │  高              │
  └─────────────┴──────────────────┴──────────────────┘

工作流程：
  1. 向量检索 + BM25 → 召回 Top-30 候选
  2. Reranker 对 30 个 (query, doc) 对逐一打分
  3. 按重排分数排序，取 Top-5 返回
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RERANKER_MODEL_PATH, RERANKER_TOP_K, RERANKER_MAX_LENGTH


class Reranker:
    """BGE-Reranker-v2-m3 重排模型封装"""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        max_length: int = None,
    ):
        """
        Args:
            model_path: 模型路径，默认从 config 读取
            device: 'cpu' / 'cuda'，默认自动检测
            max_length: 最大序列长度
        """
        from sentence_transformers import CrossEncoder

        self.model_path = model_path or RERANKER_MODEL_PATH
        self.max_length = max_length or RERANKER_MAX_LENGTH

        # 自动检测设备
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # 判断模型来源
        is_local = os.path.isdir(self.model_path)
        source = "本地" if is_local else "HuggingFace Hub"
        print(f"正在加载重排模型: {self.model_path} (device={device}, 来源={source})")

        self.model = CrossEncoder(
            self.model_path,
            device=device,
            max_length=self.max_length,
        )
        print(f"重排模型加载完成 (max_length={self.max_length})")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        对候选文档进行重排。

        Args:
            query: 用户查询文本
            documents: 候选文档列表，每个文档必须包含 "content" 字段
            top_k: 重排后返回的文档数，默认从 config 读取

        Returns:
            重排后的文档列表，按相关性降序排列
            每个文档新增 "rerank_score" 字段
        """
        if not documents:
            return []

        top_k = top_k or RERANKER_TOP_K

        # 构建 (query, doc) 对
        pairs = []
        for doc in documents:
            content = doc.get("content", "")
            # 截断过长的文档内容（避免超出模型最大长度）
            if len(content) > self.max_length * 2:
                content = content[:self.max_length * 2]
            pairs.append((query, content))

        # 批量预测
        scores = self.model.predict(pairs, batch_size=32)

        # 将分数附加到文档上
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # 按重排分数降序排序
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

    def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        top_k: int = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        批量重排多个查询的结果。

        Args:
            queries: 查询列表
            documents_list: 每个查询对应的候选文档列表
            top_k: 每个查询返回的文档数

        Returns:
            重排后的文档列表的列表
        """
        top_k = top_k or RERANKER_TOP_K
        results = []
        for query, docs in zip(queries, documents_list):
            results.append(self.rerank(query, docs, top_k=top_k))
        return results


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("BGE Reranker 重排模型测试")
    print("=" * 60)

    reranker = Reranker()

    query = "人参的性味归经是什么？"
    documents = [
        {"id": "d1", "content": "人参 甘、微苦，微温。归脾、肺、心经。"},
        {"id": "d2", "content": "人参的含量测定：照高效液相色谱法测定人参皂苷含量。"},
        {"id": "d3", "content": "黄芪 补气固表，利尿托毒，排脓，敛疮生肌。"},
        {"id": "d4", "content": "人参的功能与主治：大补元气，复脉固脱，补脾益肺，生津养血，安神益智。"},
        {"id": "d5", "content": "高效液相色谱法（通则0512）的色谱条件与系统适用性试验。"},
    ]

    print(f"\n查询: '{query}'")
    print(f"候选文档数: {len(documents)}")

    reranked = reranker.rerank(query, documents, top_k=3)

    print(f"\n重排后 Top-3:")
    for i, doc in enumerate(reranked, 1):
        score = doc["rerank_score"]
        content = doc["content"][:60]
        print(f"  {i}. [score={score:.6f}] {content}")
