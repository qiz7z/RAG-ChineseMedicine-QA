# -*- coding: utf-8 -*-
"""
Embedding 模型封装
==================
使用 BAAI/bge-large-zh-v1.5 模型将文本转为向量。

BGE 模型特点：
  - 中文语义理解能力强（C-MTEB 榜单排名靠前）
  - 输出 1024 维向量
  - 编码 query 时需要添加指令前缀，编码文档时不需要
  - 推荐使用归一化向量（normalize_embeddings=True），配合余弦相似度
"""
import sys
from pathlib import Path
from typing import List
import numpy as np

# 设置路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, BGE_QUERY_INSTRUCTION


class Embedder:
    """BGE Embedding 模型封装"""

    def __init__(self, model_name: str = None, device: str = None):
        """
        Args:
            model_name: 模型路径或 HuggingFace 模型名，默认从 config 读取
                        优先加载本地 models/bge-large-zh-v1.5/，不存在则从 HF 下载
            device: 'cpu' / 'cuda' / 'cuda:0'，默认自动检测
        """
        from sentence_transformers import SentenceTransformer
        import os

        self.model_name = model_name or EMBEDDING_MODEL_NAME

        # 自动检测设备
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # 判断模型来源（本地目录 or HuggingFace Hub）
        is_local = os.path.isdir(self.model_name)
        source = "本地" if is_local else "HuggingFace Hub"
        print(f"正在加载 Embedding 模型: {self.model_name} (device={device}, 来源={source})")

        self.model = SentenceTransformer(self.model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"模型加载完成，输出维度: {self.embedding_dim}")

    def encode_documents(self, texts: List[str], batch_size: int = None,
                         show_progress: bool = True) -> np.ndarray:
        """
        编码文档（chunk 内容），不需要添加查询指令。

        Args:
            texts: 文档文本列表
            batch_size: 批量大小
            show_progress: 是否显示进度条

        Returns:
            归一化后的向量矩阵 (N, dim)
        """
        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询文本，添加 BGE 查询指令前缀。

        Args:
            query: 用户查询文本

        Returns:
            归一化后的查询向量 (dim,)
        """
        query_with_instruction = BGE_QUERY_INSTRUCTION + query
        embedding = self.model.encode(
            [query_with_instruction],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)

    def encode_queries(self, queries: List[str], batch_size: int = None) -> np.ndarray:
        """批量编码查询"""
        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        queries_with_instruction = [BGE_QUERY_INSTRUCTION + q for q in queries]
        embeddings = self.model.encode(
            queries_with_instruction,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)


# ============================================================
# 命令行入口：测试 Embedding 模型
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Embedding 模型测试")
    print("=" * 60)

    embedder = Embedder()

    # 测试文档编码
    docs = [
        "人参的性味与归经：甘、微苦，微温。归脾、肺、心经。",
        "黄芪的功能与主治：补气固表，利尿托毒，排脓，敛疮生肌。",
        "高效液相色谱法（通则0512）测定含量。",
    ]
    doc_vectors = embedder.encode_documents(docs)
    print(f"\n文档编码: {len(docs)} 条 → 向量形状 {doc_vectors.shape}")

    # 测试查询编码
    query = "人参的性味归经是什么？"
    query_vector = embedder.encode_query(query)
    print(f"查询编码: '{query}' → 向量形状 {query_vector.shape}")

    # 计算相似度
    from numpy.linalg import norm
    similarities = doc_vectors @ query_vector  # 归一化向量内积 = 余弦相似度
    print(f"\n相似度排序:")
    for idx in np.argsort(similarities)[::-1]:
        print(f"  {similarities[idx]:.4f}  {docs[idx][:50]}")
