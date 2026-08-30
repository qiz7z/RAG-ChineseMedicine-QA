# -*- coding: utf-8 -*-
"""
纯标准组件 Baseline（LangChain 版对照组）
==========================================
只用 LangChain 开箱即用的标准组件做向量检索 + 生成，
不含任何定制逻辑（无 BM25 混合、无元数据过滤、无重排），
作为对比实验的对照组，用于量化手撕优化管线带来的增益。

输出 Document 的 metadata 形状与 HybridRetriever 保持一致，
评估适配器（eval_adapter.py）可无缝切换两种引擎。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from embeddings import build_embeddings

LC_INDEX_DIR = str(PROJECT_ROOT / "data" / "vectorstore" / "langchain_faiss")


class StandardBaselineRetriever:
    """纯标准 LangChain 向量检索（开箱即用对照组）"""

    def __init__(self, default_k: int = 5):
        from langchain_community.vectorstores import FAISS
        import logging
        logging.getLogger("langchain_community.vectorstores.FAISS").setLevel(
            logging.ERROR
        )

        self.default_k = default_k
        self.embeddings = build_embeddings()
        self.vectorstore = FAISS.load_local(
            LC_INDEX_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, top_k: int = None) -> list:
        """纯向量检索，返回与 HybridRetriever 相同形状的 Document 列表"""
        import time

        top_k = top_k or self.default_k
        t0 = time.time()
        pairs = self.vectorstore.similarity_search_with_score(query, k=top_k)
        docs = []
        for doc, score in pairs:
            meta = dict(doc.metadata)
            meta.setdefault("sources", [])
            meta["score"] = float(score)
            meta["rerank_score"] = None
            docs.append(Document(page_content=doc.page_content, metadata=meta))
        self.last_latency = time.time() - t0
        return docs

    def hybrid_search(self, query: str, top_k: int = None, drug_filter=None):
        """与 HybridRetriever.hybrid_search 同签名的适配方法。

        刻意忽略 drug_filter / 元数据过滤 / BM25 / 重排——
        纯标准组件对照组的定义就是"开箱即用"。
        """
        import time

        total_start = time.time()
        docs = self.retrieve(query, top_k)
        parsed = {
            "raw_query": query,
            "drug_names": [],
            "sections": [],
            "intent": "通用查询",
            "keywords": [],
            "filter": None,
            "is_horizontal": False,
            "category_filter": None,
        }
        latency = time.time() - total_start
        return docs, parsed, latency, {"vector_search": self.last_latency}
