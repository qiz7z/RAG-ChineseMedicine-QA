# -*- coding: utf-8 -*-
"""
混合检索器（LangChain 版）
==========================
以 LangChain BaseRetriever 接口实现与手撕版 Retriever.search() 等价的
混合检索管线，保证对比实验同口径：

  查询解析 → 向量检索(LC FAISS) + BM25检索(复用手撕版索引)
           → RRF 融合(复用) → 元数据补全 → 重排(复用)

差异点：
  - 向量库为 LangChain FAISS 封装（data/vectorstore/langchain_faiss/）
  - BM25 / SQLite 元数据 / 查询解析 / RRF / 重排模块直接复用手撕版实现
"""
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Callable

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))                    # embeddings 平铺导入
sys.path.insert(0, str(APP_DIR.parent / "src"))     # config / src 各模块

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document

from config import (
    BM25_INDEX_PATH,
    SQLITE_DB_PATH,
    VECTOR_TOP_K,
    BM25_TOP_K,
    RRF_K,
    RRF_TOP_N,
    RERANKER_TOP_K,
)
from indexing.keyword_index import BM25Index, JiebaTokenizer
from indexing.metadata_store import MetadataStore
from indexing.fusion import rrf_fusion
from retrieval.query_parser import QueryParser
from retrieval.reranker import Reranker

from embeddings import build_embeddings

# LangChain 版向量索引目录（与手撕版 data/vectorstore/chroma/ 隔离）
LC_INDEX_DIR = str(PROJECT_ROOT / "data" / "vectorstore" / "langchain_faiss")

# QueryParser 返回的分类简称 → SQLite 实际分类值映射。
# 手撕版直接用简称（"药材"/"成方制剂"）做过滤，与 SQLite 实际值
# （"药材和饮片"/"成方制剂和单味制剂"）不匹配，导致横向查询两路召回
# 均为空——这是横向条件查询 Hit@5 仅 30% 的原因之一。LC 版在此修复。
CATEGORY_FILTER_MAP = {
    "药材": "药材和饮片",
    "成方制剂": "成方制剂和单味制剂",
    "药材和饮片": "药材和饮片",
    "成方制剂和单味制剂": "成方制剂和单味制剂",
    "植物油脂和提取物": "植物油脂和提取物",
}


def doc_to_result_dict(doc: Document) -> dict:
    """把检索 Document 转成与手撕版 SearchResult.to_dict() 同构的字典，
    供 prompts.build_context_section / PostProcessor / API Schema 直接消费。"""
    m = doc.metadata
    return {
        "chunk_id": m.get("chunk_id", ""),
        "content": doc.page_content,
        "score": m.get("score", 0),
        "drug_name": m.get("drug_name", ""),
        "pinyin_name": m.get("pinyin_name", ""),
        "latin_name": m.get("latin_name", ""),
        "category": m.get("category", ""),
        "section": m.get("section", ""),
        "chunk_type": m.get("chunk_type", ""),
        "is_yinpian": bool(m.get("is_yinpian", False)),
        "parent_drug": m.get("parent_drug", ""),
        "char_count": m.get("char_count", 0),
        "sources": m.get("sources", []),
        "rerank_score": m.get("rerank_score"),
    }


def make_metadata_filter(
    expanded_drugs: Optional[set], category: Optional[str]
) -> Optional[Callable[[dict], bool]]:
    """构造 LC FAISS 的元数据过滤谓词（filter 支持传 callable）。

    语义与手撕版 chroma_filter + $and 组合一致：
      - expanded_drugs 非空: drug_name 必须在扩展集合内（$in）
      - category 非空: category 精确相等
    """
    def _match(meta: dict) -> bool:
        if expanded_drugs and meta.get("drug_name") not in expanded_drugs:
            return False
        if category and meta.get("category") != category:
            return False
        return True

    if not expanded_drugs and not category:
        return None
    return _match


class HybridRetriever(BaseRetriever):
    """LangChain 版混合检索器（向量 + BM25 + RRF + 重排）"""

    # pydantic 字段（BaseRetriever 是 pydantic 模型，非序列化对象用 Any 声明）
    vectorstore: Any = None
    bm25_index: Any = None
    meta_store: Any = None
    query_parser: Any = None
    reranker: Any = None
    enable_filter: bool = True
    k_vector: int = VECTOR_TOP_K
    k_bm25: int = BM25_TOP_K
    k_rrf: int = RRF_K
    n_rrf: int = RRF_TOP_N

    @classmethod
    def create(cls, enable_reranker: bool = None, enable_filter: bool = True):
        """加载全部依赖并装配检索器（与手撕版 Retriever.__init__ 对齐）"""
        from langchain_community.vectorstores import FAISS
        import logging
        logging.getLogger("langchain_community.vectorstores.FAISS").setLevel(
            logging.ERROR
        )

        from config import ENABLE_RERANKER

        enable_reranker = (
            ENABLE_RERANKER if enable_reranker is None else enable_reranker
        )

        t0 = time.time()
        print("  加载 LangChain FAISS 向量索引...")
        embeddings = build_embeddings()
        vectorstore = FAISS.load_local(
            LC_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"  向量索引加载完成: {vectorstore.index.ntotal} 条 ({time.time() - t0:.1f}s)")

        t0 = time.time()
        bm25_index = BM25Index()
        bm25_index.load(BM25_INDEX_PATH)
        print(f"  BM25 索引加载完成: {bm25_index.count()} 条 ({time.time() - t0:.1f}s)")

        t0 = time.time()
        meta_store = MetadataStore(SQLITE_DB_PATH)
        drug_names = meta_store.get_all_drug_names()
        query_parser = QueryParser(drug_names=drug_names)
        print(f"  查询解析器初始化完成: {len(drug_names)} 个药品名 ({time.time() - t0:.1f}s)")

        JiebaTokenizer.add_custom_words(drug_names)

        reranker = None
        if enable_reranker:
            t0 = time.time()
            reranker = Reranker()
            print(f"  重排模型加载完成 ({time.time() - t0:.1f}s)")
        else:
            print("  重排模型: 未启用")

        return cls(
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            meta_store=meta_store,
            query_parser=query_parser,
            reranker=reranker,
            enable_filter=enable_filter,
        )

    # ----------------------------------------------------------
    # LangChain BaseRetriever 接口
    # ----------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        top_k: int = None,
        drug_filter: Optional[str] = None,
    ) -> List[Document]:
        """执行混合检索，返回按相关度排序的 Document 列表。

        score / rerank_score / sources 等检索元信息放在 Document.metadata 中。
        """
        top_k = top_k or RERANKER_TOP_K
        docs, _, _, _ = self.hybrid_search(
            query, top_k=top_k, drug_filter=drug_filter
        )
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """同步实现（组件均为同步，避免引入线程语义差异）"""
        return self._get_relevant_documents(
            query,
            run_manager=None,
            **kwargs,
        )

    # ----------------------------------------------------------
    # 混合检索管线（与手撕版 Retriever.search() 逐段对齐）
    # ----------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        drug_filter: Optional[str] = None,
    ):
        """完整混合检索。

        Returns:
            (documents, parsed, latency, component_latency)
        """
        top_k = top_k or RERANKER_TOP_K
        component_latency = {}
        total_start = time.time()

        # ============ 1. 查询解析 ============
        t0 = time.time()
        parsed = self.query_parser.parse(query)
        # 手动药品过滤覆盖（与手撕版语义一致）
        if drug_filter:
            parsed["drug_names"] = [drug_filter]
        component_latency["query_parsing"] = time.time() - t0

        # ============ 2. 向量检索（LC FAISS） ============
        t0 = time.time()
        filter_drugs = parsed["drug_names"] if self.enable_filter else []
        expanded_drugs = None
        if filter_drugs:
            all_drug_names = self.meta_store.get_all_drug_names()
            expanded_drugs = set()
            for fd in filter_drugs:
                for dn in all_drug_names:
                    if fd in dn or dn in fd:
                        expanded_drugs.add(dn)

        category = (
            parsed.get("category_filter")
            if self.enable_filter and parsed.get("is_horizontal")
            else None
        )
        # 修复分类简称与 SQLite 实际值不匹配的问题（见 CATEGORY_FILTER_MAP 注释）
        if category:
            category = CATEGORY_FILTER_MAP.get(category, category)
        meta_filter = make_metadata_filter(expanded_drugs, category)

        if meta_filter is not None:
            # 带过滤：遍历满足条件的候选再精确检索
            candidates = [
                (chunk_id, meta)
                for chunk_id, meta in self._iter_docstore_metadata()
                if meta_filter(meta)
            ]
            vector_results = self._search_subset(query, candidates, self.k_vector)
        else:
            pairs = self.vectorstore.similarity_search_with_score(
                query, k=self.k_vector
            )
            vector_results = [
                {
                    "id": doc.metadata.get("chunk_id", ""),
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": float(score),
                    "source": "vector",
                }
                for doc, score in pairs
            ]
        component_latency["vector_search"] = time.time() - t0

        # ============ 3. BM25 检索（复用手撕版索引与后过滤逻辑） ============
        t0 = time.time()
        bm25_top_k = self.k_bm25
        if self.enable_filter and parsed["drug_names"]:
            bm25_top_k = self.k_bm25 * 4  # 扩大召回，过滤后仍保证足够候选
        bm25_results = self.bm25_index.query(query, top_k=bm25_top_k)

        if self.enable_filter and parsed["drug_names"]:
            bm25_results = self._filter_bm25(
                bm25_results, lambda meta: any(
                    fd in meta.get("drug_name", "") or meta.get("drug_name", "") in fd
                    for fd in parsed["drug_names"]
                )
            )

        if self.enable_filter and parsed.get("is_horizontal") and parsed.get("category_filter"):
            cat = parsed["category_filter"]
            bm25_results = self._filter_bm25(
                bm25_results, lambda meta: meta.get("category", "") == cat
            )
        component_latency["bm25_search"] = time.time() - t0

        # ============ 4. RRF 融合 ============
        t0 = time.time()
        fused = rrf_fusion([vector_results, bm25_results], k=self.k_rrf, top_n=self.n_rrf)
        component_latency["rrf_fusion"] = time.time() - t0

        # ============ 5. 元数据补全 ============
        t0 = time.time()
        if fused:
            metas = self.meta_store.get_by_ids([r["id"] for r in fused])
            meta_map = {m["chunk_id"]: m for m in metas}
            for r in fused:
                r["metadata"] = meta_map.get(r["id"], {})
        component_latency["metadata_enrichment"] = time.time() - t0

        # ============ 6. 重排 ============
        t0 = time.time()
        if self.reranker and fused:
            reranked = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            reranked = sorted(
                fused, key=lambda x: x.get("rrf_score", 0), reverse=True
            )[:top_k]
        component_latency["reranking"] = time.time() - t0

        # ============ 7. 转换为 Document ============
        documents = []
        for r in reranked:
            meta = dict(r.get("metadata", {}))
            meta["score"] = r.get("rerank_score") if self.reranker else r.get("rrf_score", 0)
            meta["rerank_score"] = r.get("rerank_score")
            meta["sources"] = r.get("sources", [])
            meta["rrf_score"] = r.get("rrf_score")
            documents.append(Document(page_content=r.get("content", ""), metadata=meta))

        component_latency["context_assembly"] = 0.0
        return documents, parsed, time.time() - total_start, component_latency

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------

    def _iter_docstore_metadata(self):
        """遍历 LC FAISS docstore 的 (chunk_id, metadata)"""
        for doc_id, doc in self.vectorstore.docstore._dict.items():
            yield doc_id, doc.metadata

    def _search_subset(self, query: str, candidates, top_k: int) -> List[dict]:
        """在满足过滤条件的候选子集内做向量检索（带过滤时的召回路径）。

        与手撕版 chroma 过滤路径等价：重建候选子集向量，用查询向量做内积
        打分后取 top_k（向量已归一化，内积 = 余弦相似度）。
        """
        import numpy as np

        if not candidates:
            return []

        # docstore_id → FAISS 内部索引号的映射
        id_to_internal = {
            doc_id: idx
            for idx, doc_id in self.vectorstore.index_to_docstore_id.items()
        }
        pairs = [
            (id_to_internal[cid], cid, meta)
            for cid, meta in candidates
            if cid in id_to_internal
        ]
        if not pairs:
            return []

        # 查询向量（BGEEmbeddings.embed_query 已附加指令前缀并归一化）
        qvec = np.array([self.vectorstore.embeddings.embed_query(query)], dtype=np.float32)

        internal_idxs = [p[0] for p in pairs]
        sub_vecs = np.vstack(
            [self.vectorstore.index.reconstruct(i) for i in internal_idxs]
        ).astype(np.float32)
        scores = (sub_vecs @ qvec.T).ravel()

        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for local in order:
            _, cid, meta = pairs[int(local)]
            results.append({
                "id": cid,
                "content": self.vectorstore.docstore.search(cid).page_content,
                "metadata": dict(meta),
                "score": float(scores[int(local)]),
                "source": "vector",
            })
        return results

    def _filter_bm25(self, bm25_results: List[dict], predicate) -> List[dict]:
        """对 BM25 结果按 SQLite 元数据过滤（与手撕版后过滤逻辑一致）"""
        if not bm25_results:
            return []
        ids = [r["id"] for r in bm25_results]
        metas = self.meta_store.get_by_ids(ids)
        meta_map = {m["chunk_id"]: m for m in metas}
        filtered = []
        for r in bm25_results:
            meta = meta_map.get(r["id"], {})
            if predicate(meta):
                r["metadata"] = meta
                filtered.append(r)
        return filtered
