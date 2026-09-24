# -*- coding: utf-8 -*-
"""
混合检索器（标准 LangChain 组件搭建）
====================================
技术栈全部为官方组件，自定义部分只保留两块"正常生产实践"的胶水：

  FAISS 向量库(langchain_community) ─┐
                                     ├→ EnsembleRetriever(RRF 融合)
  BM25Retriever(langchain_community)─┘        │
                                              ↓
                       ContextualCompressionRetriever + CrossEncoder 重排
                        （BgeRerankerCompressor：标准 BaseDocumentCompressor
                          接口 + HuggingFaceCrossEncoder 组件）

  PharmacopoeiaRetriever —— 最外层自定义 BaseRetriever：
    按查询动态生成元数据过滤（药品名 $in / 横向分类），通过
    vectorstore.as_retriever(search_kwargs={"filter": fn}) 与
    BM25 结果后过滤接入，其余全部走标准组件。
"""
import re
import sys
import time
import json
import pickle
import logging
from pathlib import Path
from typing import Any, List, Optional

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import jieba
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from config import (
    CHUNKS_JSON_PATH,
    LC_INDEX_DIR,
    LC_BM25_PATH,
    LC_DRUG_NAMES_PATH,
    VECTOR_K,
    BM25_K,
    RRF_WEIGHTS,
    RERANKER_MODEL_PATH,
    RERANKER_TOP_N,
    RERANKER_MAX_LENGTH,
    ENABLE_RERANKER,
    ENABLE_SECTION_BOOST,
    FINAL_TOP_N,
    RAG_DEVICE,
)
from embeddings import build_embeddings
from query_understanding import QueryAnalyzer

logger = logging.getLogger(__name__)
logging.getLogger("langchain_community.vectorstores.FAISS").setLevel(logging.ERROR)

# chunks.json 中写入 Document.metadata 的字段
METADATA_FIELDS = [
    "chunk_id", "drug_name", "pinyin_name", "latin_name", "category",
    "section", "chunk_type", "is_yinpian", "is_sub_formulation",
    "parent_drug", "char_count",
]


# ------------------------------------------------------------
# BM25 分词（模块级函数，保证可被 pickle 按引用保存/恢复）
# ------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """jieba 分词，过滤纯标点/空白 token（中文检索的标准前处理）"""
    return [t for t in jieba.lcut(text) if re.search(r"\w", t)]


# ------------------------------------------------------------
# 自定义重排压缩器（标准 BaseDocumentCompressor 接口）
# ------------------------------------------------------------

class BgeRerankerCompressor(BaseDocumentCompressor):
    """CrossEncoder 重排压缩器：对 query-doc 对打分，返回 top_n"""

    cross_encoder: Any = None
    top_n: int = 5

    def compress_documents(
        self, documents: List[Document], query: str, callbacks=None, **kwargs
    ) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, d.page_content) for d in documents]
        scores = self.cross_encoder.score(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        out = []
        for doc, score in ranked[: self.top_n]:
            meta = dict(doc.metadata)
            meta["rerank_score"] = float(score)
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out


# ------------------------------------------------------------
# BM25 结果后过滤（标准 BaseRetriever 装饰模式）
# ------------------------------------------------------------

class FilteredRetriever(BaseRetriever):
    """包装一个检索器，按谓词过滤其返回文档（用于 BM25 路的元数据过滤）"""

    base: Any = None
    predicate: Any = None

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        docs = self.base.invoke(query)
        return [d for d in docs if self.predicate(d.metadata)]


# ------------------------------------------------------------
# 主检索器：动态元数据过滤 + 标准混合管线
# ------------------------------------------------------------

def _make_filter(expanded_drugs: Optional[set], category: Optional[str]):
    """构造 LC FAISS 的元数据过滤谓词（drug_name $in + category 等值）"""

    def _match(meta: dict) -> bool:
        if expanded_drugs and meta.get("drug_name") not in expanded_drugs:
            return False
        if category and meta.get("category") != category:
            return False
        return True

    return None if (not expanded_drugs and not category) else _match


def _section_hits(doc: Document, expected_sections: List[str]) -> int:
    """统计该文档正文命中了多少个期望章节。

    判据与评测 strict 口径（langchain_app/eval.py::_section_match）一致：
    在正文里找 `【期望章节】` 标记，而不是比对 metadata.section——后者是 ETL 的
    合并桶名（临床应用 / 药品概要 / 完整条目），与药典原生章节名不同源。
    """
    content = doc.page_content or ""
    section_field = (doc.metadata or {}).get("section") or ""
    hits = 0
    for sec in expected_sections:
        if f"【{sec}】" in content:
            hits += 1
        elif sec == "来源" and section_field in ("药品概要", "完整条目"):
            # 【来源】正文在这两种 chunk 里不带【】标记
            hits += 1
    return hits


def _name_exactness(doc: Document, queried_drugs: List[str]) -> int:
    """药名与查询药名的贴合度：2=该药本身或其饮片，1=含该药名的其他制剂，0=无关。

    查询药名到候选的扩展是**子串**方向（黄芪 → 黄芪-饮片/黄芪颗粒/炙黄芪），
    所以「问黄芪」时候选池会混入大量含黄芪的成药。strict 口径只认「该药本身 +
    其饮片」，排序上也应对齐，否则成药会把正主挤下去（实测修正药名后
    `炙黄芪` 成为合法条目，把 `黄芪-饮片` 从第 1 位压到第 3~5 位）。
    与主项目 src/retrieval/retriever.py::_name_exactness 同口径。
    """
    name = (doc.metadata or {}).get("drug_name") or ""
    if not name or not queried_drugs:
        return 0
    best = 0
    for q in queried_drugs:
        if not q:
            continue
        if name == q or name == f"{q}-饮片":
            return 2
        if q in name:
            best = max(best, 1)
    return best


def _promote_section_hits(
    docs: List[Document],
    expected_sections: List[str],
    queried_drugs: Optional[List[str]] = None,
) -> List[Document]:
    """把命中目标章节的文档提到最前；命中数相同时，**药名更贴合**的优先。

    排序键 = (章节命中数, 药名贴合度)，组内保持原序。
    只在候选集内重排，不改变召回集合本身；若无人命中目标章节则原序返回。
    """
    scored = [
        ((_section_hits(d, expected_sections), _name_exactness(d, queried_drugs or [])), d)
        for d in docs
    ]
    if not any(key[0] for key, _ in scored):
        return docs
    # sorted 稳定：键相同时保持原序
    return [d for _, d in sorted(scored, key=lambda x: x[0], reverse=True)]


class PharmacopoeiaRetriever(BaseRetriever):
    """药典混合检索器（标准组件 + 查询级动态过滤）"""

    vectorstore: Any = None
    bm25: Any = None
    analyzer: Any = None
    enable_reranker: bool = ENABLE_RERANKER
    enable_bm25: bool = True
    reranker_top_n: int = RERANKER_TOP_N
    final_top_n: int = FINAL_TOP_N
    k_vector: int = VECTOR_K
    k_bm25: int = BM25_K

    # ----------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        drug_filter: Optional[str] = None,
    ) -> List[Document]:
        t0 = time.time()
        info = self.analyzer.analyze(query)
        if drug_filter:
            # 手动药品过滤覆盖自动解析结果（/api/v1/search?drug= 用法）
            info.drug_names = [drug_filter]
            info.expanded_drugs = self.analyzer.expand([drug_filter])
            info.is_horizontal = False
            info.category = None
        filt = _make_filter(info.expanded_drugs, info.category)

        if filt is None:
            # 无过滤：标准 Ensemble（内置 RRF 融合）
            legs = [self.vectorstore.as_retriever(search_kwargs={"k": self.k_vector})]
            if self.enable_bm25:
                legs.append(self.bm25)
            ensemble = EnsembleRetriever(
                retrievers=legs, weights=list(RRF_WEIGHTS[: len(legs)])
            )
            docs = ensemble.invoke(query)
        else:
            # 带过滤：向量路走 search_kwargs.filter，BM25 路走结果后过滤
            v = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": self.k_vector,
                    "filter": filt,
                    # ⚠️ 必须放大 fetch_k。langchain_community 的 FAISS 过滤是
                    # 「先按向量取全局 top-fetch_k，再对结果套 filter」
                    # （见 faiss.py：`index.search(vector, k if filter is None else fetch_k)`，
                    #  fetch_k 默认仅 20）。于是"该药品的切片不在全局 top-20"时会整批丢失，
                    #  召回数远小于实际条数——这是标准版 strict 长期低于手撕版的主因
                    #  （手撕版是对过滤后的子集做完整检索）。
                    #  取全库规模，语义上等价于"先过滤再检索"；IndexFlatIP 暴力扫描
                    #  11k×1024 仅数毫秒，代价可接受。
                    "fetch_k": self.vectorstore.index.ntotal,
                }
            )
            legs = [v]
            if self.enable_bm25 and self.bm25 is not None:
                # 过滤损失召回，扩大 BM25 召回量补偿。
                # 用浅拷贝而非直接改共享实例的 k（pydantic model_copy 只复制
                # 字段，重量级 BM25Okapi 底层索引共享）——避免并发请求互相踩 k
                bm25_scaled = self.bm25.model_copy()
                bm25_scaled.k = self.k_bm25 * 4
                legs.append(FilteredRetriever(base=bm25_scaled, predicate=filt))
            ensemble = EnsembleRetriever(
                retrievers=legs, weights=list(RRF_WEIGHTS[: len(legs)])
            )
            docs = ensemble.invoke(query)
            if self.enable_bm25 and self.bm25 is not None:
                self.bm25.k = self.k_bm25

        # 章节感知召回：把正文含【目标章节】标记的候选提前（组内保持原序）。
        # 放在截断/重排之前，保证 final_top_n 截的是已提升过的顺序。
        if ENABLE_SECTION_BOOST and info.sections and docs:
            docs = _promote_section_hits(docs, info.sections, info.drug_names)

        logger.info("混合检索 %d 条 (解析+召回 %.2fs)", len(docs), time.time() - t0)

        # 重排（CrossEncoder 压缩器）
        if self.enable_reranker and docs:
            compressor = BgeRerankerCompressor(
                cross_encoder=_get_cross_encoder(), top_n=self.final_top_n
            )
            docs = compressor.compress_documents(docs, query)
        elif docs:
            # 关重排：EnsembleRetriever 返回两路并集（实测单查询可达 50+ 条），
            # 必须按 final_top_n 截断——否则下游会把几十条塞进 LLM 上下文，
            # 且与手撕版（两种模式都固定返回 RERANKER_TOP_K 条）不可比。
            docs = docs[: self.final_top_n]

        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager, **kwargs):
        return self._get_relevant_documents(query, run_manager=None, **kwargs)


# CrossEncoder 进程内单例（模型加载约 5s，避免每查询重复加载）
_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        # RAG_DEVICE 为空时沿用 HuggingFaceCrossEncoder 的默认设备（自动选 CUDA）
        kwargs = {"device": RAG_DEVICE} if RAG_DEVICE else {}
        _CROSS_ENCODER = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL_PATH,
            model_kwargs={**kwargs, "max_length": RERANKER_MAX_LENGTH},
        )
        logger.info("重排模型已加载 (device=%s, max_length=%d)",
                    RAG_DEVICE or "auto", RERANKER_MAX_LENGTH)
    return _CROSS_ENCODER


# ------------------------------------------------------------
# 装配入口
# ------------------------------------------------------------

def load_documents():
    """chunks.json → Document 列表"""
    chunks = json.loads(CHUNKS_JSON_PATH.read_text(encoding="utf-8"))
    documents = []
    for c in chunks:
        content = (c.get("content") or "").strip()
        if not content:
            continue
        metadata = {k: c.get(k) for k in METADATA_FIELDS}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_hybrid_retriever(
    enable_reranker: bool = None,
    enable_bm25: bool = True,
) -> PharmacopoeiaRetriever:
    """加载索引并装配检索器

    Args:
        enable_reranker: None（默认）= 取 config.ENABLE_RERANKER（当前为关闭）；
                         显式传 True/False 可覆盖（评测消融用）。
        enable_bm25: 是否启用 BM25 路（消融用）。
    """
    if enable_reranker is None:
        enable_reranker = ENABLE_RERANKER
    logger.info("加载 FAISS 索引: %s", LC_INDEX_DIR)
    vectorstore = FAISS.load_local(
        str(LC_INDEX_DIR),
        build_embeddings(),
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS 就绪: %d 条", vectorstore.index.ntotal)

    bm25 = None
    if enable_bm25:
        with open(LC_BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        logger.info("BM25 就绪: %d 条", len(bm25.docs))

    drug_names = json.loads(LC_DRUG_NAMES_PATH.read_text(encoding="utf-8"))
    analyzer = QueryAnalyzer(drug_names)

    return PharmacopoeiaRetriever(
        vectorstore=vectorstore,
        bm25=bm25,
        analyzer=analyzer,
        enable_reranker=enable_reranker,
        enable_bm25=enable_bm25,
    )
