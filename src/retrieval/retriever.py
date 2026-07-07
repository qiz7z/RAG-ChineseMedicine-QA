# -*- coding: utf-8 -*-
"""
检索引擎（主模块）
==================
将查询解析、向量检索、BM25检索、RRF融合、重排模型整合为统一接口。

完整检索流程：
  ┌──────────────────────────────────────────────────────────────┐
  │  用户查询："人参的性味归经是什么？"                            │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  1. 查询解析（QueryParser）                                   │
  │     药品名=["人参"]  章节=["性味与归经"]  意图="属性查询"      │
  │     filter={"drug_name":"人参"}                               │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────┬───────────────────────────────────────────────┐
  │  2a. 向量检索 │  2b. BM25 检索                                │
  │  (Chroma)    │  (jieba + BM25Okapi)                          │
  │  top_k=15    │  top_k=15                                     │
  │  + 元数据过滤 │                                               │
  └──────┬───────┴────────┬──────────────────────────────────────┘
         │                │
         ▼                ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  3. RRF 融合 (k=60)                                          │
  │     合并两路结果 → top_n=30 候选                              │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  4. 重排（BGE-Reranker-v2-m3）                                │
  │     对 30 个 (query, doc) 对逐一打分 → top_k=5              │
  └──────────────────────────┬───────────────────────────────────┘
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  5. 元数据补全（SQLite）+ 上下文组装                          │
  │     返回结构化结果 + 格式化的上下文文本                        │
  └──────────────────────────────────────────────────────────────┘
"""
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    VECTOR_TOP_K, BM25_TOP_K, RRF_K, RRF_TOP_N,
    RERANKER_TOP_K, ENABLE_RERANKER,
    CONTEXT_MAX_CHARS_PER_CHUNK, CONTEXT_MAX_TOTAL_CHARS,
    BM25_INDEX_PATH, SQLITE_DB_PATH,
)
from indexing.embedder import Embedder
from indexing.vector_index import ChromaVectorIndex
from indexing.keyword_index import BM25Index
from indexing.fusion import rrf_fusion
from indexing.metadata_store import MetadataStore
from retrieval.query_parser import QueryParser
from retrieval.reranker import Reranker


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class SearchResult:
    """
    单条检索结果。

    封装了 chunk 的所有信息，便于下游生成模块使用。
    """
    chunk_id: str                           # chunk 唯一 ID
    content: str                            # chunk 文本内容
    score: float                            # 最终排序分数（重排分数或 RRF 分数）
    drug_name: str = ""                     # 药品名
    pinyin_name: str = ""                   # 拼音名
    latin_name: str = ""                    # 拉丁名
    category: str = ""                      # 分类（药材/饮片/提取物/成方制剂）
    section: str = ""                       # 章节（性味与归经/含量测定等）
    chunk_type: str = ""                    # chunk 类型
    is_yinpian: bool = False                # 是否饮片
    parent_drug: str = ""                   # 父级药品名（子方剂用）
    char_count: int = 0                     # 字符数
    sources: List[str] = field(default_factory=list)  # 命中来源（vector/bm25）
    rerank_score: Optional[float] = None    # 重排分数（未重排则为 None）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "drug_name": self.drug_name,
            "pinyin_name": self.pinyin_name,
            "latin_name": self.latin_name,
            "category": self.category,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "is_yinpian": self.is_yinpian,
            "parent_drug": self.parent_drug,
            "char_count": self.char_count,
            "sources": self.sources,
            "rerank_score": self.rerank_score,
        }


@dataclass
class RetrievalResponse:
    """
    检索完整响应，包含解析信息、检索结果和上下文。
    """
    query: str                              # 原始查询
    parsed: Dict[str, Any]                  # 查询解析结果
    results: List[SearchResult]             # 检索结果列表
    context: str                            # 组装的上下文文本（供 LLM 生成用）
    latency: float                          # 检索总耗时（秒）
    component_latency: Dict[str, float]     # 各阶段耗时

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "parsed": self.parsed,
            "results": [r.to_dict() for r in self.results],
            "context": self.context,
            "latency": self.latency,
            "component_latency": self.component_latency,
        }


# ============================================================
# 检索引擎
# ============================================================

class Retriever:
    """
    检索引擎：统一协调多路召回、融合、重排、上下文组装。

    使用方式：
        retriever = Retriever()
        response = retriever.search("人参的性味归经是什么？")
        print(response.context)     # 获取上下文文本
        for r in response.results:  # 遍历检索结果
            print(r.drug_name, r.section, r.score)
    """

    def __init__(
        self,
        enable_reranker: bool = None,
        enable_filter: bool = True,
    ):
        """
        Args:
            enable_reranker: 是否启用重排模型，默认从 config 读取
            enable_filter: 是否启用基于查询解析的元数据过滤
        """
        self.enable_reranker = ENABLE_RERANKER if enable_reranker is None else enable_reranker
        self.enable_filter = enable_filter

        print("=" * 60)
        print("初始化检索引擎")
        print("=" * 60)

        # 1. 加载 Embedder
        t0 = time.time()
        self.embedder = Embedder()
        print(f"  Embedder 加载完成 ({time.time() - t0:.1f}s)")

        # 2. 加载向量索引
        t0 = time.time()
        self.vector_index = ChromaVectorIndex()
        print(f"  向量索引加载完成: {self.vector_index.count()} 条 ({time.time() - t0:.1f}s)")

        # 3. 加载 BM25 索引
        t0 = time.time()
        self.bm25_index = BM25Index()
        self.bm25_index.load(BM25_INDEX_PATH)
        print(f"  BM25 索引加载完成: {self.bm25_index.count()} 条 ({time.time() - t0:.1f}s)")

        # 4. 加载元数据存储
        t0 = time.time()
        self.meta_store = MetadataStore(SQLITE_DB_PATH)
        print(f"  元数据存储加载完成: {self.meta_store.count()} 条 ({time.time() - t0:.1f}s)")

        # 5. 加载查询解析器（从 SQLite 获取所有药品名）
        t0 = time.time()
        drug_names = self.meta_store.get_all_drug_names()
        self.query_parser = QueryParser(drug_names=drug_names)
        print(f"  查询解析器初始化完成: {len(drug_names)} 个药品名 ({time.time() - t0:.1f}s)")

        # 5.5 将药品名添加到 jieba 自定义词典，确保查询时药品名不被切分
        from indexing.keyword_index import JiebaTokenizer
        JiebaTokenizer.add_custom_words(drug_names)

        # 6. 加载重排模型（可选）
        self.reranker = None
        if self.enable_reranker:
            t0 = time.time()
            self.reranker = Reranker()
            print(f"  重排模型加载完成 ({time.time() - t0:.1f}s)")
        else:
            print("  重排模型: 未启用")

        print("=" * 60)
        print("检索引擎初始化完成!")
        print("=" * 60)

    # ----------------------------------------------------------
    # 核心检索方法
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = None,
        drug_filter: Optional[str] = None,
    ) -> RetrievalResponse:
        """
        执行完整检索流程。

        Args:
            query: 用户查询文本
            top_k: 最终返回的结果数，默认从 config 读取
            drug_filter: 手动指定药品名过滤（覆盖自动解析结果）

        Returns:
            RetrievalResponse 对象，包含解析信息、检索结果和上下文
        """
        total_start = time.time()
        component_latency = {}
        top_k = top_k or RERANKER_TOP_K

        # ============ 1. 查询解析 ============
        t0 = time.time()
        parsed = self.query_parser.parse(query)

        # 手动药品过滤覆盖
        if drug_filter:
            parsed["drug_names"] = [drug_filter]
            parsed["filter"] = {"drug_name": drug_filter}

        component_latency["query_parsing"] = time.time() - t0

        # ============ 2. 向量检索 ============
        t0 = time.time()
        query_emb = self.embedder.encode_query(query)

        # 元数据过滤：识别到药品名时始终过滤
        # 注意：FAISS 过滤器做精确匹配，需要扩展为 $in 以包含 "-饮片" 等变体
        chroma_filter = None
        if self.enable_filter and parsed["drug_names"]:
            filter_drugs = parsed["drug_names"]
            # 从 SQLite 获取所有匹配的药品名（包括 "-饮片" 等变体）
            all_drug_names = self.meta_store.get_all_drug_names()
            expanded_drugs = set()
            for fd in filter_drugs:
                for dn in all_drug_names:
                    if fd in dn or dn in fd:
                        expanded_drugs.add(dn)
            if expanded_drugs:
                chroma_filter = {"drug_name": {"$in": list(expanded_drugs)}}

        # 横向条件查询：按 category 过滤
        if parsed.get("is_horizontal") and parsed.get("category_filter"):
            if chroma_filter is None:
                chroma_filter = {"category": parsed["category_filter"]}
            else:
                # 同时有药品名过滤和分类过滤时，合并条件
                chroma_filter = {
                    "$and": [
                        chroma_filter,
                        {"category": parsed["category_filter"]},
                    ]
                }

        vector_results = self.vector_index.query(
            query_embedding=query_emb,
            top_k=VECTOR_TOP_K,
            where=chroma_filter,
        )
        component_latency["vector_search"] = time.time() - t0

        # ============ 3. BM25 检索 ============
        t0 = time.time()
        # 如果识别到药品名，增大 BM25 召回量以补偿过滤后的损失
        bm25_top_k = BM25_TOP_K
        if self.enable_filter and parsed["drug_names"]:
            bm25_top_k = BM25_TOP_K * 4  # 扩大召回，过滤后仍保证足够候选
        bm25_results = self.bm25_index.query(query, top_k=bm25_top_k)

        # 如果识别到药品名（单个或多个），对 BM25 结果进行元数据过滤
        # BM25 索引本身不存储元数据，需要从 SQLite 查询后过滤
        if self.enable_filter and parsed["drug_names"]:
            filter_drugs = parsed["drug_names"]
            if bm25_results:
                bm25_ids = [r["id"] for r in bm25_results]
                bm25_metas = self.meta_store.get_by_ids(bm25_ids)
                bm25_meta_map = {m["chunk_id"]: m for m in bm25_metas}
                filtered_bm25 = []
                for r in bm25_results:
                    meta = bm25_meta_map.get(r["id"], {})
                    drug = meta.get("drug_name", "")
                    # 检查是否匹配任一过滤药品（精确匹配或子串匹配）
                    for filter_drug in filter_drugs:
                        if filter_drug in drug or drug in filter_drug:
                            r["metadata"] = meta
                            filtered_bm25.append(r)
                            break
                bm25_results = filtered_bm25

        # 横向条件查询：对 BM25 结果按 category 过滤
        if self.enable_filter and parsed.get("is_horizontal") and parsed.get("category_filter"):
            cat_filter = parsed["category_filter"]
            if bm25_results:
                bm25_ids = [r["id"] for r in bm25_results]
                bm25_metas = self.meta_store.get_by_ids(bm25_ids)
                bm25_meta_map = {m["chunk_id"]: m for m in bm25_metas}
                filtered_bm25 = []
                for r in bm25_results:
                    meta = bm25_meta_map.get(r["id"], {})
                    if meta.get("category", "") == cat_filter:
                        r["metadata"] = meta
                        filtered_bm25.append(r)
                bm25_results = filtered_bm25

        component_latency["bm25_search"] = time.time() - t0

        # ============ 4. RRF 融合 ============
        t0 = time.time()
        fused = rrf_fusion(
            [vector_results, bm25_results],
            k=RRF_K,
            top_n=RRF_TOP_N,
        )
        component_latency["rrf_fusion"] = time.time() - t0

        # ============ 5. 元数据补全 ============
        t0 = time.time()
        if fused:
            chunk_ids = [r["id"] for r in fused]
            metas = self.meta_store.get_by_ids(chunk_ids)
            meta_map = {m["chunk_id"]: m for m in metas}
        else:
            meta_map = {}

        # 将元数据合并到融合结果中
        for r in fused:
            meta = meta_map.get(r["id"], {})
            r["metadata"] = meta
        component_latency["metadata_enrichment"] = time.time() - t0

        # ============ 6. 重排 ============
        t0 = time.time()
        if self.reranker and fused:
            reranked = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            # 不重排，直接用 RRF 分数排序
            reranked = sorted(fused, key=lambda x: x.get("rrf_score", 0), reverse=True)[:top_k]
        component_latency["reranking"] = time.time() - t0

        # ============ 7. 转换为 SearchResult ============
        results = []
        for r in reranked:
            meta = r.get("metadata", {})
            results.append(SearchResult(
                chunk_id=r["id"],
                content=r.get("content", ""),
                score=r.get("rerank_score") if self.reranker else r.get("rrf_score", 0),
                drug_name=meta.get("drug_name", ""),
                pinyin_name=meta.get("pinyin_name", ""),
                latin_name=meta.get("latin_name", ""),
                category=meta.get("category", ""),
                section=meta.get("section", ""),
                chunk_type=meta.get("chunk_type", ""),
                is_yinpian=bool(meta.get("is_yinpian", False)),
                parent_drug=meta.get("parent_drug", ""),
                char_count=meta.get("char_count", 0),
                sources=r.get("sources", []),
                rerank_score=r.get("rerank_score"),
            ))

        # ============ 8. 上下文组装 ============
        t0 = time.time()
        context = self._assemble_context(results, parsed)
        component_latency["context_assembly"] = time.time() - t0

        total_latency = time.time() - total_start

        return RetrievalResponse(
            query=query,
            parsed=parsed,
            results=results,
            context=context,
            latency=total_latency,
            component_latency=component_latency,
        )

    # ----------------------------------------------------------
    # 上下文组装
    # ----------------------------------------------------------

    def _assemble_context(
        self,
        results: List[SearchResult],
        parsed: Dict[str, Any],
    ) -> str:
        """
        将检索结果组装为 LLM 可用的上下文文本。

        格式：
            【检索结果 1】药品：人参 | 章节：性味与归经
            人参 甘、微苦，微温。归脾、肺、心经。
            ---
            【检索结果 2】药品：人参 | 章节：功能与主治
            大补元气，复脉固脱，补脾益肺，生津养血，安神益智。
            ---

        Args:
            results: 检索结果列表
            parsed: 查询解析结果

        Returns:
            格式化的上下文文本
        """
        if not results:
            return "未找到相关内容。"

        parts = []
        total_chars = 0

        for i, r in enumerate(results, 1):
            # 截断过长的内容
            content = r.content[:CONTEXT_MAX_CHARS_PER_CHUNK]

            # 检查总字符数限制
            if total_chars + len(content) > CONTEXT_MAX_TOTAL_CHARS:
                remaining = CONTEXT_MAX_TOTAL_CHARS - total_chars
                if remaining > 50:  # 至少保留 50 字符才添加
                    content = content[:remaining]
                    parts.append(self._format_chunk(i, r, content))
                break

            parts.append(self._format_chunk(i, r, content))
            total_chars += len(content)

        return "\n---\n".join(parts)

    def _format_chunk(self, index: int, result: SearchResult, content: str) -> str:
        """格式化单个 chunk"""
        header_parts = [f"【检索结果 {index}】"]

        if result.drug_name:
            header_parts.append(f"药品：{result.drug_name}")
        if result.section:
            header_parts.append(f"章节：{result.section}")
        if result.category and result.category != result.drug_name:
            header_parts.append(f"分类：{result.category}")

        header = " | ".join(header_parts)
        content_clean = content.strip().replace("\n\n", "\n")

        return f"{header}\n{content_clean}"

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    def search_simple(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        简化版检索接口，直接返回结果列表（字典格式）。

        适用于不需要详细响应信息的场景。
        """
        response = self.search(query, top_k=top_k)
        return [r.to_dict() for r in response.results]

    def get_context(self, query: str, top_k: int = None) -> str:
        """
        仅获取上下文文本（供 LLM 生成用）。
        """
        response = self.search(query, top_k=top_k)
        return response.context

    def close(self):
        """关闭数据库连接"""
        if self.meta_store:
            self.meta_store.close()


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("\n" + "=" * 60)
    print("检索引擎测试")
    print("=" * 60)

    retriever = Retriever()

    test_queries = [
        "人参的性味归经是什么？",
        "双黄连口服液的含量测定方法",
        "哪些药材含有挥发油成分？",
        "高效液相色谱法的操作步骤",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: '{query}'")
        print("-" * 60)

        response = retriever.search(query)

        print(f"  解析: 药品={response.parsed['drug_names']}, "
              f"章节={response.parsed['sections']}, "
              f"意图={response.parsed['intent']}")
        print(f"  耗时: {response.latency:.3f}s")
        print(f"  各阶段: {response.component_latency}")
        print(f"  结果数: {len(response.results)}")

        for i, r in enumerate(response.results, 1):
            score = f"rerank={r.rerank_score:.4f}" if r.rerank_score else f"rrf={r.score:.6f}"
            print(f"    {i}. [{score}] {r.drug_name} > {r.section}")
            print(f"       {r.content[:80]}...")

    retriever.close()
    print(f"\n{'='*60}")
    print("✅ 检索引擎测试完成!")
