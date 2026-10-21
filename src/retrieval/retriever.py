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
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    VECTOR_TOP_K, BM25_TOP_K, RRF_K, RRF_TOP_N,
    RERANKER_TOP_K, ENABLE_RERANKER, ENABLE_SECTION_BOOST,
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
    source_scores: Dict[str, float] = field(default_factory=dict)  # 各路原始分（vector 余弦/BM25 分）
    rerank_score: Optional[float] = None    # 重排分数（未重排则为 None）
    table_markdown: Optional[str] = None    # 该块关联的表格（Markdown），无表格则为 None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "table_markdown": self.table_markdown,
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
            "source_scores": self.source_scores,
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

        # 4.5 常用药材优先表（横向查询排序用）
        self.common_herbs = self._load_common_herbs()
        if self.common_herbs:
            print(f"  常用药材优先表: {len(self.common_herbs)} 味")

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
            expanded_drugs = self._expand_drug_variants(filter_drugs, all_drug_names)
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
                    # 检查是否匹配任一过滤药品（单向：查询药名是检索到药名的子串）
                    # 反向匹配（drug in filter_drug）会把更短的无关药名放进来，
                    # 与向量路保持一致的口径，见上方 expanded_drugs 的说明。
                    for filter_drug in filter_drugs:
                        if filter_drug in drug:
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

        # ============ 5.4 条件语义：成分查询 / 性味条件 ============
        # 「哪些药材含挥发油」「XX的含量测定」的目标章节不是功能主治，
        # 需要显式切到鉴别/检查/含量测定等章节。
        is_composition = self._is_composition_query(query)
        parsed["is_composition"] = is_composition
        sexing = self._extract_sexing_condition(query)
        if sexing:
            parsed["sexing_condition"] = {
                "tastes": sorted(sexing[0]),
                "natures": sorted(sexing[1]),
            }

        # ============ 5.5 章节感知召回 ============
        # 查询解析出的目标章节此前只用于药品名过滤，章节信息被浪费，导致
        # "药品召回对了、章节没进 top-k"。这里在候选集内把正文含【目标章节】
        # 标记的 chunk 提前（同命中数保持 RRF 原序），不引入任何模型。
        if ENABLE_SECTION_BOOST and fused:
            if is_composition:
                comp_sections = list(self._COMPOSITION_SECTIONS)
                # 显式问到某章节时（如「鉴别方法」），该章节排到最前
                for sec in reversed(parsed.get("sections") or []):
                    if sec in comp_sections:
                        comp_sections.remove(sec)
                        comp_sections.insert(0, sec)
                fused = self._promote_composition_sections(fused, comp_sections)
            elif parsed.get("sections"):
                fused = self._promote_section_hits(
                    fused, parsed["sections"], parsed.get("drug_names") or []
                )
        # 性味是硬条件，放在章节提升之后做稳定重排（符合的提前，其余保序）
        if fused and sexing:
            fused = self._promote_sexing_hits(fused, sexing[0], sexing[1])
        component_latency["section_boost"] = time.time() - t0

        # ============ 6. 重排 ============
        t0 = time.time()
        if self.reranker and fused:
            reranked = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            # 不重排：fused 已按「章节命中数 → RRF 分数」排好序，直接截断。
            # 未启用章节提升时，等价于按 rrf_score 降序取 top_k（与改造前一致）。
            reranked = fused[:top_k]
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
                table_markdown=meta.get("table_markdown") or None,
                sources=r.get("sources", []),
                source_scores=r.get("source_scores") or {},
                rerank_score=r.get("rerank_score"),
            ))

        # ============ 7.5 横向条件查询聚合 ============
        # 「哪些药材归肺经」这类问题，top-k 相似度检索天然覆盖不全；
        # 改为全库按条件筛选、按药品聚合，聚合结果优先，其余结果补足。
        if parsed.get("is_horizontal"):
            t0 = time.time()
            horizontal = self._horizontal_search(query, parsed, top_k)
            if horizontal:
                # 横向查询只返回符合条件的结果，不用不相关结果补位
                results = horizontal
            component_latency["horizontal_search"] = time.time() - t0

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
    # 横向条件查询：全库筛选 + 按药品聚合
    # ----------------------------------------------------------

    _HORIZONTAL_STOP = re.compile(
        r"(请|帮我|我想|想知道|有哪些|有什么|哪些|哪味|哪个|什么|哪种|"
        r"药材|中药|药物|中成药|方剂|"
        r"具有|含有?|可以|能够|能|可|用来|用于|"
        r"功效|作用|疗效|效果|主治|功能|"
        r"的|是|有|吗|呢|？|\?|，|,|。)"
    )

    _HORIZONTAL_SECTION_PRIORITY = {
        "功能与主治": 0,
        "性味与归经": 1,
        "注意": 2,
        "用法与用量": 3,
        "临床应用": 4,
    }

    # 成分/检测类横向查询的代表块优先级：证据在鉴别/检查/含量测定，不在功能主治
    _COMPOSITION_SECTION_PRIORITY = {
        "含量测定": 0,
        "鉴别": 1,
        "检查": 2,
        "特征图谱": 3,
        "指纹图谱": 4,
        "浸出物": 5,
        "性状": 6,
        "完整条目": 7,
    }

    def _extract_conditions(self, query: str) -> List[str]:
        """从横向查询中提取条件词（归经 / 功效 / 毒性等）"""
        conds = []

        # 归经条件：归肺经 / 入肺经
        for m in re.finditer(r"[归入][\u4e00-\u9fa5]{1,4}经", query):
            conds.append(m.group(0))

        # 毒性：正文里通常写作「有毒」
        if "毒性" in query or "有毒" in query:
            conds.append("有毒")

        # 去掉疑问词与泛称后的核心短语
        cleaned = self._HORIZONTAL_STOP.sub(" ", query)
        for token in re.split(r"[\s、和与及或]+", cleaned):
            token = token.strip()
            if len(token) < 2:
                continue
            # 「毒性」统一成「有毒」，避免「缓解药物毒性」这类描述误命中
            if token in ("毒性", "有毒的"):
                conds.append("有毒")
            else:
                conds.append(token)

        # 去重保序
        seen, out = set(), []
        for c in conds:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    _MERIDIAN_RE = re.compile(r"[归入]([\u4e00-\u9fa5、，,和及]+?)经")
    _MERIDIAN_SPLIT = re.compile(r"[、，,和及]")

    @staticmethod
    def _load_common_herbs() -> Dict[str, int]:
        """加载常用药材优先表，返回 药名 → 优先级序号（越小越常用）"""
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "data" / "lexicon" / "common_herbs.txt"
        )
        if not path.exists():
            return {}
        ranks: Dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            ranks.setdefault(name, len(ranks))
        return ranks

    # 功效类条件词的同义扩展（药典用词与用户说法常不一致）
    _HORIZONTAL_SYNONYMS = {
        "活血化瘀": ["活血", "化瘀", "祛瘀", "散瘀"],
        "安神": ["安神", "宁心", "养心"],
        "利水渗湿": ["利水", "渗湿", "利尿"],
        "清热解毒": ["清热", "解毒"],
        "补气": ["补气", "益气", "大补元气", "补中"],
        "补血": ["补血", "养血"],
        "止咳": ["止咳", "镇咳"],
        "化痰": ["化痰", "祛痰"],
        # 成分查询：药典正文用「小檗碱」，用户常说「黄连素」
        "黄连素": ["黄连素", "小檗碱", "盐酸小檗碱"],
    }

    # 成分/检测类查询要看鉴别、检查、含量测定，不能只搜功能主治。
    # 成分名词用于横向查询关键词；方法/章节词只用于判定查询类型。
    _COMPOSITION_TERMS = (
        '挥发油', '皂苷', '皂昔', '黄酮', '生物碱', '鞣质', '多糖',
        '黄连素', '小檗碱',
    )
    _COMPOSITION_METHOD_HINTS = (
        '含量测定', '鉴别', '色谱', '薄层', '特征图谱', '指纹图谱',
    )
    _COMPOSITION_STRONG_HINTS = _COMPOSITION_TERMS + _COMPOSITION_METHOD_HINTS
    _COMPOSITION_HINTS = _COMPOSITION_STRONG_HINTS + ('含有', '成分')
    _COMPOSITION_SECTIONS = [
        '鉴别', '检查', '含量测定', '浸出物', '性状',
        '特征图谱', '指纹图谱', '完整条目',
    ]

    # 性味条件：甘温 / 苦、寒 / 味甘性温 / 性味：甘、微温
    _SEXING_RE = re.compile(
        r"(?:性味|性|味)?\s*[:：]?\s*"
        r"(?P<tastes>(?:微?[甘苦辛酸咸淡涩][、，,]?\s*)+)"
        r"性?\s*"
        r"(?P<nature>微温|微寒|大热|大寒|温|热|寒|凉|平)"
    )
    _TASTE_SPLIT = re.compile(r"[、，,]")
    _SEXING_MARKERS = ("【性味与归经】", "【性味】", "【归经】")

    # 药典语料 OCR 噪声：「小檗碱」被识别成小案碱/小槩碱/小棄碱…（实测 160+ 块，
    # 无一条写作标准字），GLOB 只锚定首尾字即可覆盖全部变体。
    _COMPOSITION_GLOB_HINTS = {
        "小檗碱": "*小?碱*",
        "盐酸小檗碱": "*小?碱*",
    }

    @classmethod
    def _composition_glob_patterns(cls, terms: List[str]) -> List[str]:
        """成分词的 OCR 容错 GLOB 模式（与关键词 OR 合并）。"""
        return list(dict.fromkeys(
            cls._COMPOSITION_GLOB_HINTS[t]
            for t in terms
            if t in cls._COMPOSITION_GLOB_HINTS
        ))

    # 通用成分词：「麻黄碱」「黄芪甲苷」「氨基酸」这类不可能全部穷举，
    # 在「含/成分/测定」语境下用后缀字兜底识别。
    _COMPOSITION_RE = re.compile(
        r"[\u4e00-\u9fa5]{2,4}(?:素|碱|苷|酮|酸|醇|酯|酚|醛|醚|胺|肽)"
    )

    @classmethod
    def _extract_composition_terms(cls, query: str) -> List[str]:
        """提取查询中的成分名词（固定词表 + 后缀字兜底）。"""
        terms = [t for t in cls._COMPOSITION_TERMS if t in query]
        if any(w in query for w in ('含', '成分', '测定', '鉴别', '色谱')):
            cleaned = re.sub(r"(含有|含|具有|有|能|可)", " ", query)
            for m in cls._COMPOSITION_RE.finditer(cleaned):
                term = m.group(0)
                if len(term) >= 3 and term not in terms:
                    terms.append(term)
        return terms

    @classmethod
    def _is_composition_query(cls, query: str) -> bool:
        """判断是否为成分/检测类查询（问「含什么」「怎么测」而不是「治什么」）。"""
        if cls._extract_composition_terms(query):
            return True
        if any(h in query for h in cls._COMPOSITION_METHOD_HINTS):
            return True
        if not any(h in query for h in ('含有', '成分')):
            return False
        # 「含有哪些功效」这类仍在问功效，不算成分查询
        return not any(w in query for w in ('功效', '功能', '主治', '作用'))

    def _expand_conditions(self, conditions: List[str]) -> List[str]:
        """条件词同义扩展，提高召回覆盖"""
        expanded = []
        for c in conditions:
            expanded.append(c)
            expanded.extend(self._HORIZONTAL_SYNONYMS.get(c, []))
        return list(dict.fromkeys(expanded))

    @classmethod
    def _extract_sexing_condition(cls, query: str):
        """解析「性味甘温 / 苦、寒 / 味甘性温」这类条件，返回 (味集合, 性集合) 或 None"""
        m = cls._SEXING_RE.search(query)
        if not m:
            return None
        tastes = {
            t.strip().lstrip('微')
            for t in cls._TASTE_SPLIT.split(m.group('tastes'))
            if t.strip()
        }
        nature = m.group('nature').lstrip('微大')
        return tastes, {nature}

    @classmethod
    def _parse_sexing_text(cls, text: str):
        """从【性味与归经】正文解析 (味集合, 性)，解析失败返回 (set(), '')"""
        text = text or ''
        # chunk 正文形如「人参 · 性味与归经\n【性味与归经】甘、微苦，微温。归…」
        # 先定位章节标记，避免把药名行当成味描述
        for marker in cls._SEXING_MARKERS:
            idx = text.find(marker)
            if idx >= 0:
                text = text[idx + len(marker):]
                break
        head = re.split(r'[。；;]', text, 1)[0]
        # 去掉「归…经」部分
        head = re.split(r'归', head, 1)[0]
        m = re.search(r'(微温|微寒|大热|大寒|温|热|寒|凉|平)\s*$', head)
        if not m:
            return set(), ''
        nature = m.group(1).lstrip('微大')
        tastes_part = head[:m.start()]
        tastes = {
            t.strip().lstrip('微')
            for t in re.split(r'[、，,]', tastes_part)
            if t.strip()
        }
        return tastes, nature

    @classmethod
    def _sexing_match(
        cls,
        chunk: Dict[str, Any],
        tastes: set,
        natures: set,
    ) -> bool:
        """候选条目的性味是否满足查询条件（味取交集，性须一致）。"""
        meta = chunk.get('metadata') or {}
        content = chunk.get('content') or ''
        if meta.get('section') != '性味与归经' and not any(
            m in content for m in cls._SEXING_MARKERS
        ):
            return False
        chunk_tastes, chunk_nature = cls._parse_sexing_text(content)
        if not chunk_tastes and not chunk_nature:
            return False
        if tastes and not (tastes & chunk_tastes):
            return False
        if natures and chunk_nature not in natures:
            return False
        return True

    @classmethod
    def _promote_sexing_hits(
        cls,
        candidates: List[Dict[str, Any]],
        tastes: set,
        natures: set,
    ) -> List[Dict[str, Any]]:
        """把性味符合条件（如「甘温」）的候选提前，其余保持原序。"""
        scored = [
            (1 if cls._sexing_match(c, tastes, natures) else 0, c)
            for c in candidates
        ]
        if not any(s for s, _ in scored):
            return candidates
        return [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)]

    @staticmethod
    def _section_base(section: str) -> str:
        """去掉 ETL 切片后缀：含量测定-成分1 → 含量测定，鉴别-段2 → 鉴别"""
        return re.split(r'-(?:成分|项|段)', section or '', 1)[0]

    @classmethod
    def _promote_composition_sections(
        cls,
        candidates: List[Dict[str, Any]],
        sections: List[str],
    ) -> List[Dict[str, Any]]:
        """成分/检测类查询按 chunk 的 section 字段提升（含 -成分N / -段N 子块）。

        这类查询的目标章节（鉴别/检查/含量测定）是 ETL 的详述类章节，
        metadata.section 与药典原生章节名同源；不像【功能与主治】那样被
        合并成「临床应用」，因此这里比对 section 字段比数【】标记更准。
        """
        rank = {s: i for i, s in enumerate(sections)}
        default = len(sections)
        scored = [
            (
                rank.get(
                    cls._section_base((c.get('metadata') or {}).get('section') or ''),
                    default,
                ),
                c,
            )
            for c in candidates
        ]
        if all(s == default for s, _ in scored):
            return candidates
        return [c for _, c in sorted(scored, key=lambda x: x[0])]

    def _extract_meridian_targets(self, query: str) -> List[str]:
        """提取归经查询的目标脏腑，如「归肺经」→ ["肺"]"""
        m = self._MERIDIAN_RE.search(query)
        if not m:
            return []
        return [t for t in self._MERIDIAN_SPLIT.split(m.group(1)) if t]

    def _horizontal_search(
        self,
        query: str,
        parsed: Dict[str, Any],
        top_k: int,
    ) -> List[SearchResult]:
        """
        横向条件查询检索。

        向量/BM25 的 top-k 只能覆盖少数条目，回答不了「哪些药材归肺经」。
        这里改为：全库按条件筛选 → 按药品聚合 → 每药取一个代表块，
        按品种热度 + 命中块数排序。

        归经条件单独处理：正文写作「归脾、肺、心、肾经」，
        直接 LIKE「归肺经」匹配不到，需要解析脏腑列表后判断包含关系。
        """
        category = parsed.get("category_filter")
        rows: List[Dict] = []
        match_terms: List[str] = []

        is_composition = bool(parsed.get("is_composition"))
        sexing = self._extract_sexing_condition(query)

        def _row_matches_sexing(row: Dict) -> bool:
            return self._sexing_match(
                {
                    "content": row["content"],
                    "metadata": {"section": row.get("section", "")},
                },
                sexing[0],
                sexing[1],
            )

        targets = self._extract_meridian_targets(query)
        if targets:
            match_terms = list(targets)
            for row in self.meta_store.filter(section="性味与归经", category=category):
                for m in self._MERIDIAN_RE.finditer(row["content"]):
                    organs = self._MERIDIAN_SPLIT.split(m.group(1))
                    if any(t in organs for t in targets):
                        rows.append(row)
                        break
            if sexing:
                rows = [r for r in rows if _row_matches_sexing(r)]
        elif sexing:
            # 「性味甘温的药材」：药典写作「甘、微苦，微温」「苦，寒」…
            # LIKE 原样搜不到，改为全库解析性味章节后按集合判断。
            for row in self.meta_store.filter(section="性味与归经", category=category):
                if _row_matches_sexing(row):
                    rows.append(row)
        else:
            # 成分类查询优先取具体成分名：「含挥发油成分」整词 LIKE 不命中正文
            raw_conditions = (
                self._extract_composition_terms(query) if is_composition else []
            )
            if not raw_conditions:
                raw_conditions = self._extract_conditions(query)
            if not raw_conditions:
                return []
            match_terms = self._expand_conditions(raw_conditions)

            # 毒性只看性味/注意（避免「解毒」等用法描述误命中）；
            # 短条目只有完整条目块，需要一并纳入
            if any(t in ("有毒", "毒性") for t in match_terms):
                sections = ["性味与归经", "注意", "完整条目"]
                section_prefixes = None
            elif is_composition:
                sections = list(self._COMPOSITION_SECTIONS)
                # 「含量测定-成分1」「鉴别-段2」等切片块不在精确 IN 里，需前缀匹配
                section_prefixes = [s for s in sections if s != "完整条目"]
            else:
                sections = list(self._HORIZONTAL_SECTION_PRIORITY.keys()) + ["完整条目"]
                section_prefixes = None

            rows = self.meta_store.search_content(
                keywords=match_terms,
                sections=sections,
                section_prefixes=section_prefixes,
                glob_patterns=self._composition_glob_patterns(match_terms),
                category=category,
            )

        if not rows:
            return []

        groups: Dict[str, List[Dict]] = {}
        for row in rows:
            groups.setdefault(row["drug_name"], []).append(row)

        hot = self.meta_store.drug_chunk_counts(category=category)
        priority = (
            self._COMPOSITION_SECTION_PRIORITY
            if is_composition
            else self._HORIZONTAL_SECTION_PRIORITY
        )

        ranked = []
        for drug, blocks in groups.items():
            blocks.sort(key=lambda b: (
                priority.get(self._section_base(b["section"]), 99),
                -b.get("char_count", 0),
            ))
            best = blocks[0]
            # 条件词在证据块中的命中密度：比单纯条目热度更贴近查询条件
            hits = sum(
                sum(b["content"].count(t) for t in match_terms)
                for b in blocks
            )
            ranked.append((
                self.common_herbs.get(drug, 10 ** 6),  # 常用药优先
                -hits,
                -len(blocks),
                -hot.get(drug, 0),
                len(drug),
                best,
            ))
        ranked.sort(key=lambda x: x[:5])

        results = []
        for i, item in enumerate(ranked[:top_k]):
            block = item[5]
            results.append(SearchResult(
                chunk_id=block["chunk_id"],
                content=block["content"],
                score=1.0 / (i + 1),
                drug_name=block["drug_name"],
                pinyin_name=block.get("pinyin_name", ""),
                latin_name=block.get("latin_name", ""),
                category=block.get("category", ""),
                section=block.get("section", ""),
                chunk_type=block.get("chunk_type", ""),
                is_yinpian=bool(block.get("is_yinpian", False)),
                parent_drug=block.get("parent_drug", ""),
                char_count=block.get("char_count", 0),
                table_markdown=block.get("table_markdown") or None,
                sources=["horizontal"],
            ))
        return results

    # ----------------------------------------------------------
    # 药品名变体扩展
    # ----------------------------------------------------------

    @staticmethod
    def _expand_drug_variants(
        filter_drugs: List[str],
        all_drug_names: List[str],
    ) -> set:
        """把查询到的药品名扩展为"以它为主体的变体"集合（单向子串）。

        只保留 `fd in dn` 方向：如 人参 → {人参, 人参-饮片, 人参叶}。

        **反向（`dn in fd`）是历史缺陷**：查"双黄连口服液"时 `"黄连" in "双黄连口服液"`
        成立，会把黄连/黄连片/黄连胶囊拉进过滤集，抵消查询解析器已做好的最长匹配
        去重（`_extract_drug_names`），使 top-1 落到黄连条目上。
        语料中共 433 个条目的药名包含另一药名，均受此影响。

        见 tests/test_retrieval_fixes.py::TestDrugVariantExpansion
        """
        expanded = set()
        for fd in filter_drugs:
            for dn in all_drug_names:
                if fd in dn:
                    expanded.add(dn)
        return expanded

    # ----------------------------------------------------------
    # 章节感知召回
    # ----------------------------------------------------------

    @staticmethod
    def _section_hits(chunk: Dict[str, Any], expected_sections: List[str]) -> int:
        """统计该候选正文命中了多少个期望章节。

        判据与评测 strict 口径（src/eval/evaluator.py::section_match）保持一致：
        在 chunk **正文**里找 `【期望章节】` 标记，而不是比对 metadata 的 section
        字段——后者是 ETL 的合并桶名（临床应用 / 药品概要 / 完整条目），与药典
        原生章节名不同源，直接比对会大面积假阴性。
        """
        content = chunk.get("content") or ""
        section_field = (chunk.get("metadata") or {}).get("section") or ""
        hits = 0
        for sec in expected_sections:
            if f"【{sec}】" in content:
                hits += 1
            elif sec == "来源" and section_field in ("药品概要", "完整条目"):
                # 【来源】正文在这两种 chunk 里不带【】标记
                hits += 1
        return hits

    @staticmethod
    def _name_exactness(chunk: Dict[str, Any], queried_drugs: List[str]) -> int:
        """药名与查询药名的贴合度：2=该药本身或其饮片，1=含该药名的其他制剂，0=无关。

        查询药名到候选的扩展是**子串**方向（黄芪 → 黄芪-饮片/黄芪颗粒/炙黄芪），
        所以「问黄芪」时候选池里会混入大量含黄芪的成药。strict 口径只认
        「该药本身 + 其饮片」，排序上也应对齐，否则成药会把正主挤下去
        （实测：修正药名后 炙黄芪 成为合法条目，把 黄芪-饮片 从第 1 位压到第 3~5 位，
        连带 8 道黄芪相关题目的 strict@1 一起下降）。
        """
        name = (chunk.get("metadata") or {}).get("drug_name") or ""
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

    @staticmethod
    def _promote_section_hits(
        candidates: List[Dict[str, Any]],
        expected_sections: List[str],
        queried_drugs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """把命中目标章节的候选提到前面；命中数相同时，**药名更贴合**的优先。

        排序键 = (章节命中数, 药名贴合度)，组内保持 RRF 原序。
        只在候选集内重排，不改变召回集合本身；若没有任何候选命中目标章节，
        原序返回（说明章节信息对本次检索无增益，不做无谓扰动）。
        """
        scored = [
            ((Retriever._section_hits(c, expected_sections),
              Retriever._name_exactness(c, queried_drugs or [])), c)
            for c in candidates
        ]
        if not any(key[0] for key, _ in scored):
            return candidates
        # sorted 是稳定排序：键相同时保持 RRF 原序
        return [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)]

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
