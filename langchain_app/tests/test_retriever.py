# -*- coding: utf-8 -*-
"""检索器装配与查询理解集成测试（需本地索引，跳过条件：索引不存在）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from config import LC_INDEX_DIR, LC_BM25_PATH

pytestmark = pytest.mark.skipif(
    not (LC_INDEX_DIR.exists() and LC_BM25_PATH.exists()),
    reason="本地索引未构建（先运行 python langchain_app/build_index.py）",
)


@pytest.fixture(scope="module")
def retriever():
    from retrievers import build_hybrid_retriever

    return build_hybrid_retriever(enable_reranker=False)  # 关重排：快且确定


class TestHybridRetriever:
    def test_basic_search(self, retriever):
        docs = retriever.invoke("人参的性味归经是什么？")
        assert len(docs) > 0
        assert docs[0].metadata.get("drug_name", "").startswith("人参")

    def test_drug_filter_override(self, retriever):
        docs = retriever.invoke("人参的性味归经", drug_filter="黄芪")
        drugs = {d.metadata.get("drug_name", "") for d in docs}
        assert all("黄芪" in d or d in "黄芪" for d in drugs)

    def test_horizontal_category_filter(self, retriever):
        docs = retriever.invoke("哪些药材有补气功效？")
        cats = {d.metadata.get("category") for d in docs}
        assert cats == {"药材和饮片"}

    def test_shared_bm25_k_not_mutated(self, retriever):
        """并发安全修复回归：过滤查询不得污染共享 BM25 实例的 k"""
        retriever.invoke("人参", drug_filter="人参")
        assert retriever.bm25.k == 15

    # ------------------------------------------------------------------
    # 2026-09-22 检索缺陷回归（需要真实索引）
    # 三条缺陷原先在 100 题评测集上均测不出，见 docs/08
    # ------------------------------------------------------------------

    # (查询, 应为 top-1 的药品, 不得出现在结果里的"真子串"药名)
    SUBSTRING_CASES = [
        ("双黄连口服液的功能主治是什么？", "双黄连口服液", "黄连"),
        ("复方丹参片的功能与主治", "复方丹参片", "丹参片"),
        ("黄连胶囊的注意事项", "黄连胶囊", "黄连"),
        ("五味子糖浆的用法用量", "五味子糖浆", "五味子"),
    ]

    def test_shorter_substring_drug_not_returned(self, retriever):
        """回归：药品名扩展必须单向，不得把「被查询药名的真子串」拉进结果。

        历史缺陷 `fd in dn or dn in fd`：查"双黄连口服液"会把"黄连"加进过滤集，
        与查询解析器的最长匹配去重相冲突，使 top-1 变成黄连条目。
        """
        for query, expect, shorter in self.SUBSTRING_CASES:
            drugs = [d.metadata.get("drug_name", "") for d in retriever.invoke(query)]
            assert shorter not in drugs, f"{query!r} 结果混入更短药名 {shorter!r}: {drugs}"
            assert drugs[0] == expect, f"{query!r} top-1 应为 {expect}，实为 {drugs[0]!r}"

    def test_filtered_query_returns_full_drug_chunk_set(self, retriever):
        """回归：带 filter 时 fetch_k 必须覆盖全库。

        历史缺陷：langchain FAISS 的过滤是「先取全局 top-fetch_k（默认 20）再过滤」，
        "枸杞子"的 6 个切片只召回 2 条。手撕版对过滤子集做完整检索，两版因此差 22pp。
        """
        docs = retriever.invoke("枸杞子的功能主治")
        assert len(docs) >= 5, f"枸杞子有 6 个切片，只召回 {len(docs)} 条（fetch_k 过小）"

    def test_section_aware_recall_puts_target_section_in_top2(self, retriever):
        """回归：章节感知召回——正文含【目标章节】的候选须排到最前"""
        docs = retriever.invoke("枸杞子的功能主治")
        top2 = [(d.metadata.get("drug_name"), d.metadata.get("section")) for d in docs[:2]]
        assert any("【功能与主治】" in d.page_content for d in docs[:2]), \
            f"top-2 里没有含【功能与主治】的 chunk: {top2}"
