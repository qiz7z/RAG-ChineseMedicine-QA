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
