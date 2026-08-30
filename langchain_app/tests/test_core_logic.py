# -*- coding: utf-8 -*-
"""guard / postprocess / eval 判分逻辑单元测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from guard import keyword_guard
from postprocess import postprocess, format_docs, docs_to_sources
from eval import _check_hit, _percentile


class TestKeywordGuard:
    def test_on_topic(self):
        ok, reason = keyword_guard("人参的性味归经是什么？")
        assert ok is True and reason == "keyword_on"

    def test_off_topic(self):
        ok, reason = keyword_guard("今天天气怎么样？")
        assert ok is False and reason == "keyword_off"

    def test_off_topic_beats_on_topic(self):
        # 同时含无关词与"药" → 应拒绝（无关词优先）
        ok, reason = keyword_guard("游戏里的药水怎么合成？")
        assert ok is False and reason == "keyword_off"

    def test_ambiguous(self):
        ok, reason = keyword_guard("帮我写一首诗")
        assert reason == "ambiguous"  # 交给 LLM 层


def _doc(drug="人参", section="完整条目", content="内容"):
    return Document(page_content=content, metadata={
        "chunk_id": "c1", "drug_name": drug, "section": section,
        "category": "药材和饮片", "chunk_id_src": None, "score": 0.9,
    })


class TestPostprocess:
    def test_citations_dedup(self):
        docs = [_doc(), _doc(), _doc(drug="黄芪", section="性状")]
        out = postprocess("回答内容", docs)
        assert out["citations"] == ["药典2020一部-人参-完整条目", "药典2020一部-黄芪-性状"]

    def test_no_duplicate_source_block(self):
        # 回答已含"来源"字样时不追加参考来源列表
        out = postprocess("内容……来源：药典", [_doc()])
        assert "参考来源：" not in out["answer"]

    def test_disclaimer_appended(self):
        out = postprocess("不含提醒的回答", [_doc()])
        assert "遵医嘱" in out["answer"]

    def test_format_docs(self):
        text = format_docs([_doc()])
        assert "【参考资料 1】" in text and "人参" in text

    def test_docs_to_sources_shape(self):
        s = docs_to_sources([_doc()])[0]
        assert s["drug_name"] == "人参" and s["chunk_id"] == "c1"


class TestEvalFormulas:
    def test_bidirectional_substring(self):
        # 双向子串匹配："人参" 应命中 "人参-饮片"
        assert _check_hit("人参-饮片", ["人参"]) is True
        assert _check_hit("人参", ["人参-饮片"]) is True
        assert _check_hit("黄芪", ["人参"]) is False

    def test_auto_hit_without_expected(self):
        # 无期望药品 → 有结果即命中（与主项目口径一致）
        assert _check_hit("任意药品", []) is True

    def test_percentile(self):
        vals = [0.1, 0.2, 0.3, 0.4, 1.0]
        assert _percentile(vals, 50) == 0.3
        assert _percentile(vals, 100) == 1.0
        assert _percentile([], 50) == 0.0
