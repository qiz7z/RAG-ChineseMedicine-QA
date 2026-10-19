# -*- coding: utf-8 -*-
"""切片器回归测试：药品概要里的「饮片 / 子剂型」标签。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from etl.chunker import PharmacopoeiaChunker  # noqa: E402


def _entry(**overrides):
    base = {
        "drug_name": "",
        "pinyin_name": "",
        "latin_name": "",
        "category_hint": "药材和饮片",
        "intro_text": "",
        "parent_drug": "",
        "is_yinpian": False,
        "is_sub_formulation": False,
        "sections": [],
    }
    base.update(overrides)
    return base


def _summary_content(entry):
    chunker = PharmacopoeiaChunker()
    chunker._make_summary_chunk(entry)
    assert chunker.chunks, "未生成药品概要 chunk"
    return chunker.chunks[0].content


def test_yinpian_summary_says_yinpian():
    content = _summary_content(_entry(
        drug_name="白芍-饮片", parent_drug="白芍", is_yinpian=True))
    assert "（白芍的饮片）" in content
    assert "子剂型" not in content


def test_sub_formulation_summary_says_sub_formulation():
    content = _summary_content(_entry(
        drug_name="小儿七星茶颗粒", parent_drug="小儿七星茶口服液",
        category_hint="成方制剂和单味制剂", is_sub_formulation=True))
    assert "（小儿七星茶口服液的子剂型）" in content
    assert "饮片" not in content
