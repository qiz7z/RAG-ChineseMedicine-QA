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


# ============================================================
# 表格：不重复拼接 / 超长章节切窗时不把表格切碎
# ============================================================

PRESCRIPTION_TABLE = (
    "| 【处方】黄芪606g | 白花蛇舌草408g |\n"
    "| --- | --- |\n"
    "| 茵陈606g | 金钱草408g |"
)


def _chunk_one(entry):
    return PharmacopoeiaChunker().chunk_entries([entry])


def _prescription_section(content):
    return {
        "section_name": "处方",
        "content": content,
        "table_markdown": PRESCRIPTION_TABLE,
    }


def test_prescription_table_not_duplicated_in_detailed():
    """【处方】的 content 与 table_markdown 是同一份字符串，正文只应出现一次"""
    entry = _entry(
        drug_name="乙肝宁颗粒",
        category_hint="成方制剂和单味制剂",
        sections=[_prescription_section(PRESCRIPTION_TABLE)],
    )
    body = "\n".join(c["content"] for c in _chunk_one(entry))
    assert body.count("黄芪606g") == 1


def test_prescription_table_not_duplicated_in_whole_entry():
    entry = _entry(
        drug_name="小方颗粒",
        category_hint="成方制剂和单味制剂",
        sections=[_prescription_section(PRESCRIPTION_TABLE)],
    )
    chunks = _chunk_one(entry)
    assert chunks[0]["chunk_type"] == "whole_entry"
    assert chunks[0]["content"].count("黄芪606g") == 1


def test_sliding_window_keeps_table_whole():
    """超长章节切窗时表格不参与切分，完整挂在最后一块，其余块不带表格"""
    long_text = "取本品适量，照高效液相色谱法测定。" * 120
    entry = _entry(
        drug_name="女金丸",
        category_hint="成方制剂和单味制剂",
        sections=[_prescription_section(long_text)],
    )
    windows = [c for c in _chunk_one(entry) if c["chunk_type"] == "window"]
    assert len(windows) > 1

    with_table = [c for c in windows if c["table_markdown"]]
    assert len(with_table) == 1, "完整表格只应挂在一块上"
    assert with_table[0]["chunk_id"] == windows[-1]["chunk_id"]
    assert "黄芪606g" in with_table[0]["content"]
    assert all("黄芪606g" not in c["content"] for c in windows[:-1])
