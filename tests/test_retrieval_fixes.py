# -*- coding: utf-8 -*-
"""
检索缺陷回归测试（2026-09-22）
=============================
为本次修掉的 3 类检索缺陷各加一层**不变量**断言。

为什么需要这组测试：这些缺陷在原来的 100 题评测集上**都测不出来**——
例如"药品名双向扩展"在 100 题上命中差异为 0/100（语料里没有"药名包含另一药名"
的题），只能靠人工探测发现。加题之后 Hit@5 依然不敏感（旧代码同样 6/6，
只是 top-1 变成了更短的药名），所以必须直接断言不变量。

本文件**不需要索引与模型**，只测纯逻辑，可在 CI 秒级回归。
背景与实测数据见 docs/08_检索缺陷修复与口径澄清.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval.query_parser import QueryParser          # noqa: E402
from retrieval.retriever import Retriever               # noqa: E402


# ============================================================
# 1. 横向查询的分类过滤值
# ============================================================
# 历史缺陷：_detect_category_filter 返回简称 "药材"/"成方制剂"，
# 而 chunks 里 category 的实际值是 "药材和饮片"/"成方制剂和单味制剂"，
# 过滤条件恒不命中 → 横向查询召回为空（实测 0 条）。

class TestCategoryFilterValues:
    ACTUAL_VALUES = {"药材和饮片", "成方制剂和单味制剂", "植物油脂和提取物"}

    def test_returns_actual_metadata_value(self):
        qp = QueryParser(drug_names=["人参"])
        assert qp._detect_category_filter("哪些药材含有挥发油成分？") == "药材和饮片"
        assert qp._detect_category_filter("哪些方剂能清热解毒") == "成方制剂和单味制剂"

    def test_never_returns_abbreviation(self):
        qp = QueryParser(drug_names=[])
        for q in ("哪些药材有补气功效", "哪些中成药能止咳", "什么中药能安神", "玫瑰花"):
            cat = qp._detect_category_filter(q)
            assert cat is None or cat in self.ACTUAL_VALUES, f"{q!r} → {cat!r} 不是实际分类值"

    def test_category_map_values_are_real_categories(self):
        assert set(QueryParser.CATEGORY_MAP.values()) <= self.ACTUAL_VALUES


# ============================================================
# 2. 药品名变体扩展的方向
# ============================================================
# 历史缺陷：`fd in dn or dn in fd` 双向扩展。查"双黄连口服液"时
# `"黄连" in "双黄连口服液"` 成立，会把黄连系列拉进过滤集，
# 抵消查询解析器已做好的最长匹配去重 → top-1 落到黄连。
# 语料中共 433 个条目的药名包含另一药名。

class TestDrugVariantExpansion:
    ALL = ["人参", "人参-饮片", "人参叶", "黄连", "黄连片", "黄连胶囊",
           "双黄连口服液", "丹参", "丹参片", "复方丹参片", "五味子", "五味子糖浆"]

    def test_expands_to_longer_variants(self):
        got = Retriever._expand_drug_variants(["人参"], self.ALL)
        assert {"人参", "人参-饮片", "人参叶"} <= got

    def test_does_not_pull_in_shorter_substring_drug(self):
        # 核心回归 case
        got = Retriever._expand_drug_variants(["双黄连口服液"], self.ALL)
        assert "黄连" not in got and "黄连片" not in got and "黄连胶囊" not in got

    def test_does_not_pull_in_shorter_substring_drug_2(self):
        got = Retriever._expand_drug_variants(["复方丹参片"], self.ALL)
        assert "丹参" not in got and "丹参片" not in got

    def test_invariant_no_result_is_proper_substring_of_query(self):
        """不变量：扩展结果中不得存在被查询药名的真子串（反向匹配即违反）"""
        for q in ("双黄连口服液", "复方丹参片", "黄连胶囊", "五味子糖浆"):
            got = Retriever._expand_drug_variants([q], self.ALL)
            bad = {d for d in got if d != q and d in q}
            assert not bad, f"{q!r} 扩展出真子串 {bad}"


# ============================================================
# 3. 章节感知召回
# ============================================================
# 历史缺陷：QueryParser 解析出的 sections 被弃用，导致"药名召回对了、
# 章节没进 top-k"。修法：在候选集内把正文含【目标章节】的 chunk 提前。
# 判据与评测 strict 口径一致——找正文里的【】标记，而不是比对 metadata.section
# （后者是 ETL 合并桶名：临床应用 / 药品概要 / 完整条目）。

def _chunk(content, section="完整条目", chunk_type="whole_entry"):
    return {"content": content, "metadata": {"section": section, "chunk_type": chunk_type}}


class TestSectionAwareRecall:
    def test_counts_marker_hits(self):
        c = _chunk("【性状】圆柱形。\n【功能与主治】补气。")
        assert Retriever._section_hits(c, ["功能与主治"]) == 1
        assert Retriever._section_hits(c, ["功能与主治", "性状"]) == 2
        assert Retriever._section_hits(c, ["含量测定"]) == 0

    def test_section_field_must_not_be_used(self):
        c = _chunk("【性状】圆柱形。", section="临床应用")
        assert Retriever._section_hits(c, ["性味与归经", "功能与主治"]) == 0

    def test_source_special_case(self):
        # 【来源】正文在 summary / whole_entry 里不带【】标记
        assert Retriever._section_hits(
            _chunk("概述：为菊科植物…", "药品概要", "summary"), ["来源"]) == 1
        assert Retriever._section_hits(
            _chunk("正文", "性状", "detailed"), ["来源"]) == 0

    def test_promotes_matching_candidate_to_front(self):
        cands = [_chunk("【含量测定】…"), _chunk("【功能与主治】补气。"), _chunk("【性状】…")]
        out = Retriever._promote_section_hits(cands, ["功能与主治"])
        assert out[0] is cands[1], "命中目标章节的候选没有被提到最前"

    def test_stable_order_when_nothing_matches(self):
        cands = [_chunk("【含量测定】…"), _chunk("【性状】…")]
        assert Retriever._promote_section_hits(cands, ["功能与主治"]) == cands

    def test_stable_within_same_hit_count(self):
        a, b = _chunk("【功能与主治】甲"), _chunk("【功能与主治】乙")
        other = _chunk("【性状】…")
        out = Retriever._promote_section_hits([other, a, b], ["功能与主治"])
        assert out == [a, b, other], "命中数相同的候选应保持原相对顺序"

    def test_empty_expected_sections_noop(self):
        cands = [_chunk("【性状】…")]
        assert Retriever._promote_section_hits(cands, []) == cands


# ============================================================
# 4. 药名精确度作为次级排序键
# ============================================================
# 背景：查询药名到候选的扩展是**子串**方向，所以「问黄芪」时候选池里会有
# 黄芪-饮片 / 黄芪颗粒 / 炙黄芪 等一大堆。章节命中数相同时若只按 RRF 排，
# 含黄芪的成药可能压过黄芪本身。
# 触发场景：修正 `炙黄芷→炙黄芪` 后，炙黄芪 成为合法条目，把 黄芪-饮片 从
# 第 1 位挤到第 3~5 位，8 道黄芪相关题的 strict@1 一起下降。

def _chunk2(content, drug_name, section="完整条目"):
    return {"content": content,
            "metadata": {"drug_name": drug_name, "section": section, "chunk_type": "whole_entry"}}


class TestNameExactnessTieBreak:
    def test_exactness_levels(self):
        sec = ["功能与主治"]
        assert Retriever._name_exactness(_chunk2("", "黄芪"), ["黄芪"]) == 2
        assert Retriever._name_exactness(_chunk2("", "黄芪-饮片"), ["黄芪"]) == 2
        assert Retriever._name_exactness(_chunk2("", "黄芪颗粒"), ["黄芪"]) == 1
        assert Retriever._name_exactness(_chunk2("", "炙黄芪"), ["黄芪"]) == 1
        assert Retriever._name_exactness(_chunk2("", "甘草"), ["黄芪"]) == 0
        assert Retriever._name_exactness(_chunk2("", "黄芪"), []) == 0

    def test_exact_drug_beats_containing_formulation(self):
        """章节命中数相同时，该药本身必须排在含该药名的成药之前"""
        sec = ["功能与主治"]
        formula = _chunk2("【功能与主治】补气。", "黄芪健胃膏")
        base = _chunk2("【功能与主治】补气固表。", "黄芪-饮片")
        out = Retriever._promote_section_hits([formula, base], sec, ["黄芪"])
        assert out[0] is base, "含黄芪的成药压过了黄芪本身"

    def test_section_hits_still_dominate(self):
        """章节命中数是主键：命中章节的成药仍应排在未命中的正主之前"""
        sec = ["含量测定"]
        base_no_section = _chunk2("【性状】…", "黄芪-饮片")
        formula_with_section = _chunk2("【含量测定】照高效液相色谱法…", "黄芪颗粒")
        out = Retriever._promote_section_hits([base_no_section, formula_with_section], sec, ["黄芪"])
        assert out[0] is formula_with_section

    def test_no_queried_drugs_keeps_previous_behaviour(self):
        """不传查询药名时行为与改造前一致（纯按章节命中数，稳定排序）"""
        a, b = _chunk2("【功能与主治】甲", "甲药"), _chunk2("【功能与主治】乙", "乙药")
        other = _chunk2("【性状】…", "丙药")
        assert Retriever._promote_section_hits([other, a, b], ["功能与主治"]) == [a, b, other]


# ============================================================
# 4. 性味条件：查询侧解析 + 语料侧解析 + 候选提升
# ============================================================

class TestSexingCondition:
    def test_query_forms(self):
        cases = {
            "性味甘温的药材有哪些": ({"甘"}, {"温"}),
            "味甘性温的中药": ({"甘"}, {"温"}),
            "苦、寒的药材": ({"苦"}, {"寒"}),
            "性味：甘、微温的药材": ({"甘"}, {"温"}),
            "人参味甘性微温吗": ({"甘"}, {"温"}),
        }
        for q, expected in cases.items():
            assert Retriever._extract_sexing_condition(q) == expected, q

    def test_no_false_positive(self):
        for q in ("五味子的功效", "甘草的功效", "苦参的性味", "大补元气的中药",
                  "清热解毒的药材", "活血化瘀的中药", "温性的药材"):
            assert Retriever._extract_sexing_condition(q) is None, q

    def test_parse_corpus_text(self):
        assert Retriever._parse_sexing_text(
            "【性味与归经】辛、苦，凉。归肺、肝经。"
        ) == ({"辛", "苦"}, "凉")
        # chunk 正文带「药名 · 章节」首行，不能把药名行当成味描述
        assert Retriever._parse_sexing_text(
            "人参 · 性味与归经\n【性味与归经】甘、微苦，微温。归脾、肺、心、肾经。"
        ) == ({"甘", "苦"}, "温")
        assert Retriever._parse_sexing_text(
            "【性味与归经】辛，温;有小毒。归肝经。"
        ) == ({"辛"}, "温")
        assert Retriever._parse_sexing_text("无性味信息") == (set(), "")

    def test_match_requires_taste_and_nature(self):
        chunk = {"content": "【性味与归经】苦，寒。归心经。",
                 "metadata": {"section": "性味与归经"}}
        assert Retriever._sexing_match(chunk, {"苦"}, {"寒"})
        assert not Retriever._sexing_match(chunk, {"甘"}, {"寒"})
        assert not Retriever._sexing_match(chunk, {"苦"}, {"温"})
        # 非性味章节且正文无标记 → 不参与匹配
        assert not Retriever._sexing_match(
            {"content": "苦，寒。", "metadata": {"section": "完整条目"}},
            {"苦"}, {"寒"},
        )

    def test_promote_matching_chunk_to_front(self):
        cold = {"content": "【性味与归经】苦，寒。", "metadata": {"section": "性味与归经"}}
        warm = {"content": "【性味与归经】甘，温。", "metadata": {"section": "性味与归经"}}
        assert Retriever._promote_sexing_hits([cold, warm], {"甘"}, {"温"})[0] is warm


# ============================================================
# 5. 成分/检测类查询：判定与章节提升
# ============================================================

class TestCompositionQuery:
    def test_detects_composition_queries(self):
        for q in ("哪些药材含有挥发油成分", "双黄连口服液的含量测定方法",
                  "哪些中药含黄连素", "薄层色谱鉴别方法"):
            assert Retriever._is_composition_query(q), q

    def test_efficacy_query_is_not_composition(self):
        for q in ("哪些药材含有什么功效", "哪些中药有补气功效", "人参的功效与作用"):
            assert not Retriever._is_composition_query(q), q

    def test_extracts_composition_terms_by_suffix(self):
        assert Retriever._extract_composition_terms("哪些中药含麻黄碱") == ["麻黄碱"]
        assert Retriever._extract_composition_terms("哪些药材含有挥发油成分") == ["挥发油"]
        assert Retriever._extract_composition_terms("哪些药材含黄芪甲苷") == ["黄芪甲苷"]
        assert Retriever._extract_composition_terms("胃酸过多的药材") == []

    def test_embedded_drug_name_is_not_a_drug_filter(self):
        qp = QueryParser(drug_names=["黄连", "复方黄连素片", "人参"])
        parsed = qp.parse("哪些中药含黄连素")
        assert parsed["drug_names"] == []
        assert parsed["is_horizontal"] is True
        # 正常药名查询不受影响
        assert qp.parse("人参的功效")["drug_names"] == ["人参"]

    def test_ocr_glob_pattern_for_berberine(self):
        assert Retriever._composition_glob_patterns(["小檗碱"]) == ["*小?碱*"]
        assert Retriever._composition_glob_patterns(["麻黄碱"]) == []

    def test_section_base_strips_etl_suffix(self):
        assert Retriever._section_base("含量测定-成分1") == "含量测定"
        assert Retriever._section_base("检查-项2") == "检查"
        assert Retriever._section_base("鉴别-段2") == "鉴别"
        assert Retriever._section_base("性味与归经") == "性味与归经"

    def test_promote_by_metadata_section(self):
        whole = {"content": "…", "metadata": {"section": "完整条目"}}
        assay = {"content": "…", "metadata": {"section": "含量测定-成分1"}}
        out = Retriever._promote_composition_sections(
            [whole, assay], Retriever._COMPOSITION_SECTIONS
        )
        assert out[0] is assay
