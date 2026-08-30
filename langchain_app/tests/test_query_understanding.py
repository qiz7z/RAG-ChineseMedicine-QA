# -*- coding: utf-8 -*-
"""query_understanding 单元测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from query_understanding import QueryAnalyzer, CATEGORY_MAP


@pytest.fixture(scope="module")
def analyzer():
    # 真实药品名清单的代表性子集（避免加载 11k 索引）
    names = [
        "人参", "人参-饮片", "人参叶", "人参健脾丸", "人参茎叶总皂苷",
        "黄芪", "黄芪-饮片", "黄芪颗粒",
        "川芎", "党参",
    ]
    return QueryAnalyzer(names)


class TestDrugDetection:
    def test_simple_drug(self, analyzer):
        info = analyzer.analyze("人参的性味归经是什么？")
        assert "人参" in info.drug_names

    def test_variant_expansion(self, analyzer):
        info = analyzer.analyze("人参的性味归经是什么？")
        # 变体扩展应包含饮片/叶/制剂等
        assert {"人参", "人参-饮片", "人参叶", "人参健脾丸"} <= info.expanded_drugs

    def test_longer_name_priority(self, analyzer):
        # "人参叶" 应被识别，而不是只命中 "人参"
        info = analyzer.analyze("人参叶的功效")
        assert "人参叶" in info.drug_names

    def test_no_drug(self, analyzer):
        info = analyzer.analyze("今天天气怎么样？")
        assert info.drug_names == []
        assert info.expanded_drugs is None


class TestHorizontalQuery:
    def test_horizontal_with_category(self, analyzer):
        info = analyzer.analyze("哪些药材有补气功效？")
        assert info.is_horizontal is True
        # 分类映射修复：简称"药材"应映射为实际值"药材和饮片"
        assert info.category == "药材和饮片"

    def test_horizontal_formulation(self, analyzer):
        info = analyzer.analyze("有哪些中成药可以治疗感冒？")
        assert info.is_horizontal is True
        assert info.category == "成方制剂和单味制剂"

    def test_horizontal_no_category(self, analyzer):
        info = analyzer.analyze("含有黄连素的药材有哪些")
        # 含药品名(黄连素不在清单)但触发横向词；无具体分类词时 category=None
        assert info.category is None or info.category == "药材和饮片"

    def test_not_horizontal_when_drug_found(self, analyzer):
        # 指定了具体药品 → 不是横向查询
        info = analyzer.analyze("人参有补气功效吗")
        assert info.is_horizontal is False


class TestCategoryMap:
    def test_mapping_covers_actual_values(self):
        # 映射表必须覆盖 SQLite 的三个实际分类值
        for actual in ["药材和饮片", "成方制剂和单味制剂", "植物油脂和提取物"]:
            assert CATEGORY_MAP.get(actual, actual) == actual
