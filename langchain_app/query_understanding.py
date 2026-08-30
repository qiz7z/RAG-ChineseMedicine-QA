# -*- coding: utf-8 -*-
"""
查询理解（标准版）
==================
轻量的规则式查询分析：识别药品名（含变体扩展）、横向条件查询及分类过滤。
独立实现，不依赖 src/retrieval/query_parser.py；与手撕版行为对齐但修复了
分类简称与实际值不匹配的缺陷（"药材" → "药材和饮片"）。
"""
import re
from dataclasses import dataclass, field
from typing import Optional, List

# 横向条件查询触发词（"哪些药材有补气功效"类）
_HORIZONTAL_KEYWORDS = [
    "哪些药材", "哪些中药", "哪些药物", "哪些药品", "有哪些", "什么中药",
    "什么药材", "列举", "罗列",
]

# 分类简称 → SQLite/元数据实际分类值
CATEGORY_MAP = {
    "药材": "药材和饮片",
    "成方制剂": "成方制剂和单味制剂",
    "药材和饮片": "药材和饮片",
    "成方制剂和单味制剂": "成方制剂和单味制剂",
    "植物油脂和提取物": "植物油脂和提取物",
}


@dataclass
class QueryInfo:
    """查询理解结果"""
    raw_query: str
    drug_names: List[str] = field(default_factory=list)
    expanded_drugs: Optional[set] = None   # 扩展后的药品名集合（用于 $in 式过滤）
    is_horizontal: bool = False
    category: Optional[str] = None         # 已映射到实际分类值


class QueryAnalyzer:
    """规则式查询分析器"""

    def __init__(self, drug_names: List[str]):
        self.drug_names = drug_names

    def analyze(self, query: str) -> QueryInfo:
        info = QueryInfo(raw_query=query)
        info.drug_names = self._detect_drugs(query)

        if info.drug_names:
            info.expanded_drugs = self.expand(info.drug_names)

        info.is_horizontal = (not info.drug_names) and self._is_horizontal(query)
        if info.is_horizontal:
            info.category = self._detect_category(query)
        return info

    def expand(self, drug_names: List[str]) -> set:
        """变体扩展：如"人参" → {"人参", "人参-饮片", "人参叶", ...}"""
        expanded = set()
        for fd in drug_names:
            for dn in self.drug_names:
                if fd in dn or dn in fd:
                    expanded.add(dn)
        return expanded

    # ----------------------------------------------------------

    def _detect_drugs(self, query: str) -> List[str]:
        """按长度降序匹配（避免'人参叶'被'人参'抢先命中后漏配）"""
        found = [name for name in self.drug_names if name in query]
        found.sort(key=len, reverse=True)
        return found

    def _is_horizontal(self, query: str) -> bool:
        for kw in _HORIZONTAL_KEYWORDS:
            if kw in query:
                return True
        if re.search(r"有.{0,6}(功效|作用|成分|功能|主治)", query):
            return True
        if re.search(r"(能治|用于|治疗|主治).{2,}", query) and re.search(
            r"(的药|的药材|的中药|的方|的成药|有哪些|有什么)", query
        ):
            return True
        return False

    def _detect_category(self, query: str) -> Optional[str]:
        if re.search(r"成方|中成药|成药|方剂|丸|散|膏|口服液|颗粒|胶囊", query):
            return CATEGORY_MAP["成方制剂"]
        if re.search(r"药材|中药|饮片|生药", query):
            return CATEGORY_MAP["药材"]
        return None
