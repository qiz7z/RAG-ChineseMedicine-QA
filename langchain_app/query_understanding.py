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

# 用户说法 → 药典原生章节名（用于章节感知召回；与手撕版 SECTION_MARKERS 同源）
_SECTION_SYNONYMS = {
    "性味与归经": "性味与归经", "性味归经": "性味与归经",
    "性味": "性味与归经", "归经": "性味与归经",
    "功能与主治": "功能与主治", "功能主治": "功能与主治",
    "功能": "功能与主治", "主治": "功能与主治",
    "功效": "功能与主治", "作用": "功能与主治", "疗效": "功能与主治",
    "用法与用量": "用法与用量", "用法用量": "用法与用量",
    "用法": "用法与用量", "用量": "用法与用量",
    "性状": "性状", "外观": "性状", "形状": "性状",
    "鉴别": "鉴别", "检查": "检查",
    "含量测定": "含量测定", "含量": "含量测定", "测定": "含量测定",
    "炮制": "炮制", "制法": "制法",
    "贮藏": "贮藏", "储藏": "贮藏", "保存": "贮藏",
    "处方": "处方", "规格": "规格", "来源": "来源",
    "浸出物": "浸出物", "特征图谱": "特征图谱", "指纹图谱": "指纹图谱",
}


@dataclass
class QueryInfo:
    """查询理解结果"""
    raw_query: str
    drug_names: List[str] = field(default_factory=list)
    expanded_drugs: Optional[set] = None   # 扩展后的药品名集合（用于 $in 式过滤）
    is_horizontal: bool = False
    category: Optional[str] = None         # 已映射到实际分类值
    sections: List[str] = field(default_factory=list)   # 识别到的目标章节（标准化后）


class QueryAnalyzer:
    """规则式查询分析器"""

    def __init__(self, drug_names: List[str]):
        self.drug_names = drug_names

    def analyze(self, query: str) -> QueryInfo:
        info = QueryInfo(raw_query=query)
        info.drug_names = self._detect_drugs(query)
        info.sections = self._detect_sections(query)

        if info.drug_names:
            info.expanded_drugs = self.expand(info.drug_names)

        info.is_horizontal = (not info.drug_names) and self._is_horizontal(query)
        if info.is_horizontal:
            info.category = self._detect_category(query)
        return info

    def expand(self, drug_names: List[str]) -> set:
        """变体扩展：如"人参" → {"人参", "人参-饮片", "人参叶", ...}

        只做**单向**扩展（查询药名是候选药名的子串）。反向匹配
        （`dn in fd`，如查"双黄连口服液"命中"黄连"）会引入更短的无关药名，
        与 `_detect_drugs` 的最长匹配去重相冲突，故不采用。
        """
        expanded = set()
        for fd in drug_names:
            for dn in self.drug_names:
                if fd in dn:
                    expanded.add(dn)
        return expanded

    # ----------------------------------------------------------

    def _detect_drugs(self, query: str) -> List[str]:
        """按长度降序匹配，并剔除已被更长药名包含的子串。

        例："双黄连口服液的含量测定" → ["双黄连口服液"]，不保留"黄连"——
        否则药品过滤会同时放行黄连条目，top-1 落到黄连上。
        （与 src/retrieval/query_parser.py::_extract_drug_names 的行为对齐）
        """
        found: List[str] = []
        for name in sorted(self.drug_names, key=len, reverse=True):
            if name not in query:
                continue
            if any(name in longer for longer in found):
                continue          # 已是更长命中药名的子串 → 跳过
            found.append(name)
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

    def _detect_sections(self, query: str) -> List[str]:
        """识别查询指向的药典章节（标准化为原生章节名）。

        用 set 收集，避免"性味归经"同时命中"性味"和"归经"产生重复。
        """
        return list({std for user_word, std in _SECTION_SYNONYMS.items()
                     if user_word in query})
