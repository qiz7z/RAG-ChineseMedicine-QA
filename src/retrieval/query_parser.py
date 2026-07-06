# -*- coding: utf-8 -*-
"""
查询解析器
==========
对用户输入的自然语言查询进行解析，提取结构化信息，辅助检索：

  1. 药品名识别  — 从查询中匹配药典中存在的药品名
  2. 章节识别    — 匹配药典章节标记（如"性味归经""含量测定"）
  3. 查询意图分类 — 判断用户想查什么类型的信息

解析结果用于：
  - 为向量检索添加元数据过滤条件（where={"drug_name": "人参"}）
  - 为 BM25 检索提供更精准的关键词
  - 为上下文组装提供结构化信息
"""
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SECTION_MARKERS, SHORT_SECTIONS_TO_GROUP


class QueryParser:
    """
    查询解析器：从用户查询中提取药品名、章节名、查询意图。

    设计思路：
      - 药品名识别：从 SQLite 加载所有药品名，在查询中做子串匹配
      - 章节识别：基于预定义的 SECTION_MARKERS 做关键词匹配
      - 意图分类：基于规则匹配，判断查询属于哪个信息类别
    """

    # 查询意图类型
    INTENT_TYPES = {
        "属性查询": ["性味", "归经", "性味归经", "性味与归经"],
        "功效查询": ["功能", "主治", "功能与主治", "功效", "作用", "疗效"],
        "用法查询": ["用法", "用量", "用法与用量", "怎么吃", "怎么用", "服用"],
        "鉴别查询": ["鉴别", "怎么鉴别", "如何鉴别", "真伪"],
        "检测查询": ["检查", "含量测定", "测定", "高效液相", "色谱", "薄层"],
        "炮制查询": ["炮制", "制法", "加工"],
        "性状查询": ["性状", "外观", "形状", "颜色", "质地"],
        "储藏查询": ["贮藏", "储藏", "保存", "存放"],
        "制剂查询": ["制剂", "处方", "规格", "成方"],
        "成分查询": ["成分", "含有", "含", "挥发油", "皂苷", "黄酮"],
    }

    def __init__(self, drug_names: List[str] = None):
        """
        Args:
            drug_names: 药典中所有药品名列表（从 SQLite 加载）
                        如果不提供，则不做药品名识别
        """
        # 药品名按长度降序排列，优先匹配长名（如"双黄连口服液"优先于"黄连"）
        self.drug_names = sorted(drug_names, key=len, reverse=True) if drug_names else []

        # 章节同义词映射（用户可能用的词 → 药典标准章节名）
        self._section_synonyms = self._build_section_synonyms()

        # 编译意图匹配模式
        self._intent_patterns = {}
        for intent, keywords in self.INTENT_TYPES.items():
            self._intent_patterns[intent] = re.compile(
                '|'.join(re.escape(kw) for kw in keywords)
            )

    def _build_section_synonyms(self) -> Dict[str, str]:
        """构建章节同义词映射"""
        synonyms = {}
        for marker in SECTION_MARKERS:
            synonyms[marker] = marker

        # 用户常用说法 → 标准章节名
        user_to_standard = {
            "性味": "性味与归经",
            "归经": "性味与归经",
            "性味归经": "性味与归经",
            "功效": "功能与主治",
            "主治": "功能与主治",
            "功能主治": "功能与主治",
            "作用": "功能与主治",
            "疗效": "功能与主治",
            "用法": "用法与用量",
            "用量": "用法与用量",
            "用法用量": "用法与用量",
            "服用方法": "用法与用量",
            "测定": "含量测定",
            "含量": "含量测定",
            "色谱": "含量测定",
            "保存": "贮藏",
            "储藏": "贮藏",
            "存放": "贮藏",
            "外观": "性状",
            "形状": "性状",
        }
        synonyms.update(user_to_standard)
        return synonyms

    def parse(self, query: str) -> Dict[str, Any]:
        """
        解析用户查询。

        Args:
            query: 用户查询文本

        Returns:
            解析结果字典：
            {
                "raw_query": str,           # 原始查询
                "drug_names": List[str],    # 识别到的药品名
                "sections": List[str],      # 识别到的章节名（标准化后）
                "intent": str,              # 查询意图类型
                "keywords": List[str],      # 提取的关键词
                "filter": Dict or None,     # Chroma where 过滤条件
            }
        """
        result = {
            "raw_query": query,
            "drug_names": [],
            "sections": [],
            "intent": "通用查询",
            "keywords": [],
            "filter": None,
        }

        # 1. 药品名识别
        found_drugs = self._extract_drug_names(query)
        result["drug_names"] = found_drugs

        # 2. 章节识别
        found_sections = self._extract_sections(query)
        result["sections"] = found_sections

        # 3. 意图分类
        result["intent"] = self._classify_intent(query)

        # 4. 关键词提取（简单版：去停用词后的剩余中文词组）
        result["keywords"] = self._extract_keywords(query, found_drugs, found_sections)

        # 5. 生成 Chroma 过滤条件
        result["filter"] = self._build_filter(found_drugs)

        return result

    def _extract_drug_names(self, query: str) -> List[str]:
        """从查询中识别药品名"""
        if not self.drug_names:
            return []

        found = []
        query_lower = query.lower()

        for drug_name in self.drug_names:
            if drug_name in query_lower:
                # 去重：如果已匹配到的药品名是当前药品名的子串，则跳过
                # 例如已匹配"双黄连口服液"，则跳过"黄连"
                is_substring = False
                for existing in found:
                    if drug_name in existing:
                        is_substring = True
                        break
                if not is_substring:
                    found.append(drug_name)

        return found

    def _extract_sections(self, query: str) -> List[str]:
        """从查询中识别章节名"""
        found = set()

        for user_word, standard_section in self._section_synonyms.items():
            if user_word in query:
                found.add(standard_section)

        return list(found)

    def _classify_intent(self, query: str) -> str:
        """基于规则识别查询意图"""
        for intent, pattern in self._intent_patterns.items():
            if pattern.search(query):
                return intent

        # 默认意图
        return "通用查询"

    def _extract_keywords(self, query: str, drugs: List[str], sections: List[str]) -> List[str]:
        """
        提取查询中的关键词（用于 BM25 检索增强）。

        策略：去掉药品名和章节关键词后的剩余有意义词汇。
        """
        keywords = []

        # 去掉已识别的药品名
        cleaned = query
        for drug in drugs:
            cleaned = cleaned.replace(drug, ' ')

        # 使用 jieba 分词提取剩余关键词
        try:
            import jieba
            jieba.setLogLevel(20)
            tokens = jieba.cut(cleaned, cut_all=False)

            stop_words = {'的', '是', '什么', '怎么', '如何', '有哪些', '有', '哪些',
                          '请问', '？', '?', '了', '吗', '呢', '啊', '我', '想', '要',
                          '请', '帮', '查询', '查一下', '告诉', '知道'}

            for tok in tokens:
                tok = tok.strip()
                if len(tok) >= 2 and tok not in stop_words:
                    if re.search(r'[\u4e00-\u9fffA-Za-z0-9]', tok):
                        keywords.append(tok)
        except ImportError:
            pass

        return keywords

    def _build_filter(self, drug_names: List[str]) -> Optional[Dict]:
        """
        根据识别到的药品名生成 Chroma where 过滤条件。

        如果只识别到一个药品名，直接精确过滤。
        如果识别到多个，使用 $in 操作符。
        如果没有识别到，返回 None（不过滤，全库检索）。
        """
        if not drug_names:
            return None

        if len(drug_names) == 1:
            return {"drug_name": drug_names[0]}
        else:
            return {"drug_name": {"$in": drug_names}}


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("查询解析器测试")
    print("=" * 60)

    # 模拟药品名列表
    mock_drugs = ["人参", "人参叶", "黄芪", "黄连", "双黄连口服液",
                  "六味地黄丸", "牛黄解毒片", "西洋参", "当归", "甘草"]

    parser = QueryParser(drug_names=mock_drugs)

    test_queries = [
        "人参的性味归经是什么？",
        "双黄连口服液的含量测定方法",
        "哪些药材含有挥发油成分？",
        "高效液相色谱法的操作步骤",
        "黄芪和黄连的功效有什么区别？",
        "六味地黄丸怎么服用？",
    ]

    for query in test_queries:
        print(f"\n查询: '{query}'")
        result = parser.parse(query)
        print(f"  药品名: {result['drug_names']}")
        print(f"  章节:   {result['sections']}")
        print(f"  意图:   {result['intent']}")
        print(f"  关键词: {result['keywords']}")
        print(f"  过滤:   {result['filter']}")
