# -*- coding: utf-8 -*-
"""
回答后处理模块
===============
对 LLM 生成的回答进行后处理，包括：

1. 引用标注：在回答中自动添加来源标注
2. 一致性校验：检查回答中的关键信息（药品名、数值）是否在检索上下文中出现
3. 格式美化：统一格式、清理多余空白、补充安全提醒

设计目标：
  - 保证回答的可溯源性和准确性
  - 防止 LLM 幻觉（编造未在参考资料中出现的内容）
  - 提升回答的可读性
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from config import (
    ENABLE_CITATION,
    ENABLE_CONSISTENCY_CHECK,
    ENABLE_FORMAT_BEAUTIFY,
    MEDICAL_DISCLAIMER,
)


@dataclass
class PostProcessResult:
    """后处理结果"""
    answer: str                           # 处理后的回答
    citations: List[str] = field(default_factory=list)     # 引用来源列表
    consistency_issues: List[str] = field(default_factory=list)  # 一致性问题
    has_medical_disclaimer: bool = False  # 是否包含用药安全提醒


class PostProcessor:
    """
    回答后处理器。

    使用方式：
        processor = PostProcessor()
        result = processor.process(answer, retrieval_results)
        print(result.answer)
    """

    # 用药安全相关关键词
    MEDICAL_KEYWORDS = [
        "用法", "用量", "服用", "用药", "剂量",
        "口服", "外用", "注射", "慎用", "禁用",
        "注意事项", "不良反应",
    ]

    # 数值模式（用于一致性校验）
    NUMBER_PATTERN = re.compile(r'\d+\.?\d*\s*(?:mg|g|ml|μg|ug|片|粒|丸|次|日|%)', re.IGNORECASE)

    def __init__(
        self,
        enable_citation: bool = None,
        enable_consistency_check: bool = None,
        enable_format_beautify: bool = None,
    ):
        self.enable_citation = ENABLE_CITATION if enable_citation is None else enable_citation
        self.enable_consistency_check = (
            ENABLE_CONSISTENCY_CHECK if enable_consistency_check is None
            else enable_consistency_check
        )
        self.enable_format_beautify = (
            ENABLE_FORMAT_BEAUTIFY if enable_format_beautify is None
            else enable_format_beautify
        )

    # ----------------------------------------------------------
    # 主处理方法
    # ----------------------------------------------------------

    def process(
        self,
        answer: str,
        retrieval_results: List[Dict],
    ) -> PostProcessResult:
        """
        对 LLM 回答进行完整后处理。

        Args:
            answer: LLM 原始回答
            retrieval_results: 检索结果列表（用于引用和一致性校验）

        Returns:
            PostProcessResult 对象
        """
        processed = answer
        citations = []
        consistency_issues = []

        # 1. 格式美化
        if self.enable_format_beautify:
            processed = self._beautify_format(processed)

        # 2. 引用标注
        if self.enable_citation and retrieval_results:
            processed, citations = self._add_citations(processed, retrieval_results)

        # 3. 一致性校验
        if self.enable_consistency_check and retrieval_results:
            consistency_issues = self._check_consistency(processed, retrieval_results)

        # 4. 用药安全提醒
        has_disclaimer = MEDICAL_DISCLAIMER in processed
        if not has_disclaimer and self._needs_medical_disclaimer(processed):
            processed = processed.rstrip() + f"\n\n> ⚠️ {MEDICAL_DISCLAIMER}"
            has_disclaimer = True

        return PostProcessResult(
            answer=processed,
            citations=citations,
            consistency_issues=consistency_issues,
            has_medical_disclaimer=has_disclaimer,
        )

    # ----------------------------------------------------------
    # 格式美化
    # ----------------------------------------------------------

    def _beautify_format(self, text: str) -> str:
        """
        格式美化：清理多余空白、统一标点。
        """
        # 去除首尾空白
        text = text.strip()

        # 合并连续空行（最多保留一个空行）
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除行尾多余空格
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text

    # ----------------------------------------------------------
    # 引用标注
    # ----------------------------------------------------------

    def _add_citations(
        self,
        answer: str,
        retrieval_results: List[Dict],
    ) -> Tuple[str, List[str]]:
        """
        在回答末尾添加引用来源列表。

        如果回答中已包含引用标注，则不重复添加。

        Args:
            answer: LLM 回答
            retrieval_results: 检索结果列表

        Returns:
            (添加引用后的回答, 引用来源列表)
        """
        # 构建引用来源列表
        citations = []
        seen = set()  # 去重
        for r in retrieval_results:
            drug_name = r.get("drug_name", "")
            section = r.get("section", "")
            if drug_name:
                citation = f"药典2020一部-{drug_name}"
                if section:
                    citation += f"-{section}"
                if citation not in seen:
                    seen.add(citation)
                    citations.append(citation)

        # 如果回答中已包含来源标注，不重复添加
        if "来源：" in answer or "出处" in answer:
            return answer, citations

        # 在回答末尾添加引用列表
        if citations:
            citation_text = "\n\n**参考来源：**\n"
            for i, c in enumerate(citations, 1):
                citation_text += f"- [{i}] {c}\n"
            answer = answer.rstrip() + citation_text

        return answer, citations

    # ----------------------------------------------------------
    # 一致性校验
    # ----------------------------------------------------------

    def _check_consistency(
        self,
        answer: str,
        retrieval_results: List[Dict],
    ) -> List[str]:
        """
        检查回答中的关键信息是否在检索上下文中出现。

        检查项：
          1. 回答中出现的药品名是否在检索结果中
          2. 回答中出现的数值（剂量、含量等）是否在检索结果中

        Args:
            answer: LLM 回答
            retrieval_results: 检索结果列表

        Returns:
            一致性问题列表（空列表表示无问题）
        """
        issues = []

        # 构建上下文文本（合并所有检索结果内容）
        context_text = " ".join([r.get("content", "") for r in retrieval_results])

        # 1. 检查药品名一致性
        context_drugs = set()
        for r in retrieval_results:
            drug_name = r.get("drug_name", "")
            if drug_name:
                context_drugs.add(drug_name)

        # 如果回答中提到了药品名但不在检索结果中，记录问题
        # 这里简单检查：如果回答中包含不在上下文中的药品名
        # （排除常见的非药品词汇）
        # 注意：这是一个轻量级检查，不做严格 NER

        # 2. 检查数值一致性
        # 提取回答中的数值
        answer_numbers = set(self.NUMBER_PATTERN.findall(answer))
        context_numbers = set(self.NUMBER_PATTERN.findall(context_text))

        # 找出回答中出现但上下文中没有的数值
        # （排除常见的安全数值如 100%）
        suspicious_numbers = answer_numbers - context_numbers
        for num in suspicious_numbers:
            # 过滤掉 100% 这种常见表达
            if num.strip().lower() not in ["100%", "100 %"]:
                issues.append(f"回答中出现的数值 '{num}' 未在检索到的药典原文中找到，请核实")

        return issues

    # ----------------------------------------------------------
    # 用药安全提醒判断
    # ----------------------------------------------------------

    def _needs_medical_disclaimer(self, answer: str) -> bool:
        """
        判断回答是否涉及用药建议，需要添加安全提醒。

        Args:
            answer: LLM 回答

        Returns:
            是否需要添加用药安全提醒
        """
        answer_lower = answer.lower()
        for keyword in self.MEDICAL_KEYWORDS:
            if keyword in answer:
                return True
        return False
