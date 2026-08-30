# -*- coding: utf-8 -*-
"""
回答后处理（标准版）
====================
引用标注 + 用药安全提醒。独立实现（逻辑与主项目 PostProcessor 同源）。
作为 LCEL 链尾的 RunnableLambda 使用：输入 {"answer", "docs"}，输出 {"answer", "citations"}。
"""
import re
from typing import List


def postprocess(answer: str, docs: List) -> dict:
    """为回答追加引用来源列表与安全提醒。

    Args:
        answer: LLM 生成的回答
        docs:   检索到的 Document 列表（metadata 含 drug_name/section）

    Returns:
        {"answer": 最终回答, "citations": ["药典2020一部-药品-章节", ...]}
    """
    citations = []
    seen = set()
    for d in docs:
        m = d.metadata
        cite = f"药典2020一部-{m.get('drug_name', '')}-{m.get('section', '')}"
        if m.get("drug_name") and cite not in seen:
            seen.add(cite)
            citations.append(cite)

    final = answer.strip()

    # 回答中未自带"来源/出处"段落时，追加引用列表
    if citations and not re.search(r"来源|出处|参考", final):
        ref_lines = "\n".join(f"- [{i+1}] {c}" for i, c in enumerate(citations[:5]))
        final = f"{final}\n\n**参考来源：**\n{ref_lines}"

    # 安全提醒
    if "遵医嘱" not in final:
        final = f"{final}\n\n具体用药请遵医嘱。"

    return {"answer": final, "citations": citations}


def format_docs(docs: List) -> str:
    """检索文档 → 【参考资料】文本块（喂给 RAG Prompt）"""
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        blocks.append(
            f"【参考资料 {i}】药品：{m.get('drug_name', '')} | 章节：{m.get('section', '')}\n"
            f"{d.page_content}"
        )
    return "\n\n".join(blocks)


def docs_to_sources(docs: List) -> List[dict]:
    """检索文档 → API SourceItem 形状"""
    return [
        {
            "chunk_id": d.metadata.get("chunk_id", ""),
            "drug_name": d.metadata.get("drug_name", ""),
            "section": d.metadata.get("section", ""),
            "category": d.metadata.get("category", ""),
            "content": d.page_content[:500],
            "score": d.metadata.get("score", 0),
            "rerank_score": d.metadata.get("rerank_score"),
            "sources": d.metadata.get("sources", []),
        }
        for d in docs
    ]
