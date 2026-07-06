# -*- coding: utf-8 -*-
"""
Prompt 模板系统
================
为药典智能问答系统设计专用的 Prompt 模板，确保：
1. 严格基于检索到的药典原文回答
2. 不编造、不添加参考资料以外的信息
3. 引用规范、术语专业
4. 用药安全问题提醒

设计原则：
  - System Prompt 定义角色和规则（不变）
  - Context 部分注入检索到的药典原文（动态）
  - User 部分注入用户问题（动态）
  - 多轮对话时注入历史上下文
"""
from typing import List, Dict, Optional
from config import MEDICAL_DISCLAIMER


# ============================================================
# 系统提示词
# ============================================================

SYSTEM_PROMPT = f"""你是一位精通《中国药典》的资深中药学专家。请严格基于下方【药典参考资料】回答用户的问题。

## 回答规则
1. 只能基于【药典参考资料】中的内容回答，不得添加任何参考资料以外的信息。
2. 如果参考资料中没有涉及用户问题的内容，请明确回答："药典中未收录相关信息。"，不得编造。
3. 回答中引用药典原文时，请用引号标注，并在句末注明出处（如：来源：药典2020一部-药品名-章节）。
4. 保持专业、准确、简洁，使用药典的规范术语。
5. 如果用户问题涉及用药建议，请在回答末尾添加提醒："{MEDICAL_DISCLAIMER}"
6. 如果多个检索结果涉及同一问题，请综合归纳后回答，避免重复。
7. 回答使用 Markdown 格式，表格数据以表格呈现，便于阅读。"""

# ============================================================
# Prompt 构建
# ============================================================

def build_context_section(retrieval_results: List[Dict]) -> str:
    """
    将检索结果构建为 Prompt 中的【药典参考资料】部分。

    Args:
        retrieval_results: 检索结果列表（SearchResult.to_dict() 格式）

    Returns:
        格式化的参考资料文本
    """
    if not retrieval_results:
        return "（未检索到相关药典内容）"

    parts = []
    for i, r in enumerate(retrieval_results, 1):
        drug_name = r.get("drug_name", "未知药品")
        section = r.get("section", "未知章节")
        content = r.get("content", "").strip()

        # 构建出处标注
        source_parts = [f"药品：{drug_name}"]
        if section:
            source_parts.append(f"章节：{section}")
        source = " | ".join(source_parts)

        parts.append(f"【参考资料 {i}】{source}\n{content}")

    return "\n\n".join(parts)


def build_prompt(
    user_query: str,
    retrieval_results: List[Dict],
    chat_history: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    构建完整的 LLM 消息列表（OpenAI 格式）。

    Args:
        user_query: 用户当前查询
        retrieval_results: 检索结果列表
        chat_history: 多轮对话历史（可选）

    Returns:
        OpenAI 格式的 messages 列表
    """
    # 1. 系统消息
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 2. 对话历史（如有）
    if chat_history:
        for turn in chat_history:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", ""),
            })

    # 3. 构建用户消息（包含检索上下文 + 用户问题）
    context_text = build_context_section(retrieval_results)

    user_content = f"""## 药典参考资料
---
{context_text}
---

## 用户问题
{user_query}"""

    messages.append({"role": "user", "content": user_content})

    return messages


def build_no_result_prompt(user_query: str) -> List[Dict]:
    """
    当检索结果为空时的 Prompt（直接返回未收录提示）。

    Args:
        user_query: 用户查询

    Returns:
        OpenAI 格式的 messages 列表
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""## 药典参考资料
---
（未检索到与用户问题相关的药典内容）
---

## 用户问题
{user_query}"""},
    ]
    return messages


# ============================================================
# 指代消解提示词（用于多轮对话中的"它"、"这个药"等指代）
# ============================================================

COREFERENCE_PROMPT_TEMPLATE = """请分析以下对话上下文，将用户最新问题中的指代词（如"它"、"这个药"、"该药材"、"它"等）替换为具体的药品名称。

## 对话历史
{chat_history}

## 用户最新问题
{current_query}

## 任务
1. 识别用户最新问题中的指代词
2. 根据对话历史推断指代对象
3. 输出消解后的完整问题

注意：
- 只输出消解后的问题文本，不要添加任何解释
- 如果没有指代词需要消解，直接输出原问题
- 保持问题原意，不要改变其他内容

## 消解后的问题："""


def build_coreference_prompt(
    current_query: str,
    chat_history: List[Dict],
) -> List[Dict]:
    """
    构建指代消解的 Prompt。

    Args:
        current_query: 用户当前查询
        chat_history: 对话历史

    Returns:
        OpenAI 格式的 messages 列表
    """
    # 将对话历史格式化为文本
    history_text = ""
    for turn in chat_history:
        role = "用户" if turn["role"] == "user" else "助手"
        history_text += f"{role}: {turn['content']}\n"

    prompt = COREFERENCE_PROMPT_TEMPLATE.format(
        chat_history=history_text.strip(),
        current_query=current_query,
    )

    return [
        {"role": "system", "content": "你是一个文本处理助手，只输出处理结果，不添加任何额外解释。"},
        {"role": "user", "content": prompt},
    ]
