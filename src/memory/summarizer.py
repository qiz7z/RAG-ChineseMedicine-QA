# -*- coding: utf-8 -*-
"""
会话摘要（P1：三槽摘要）
========================

把一段多轮对话压成三个槽位，让「超出最近 N 轮之后」仍能知道上下文：

    focus            当前关注什么（药品 / 章节）
    asked_sections   这轮对话问过哪些药典章节
    open_questions   用户问到但没答上的问题

与 DeerFlow 的六槽（workContext / personalContext / topOfMind /
recentMonths / earlierContext / longTermBackground）相比，砍掉了与药典问答
无关的「用户画像」——这里不需要知道用户是谁，只需要知道他在问什么药。

设计约束：
    1. 抽取失败静默：摘要只是增强，不能影响回答；
    2. 只认 JSON：解析不到就放弃这一轮，宁可没有摘要也不要脏文本；
    3. 摘要控制在 200 字内，避免挤占检索上下文预算。
"""
import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 200
MAX_SECTIONS = 6
MAX_OPEN_QUESTIONS = 3

SUMMARY_PROMPT = """你是对话摘要助手。阅读下面的多轮对话，输出一个 JSON 对象，字段固定为：

{{
  "focus": "当前正在讨论的药品与章节，一句话，如「人参 · 性味与归经」",
  "asked_sections": ["这轮对话问过的药典章节，如 性味与归经、用法与用量"],
  "open_questions": ["用户问到但没得到明确答案的问题；没有就留空数组"]
}}

要求：
1. 只输出 JSON，不要解释、不要代码块标记；
2. 药品名保留药典原名，不要改写；
3. focus 不超过 30 字，整体控制在 200 字以内；
4. 信息不足时对应字段给空字符串或空数组。

## 对话历史
{history}
"""


def _turn_field(turn, name: str) -> str:
    """兼容 DialogueTurn 对象与 dict 两种形态"""
    value = getattr(turn, name, None)
    if value is None and isinstance(turn, dict):
        value = turn.get(name)
    return (value or '').strip()


def build_history_text(history: List, limit: int = 8) -> str:
    """把最近若干轮拼成纯文本，供摘要 prompt 使用"""
    lines = []
    for turn in history[-limit:]:
        role = _turn_field(turn, 'role')
        content = _turn_field(turn, 'content')
        if not content:
            continue
        label = '用户' if role == 'user' else '助手'
        lines.append(f"{label}: {content[:300]}")
    return '\n'.join(lines)


def parse_summary(text: str) -> Optional[Dict]:
    """从 LLM 输出里解析摘要 JSON，失败返回 None"""
    if not text:
        return None
    raw = text.strip()
    raw = re.sub(r'^```(?:json)?', '', raw).strip()
    raw = re.sub(r'```$', '', raw).strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    focus = str(data.get('focus') or '').strip()[:30]
    sections = data.get('asked_sections') or []
    if not isinstance(sections, list):
        sections = []
    sections = [str(x).strip()[:12] for x in sections if str(x).strip()][:MAX_SECTIONS]

    opens = data.get('open_questions') or []
    if not isinstance(opens, list):
        opens = []
    opens = [str(x).strip()[:40] for x in opens if str(x).strip()][:MAX_OPEN_QUESTIONS]

    if not focus and not sections and not opens:
        return None
    return {'focus': focus, 'asked_sections': sections, 'open_questions': opens}


def format_summary(summary: Optional[Dict]) -> str:
    """把摘要格式化成注入用文本；空摘要返回空串"""
    if not summary:
        return ''
    parts = []
    focus = (summary.get('focus') or '').strip()
    if focus:
        parts.append(f"当前关注：{focus}")
    sections = summary.get('asked_sections') or []
    if sections:
        parts.append("已问章节：" + '、'.join(sections))
    opens = summary.get('open_questions') or []
    if opens:
        parts.append("未解决：" + '；'.join(opens))
    text = '\n'.join(parts)
    return text[:MAX_SUMMARY_CHARS]


def extract_summary(llm_client, history: List) -> Optional[Dict]:
    """调 LLM 生成三槽摘要；任何异常都返回 None（调用方静默跳过）"""
    if not llm_client or not history:
        return None
    history_text = build_history_text(history)
    if not history_text:
        return None

    messages = [
        {"role": "system", "content": "你是一个只输出 JSON 的对话摘要助手。"},
        {"role": "user", "content": SUMMARY_PROMPT.format(history=history_text)},
    ]
    try:
        raw = llm_client.chat(messages, temperature=0.0, max_tokens=512)
    except Exception as e:
        logger.warning(f"生成会话摘要失败: {e}")
        return None
    return parse_summary(raw)
