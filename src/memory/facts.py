# -*- coding: utf-8 -*-
"""
长期记忆（P2：用户事实库）
========================

从对话里抽取「关于用户的长期事实」——过敏史、在服药物、特殊人群、基础病况。
按 user_id 聚合、跨会话生效；回答时注入 prompt，让禁忌/相互作用提醒
带上用户个人情况（"我对人参过敏"→ 后续问人参自动给出警示）。

与 P1 三槽摘要的分工：
    摘要 = 这场会话在聊什么（会话级：focus / 已问章节 / 未解决问题）
    事实 = 用户是什么情况（用户级：过敏 / 用药 / 人群 / 病况，跨会话）

设计约束（与 summarizer 一致）：
    1. 抽取失败静默——事实只是增强，不能影响回答主流程；
    2. 只认 JSON——解析不到就放弃这一轮，宁可没有事实也不要脏文本；
    3. 关键词前置过滤——不相关的轮次不调 LLM，控制成本；
    4. 纠错保护——用户明确纠正或要求记住的事实置保护位，
       容量淘汰不删；旧信息被纠正时显式删除，不靠悄悄覆盖。
"""
import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_FACT_CHARS = 40          # 单条事实长度上限
MAX_FACTS_PER_TURN = 5       # 单次抽取产出上限
MAX_FACTS_TEXT_CHARS = 300   # 注入 prompt 的文本上限

# 事实类型 → 注入文本的标签
FACT_TYPE_LABELS = {
    "allergy": "过敏史",
    "medication": "在服药物",
    "population": "特殊人群",
    "condition": "基础病况",
    "other": "其他",
}

# 关键词前置过滤：命中才值得调 LLM（确定性规则，成本闸门）
FACT_SIGNAL_WORDS = (
    "过敏", "不能吃", "不能用", "忌口", "不良反应",
    "在吃", "在服", "正在吃", "正在服", "正在服用", "吃着", "用着",
    "怀孕", "孕妇", "哺乳", "备孕",
    "孩子", "儿童", "小孩", "宝宝", "老人", "老年",
    "高血压", "糖尿病", "心脏病", "肝病", "肾病", "胃病", "手术",
    "记住", "别忘", "不要忘", "一定要记住", "千万记住",
    "说错", "记错", "我不过敏", "没过敏", "停药", "停了",
)

FACTS_PROMPT = """你是用户事实抽取助手。阅读下面的对话，只抽取与**用户本人情况**相关的长期事实，输出 JSON 对象：

{{
  "facts": [
    {{"type": "allergy", "content": "对人参过敏", "remember": false}}
  ],
  "retract": []
}}

字段说明：
- facts：本次确认的事实。type 取值：
  allergy=过敏史、medication=正在服用的药物、population=特殊人群（如 孕期/6岁儿童）、
  condition=基础病况（如 高血压）、other=其他与用药相关的用户情况；
- remember：用户明确要求记住（"记住我对人参过敏"）或这是对旧信息的纠正时给 true；
- retract：用户否认/纠正的**旧事实**，照抄下方「已有事实」里的原文（如 "对三七过敏"）；
  用户说"我不过敏了""说错了，不是X是Y"时必须把被推翻的旧事实放进来。

要求：
1. 只抽用户自身的情况，不抽「问了什么药」（那是会话摘要的事）；
2. 药名保留药典原名，不要改写；content 不超过 20 字；
3. 没有新事实就给空数组；只输出 JSON，不要解释、不要代码块标记。

## 已有事实
{current_facts}

## 对话
{history}
"""


def has_fact_signal(text: str) -> bool:
    """确定性关键词过滤：像不像包含用户事实的发言"""
    if not text:
        return False
    return any(w in text for w in FACT_SIGNAL_WORDS)


def _turn_field(turn, name: str) -> str:
    """兼容 DialogueTurn 对象与 dict 两种形态"""
    value = getattr(turn, name, None)
    if value is None and isinstance(turn, dict):
        value = turn.get(name)
    return (value or '').strip()


def build_history_text(history: List, limit: int = 8) -> str:
    """把最近若干轮拼成纯文本，供事实抽取 prompt 使用"""
    lines = []
    for turn in history[-limit:]:
        role = _turn_field(turn, 'role')
        content = _turn_field(turn, 'content')
        if not content:
            continue
        label = '用户' if role == 'user' else '助手'
        lines.append(f"{label}: {content[:300]}")
    return '\n'.join(lines)


def format_facts(facts: Optional[List[Dict]]) -> str:
    """把事实列表格式化成注入 prompt 的文本；没有事实返回空串"""
    if not facts:
        return ''
    parts = []
    for f in facts:
        label = FACT_TYPE_LABELS.get(f.get('fact_type') or f.get('type') or '', '其他')
        content = (f.get('content') or '').strip()
        if content:
            parts.append(f"- {label}：{content}")
    return '\n'.join(parts)[:MAX_FACTS_TEXT_CHARS]


def parse_facts(text: str) -> Optional[Dict]:
    """从 LLM 输出里解析事实 JSON，失败返回 None。

    返回 {'facts': [{'type','content','remember'}], 'retract': [str]}，
    两部分都可能为空列表；整体无产出时返回 None。
    """
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

    facts = []
    raw_facts = data.get('facts') or []
    if isinstance(raw_facts, list):
        for item in raw_facts[:MAX_FACTS_PER_TURN]:
            if not isinstance(item, dict):
                continue
            ftype = str(item.get('type') or '').strip() or 'other'
            if ftype not in FACT_TYPE_LABELS:
                ftype = 'other'
            content = str(item.get('content') or '').strip()[:MAX_FACT_CHARS]
            if not content:
                continue
            facts.append({
                'type': ftype,
                'content': content,
                'remember': bool(item.get('remember')),
            })

    retract = []
    raw_retract = data.get('retract') or []
    if isinstance(raw_retract, list):
        retract = [
            str(x).strip()[:MAX_FACT_CHARS]
            for x in raw_retract[:MAX_FACTS_PER_TURN] if str(x).strip()
        ]

    if not facts and not retract:
        return None
    return {'facts': facts, 'retract': retract}


def extract_facts(llm_client, history: List, current_facts: str = '') -> Optional[Dict]:
    """调 LLM 抽取用户事实；任何异常都返回 None（调用方静默跳过）"""
    if not llm_client or not history:
        return None
    history_text = build_history_text(history)
    if not history_text:
        return None

    messages = [
        {"role": "system", "content": "你是一个只输出 JSON 的用户事实抽取助手。"},
        {"role": "user", "content": FACTS_PROMPT.format(
            current_facts=current_facts.strip() or '（无）',
            history=history_text,
        )},
    ]
    try:
        raw = llm_client.chat(messages, temperature=0.0, max_tokens=512)
    except Exception as e:
        logger.warning(f"抽取用户事实失败: {e}")
        return None
    return parse_facts(raw)
