# -*- coding: utf-8 -*-
"""
领域守卫（标准版）
==================
两层过滤：关键词快通道（0ms）→ LLM 语义判定。
独立实现（关键词表与主项目同源），通过一个普通函数暴露，供 LCEL
链的 RunnableBranch 条件与 API 服务层共用。
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# 医药领域关键词 —— 命中任一直接通过（与主项目同源，含药材名/剂型/药理概念）
ON_TOPIC_KEYWORDS = [
    "药", "药材", "药品", "药典", "中药", "中草药", "方剂", "成方",
    "性味", "归经", "功能", "主治", "用法", "用量", "性状", "鉴别",
    "检查", "含量", "测定", "浸出物", "炮制", "制法", "贮藏", "规格",
    "处方", "制剂", "饮片", "提取物", "特征图谱", "指纹图谱",
    "煎服", "口服", "外用", "吞服", "冲服", "另煎",
    "功效", "毒性", "副作用", "禁忌", "注意事项", "不良反应",
    "孕妇", "慎用", "忌用", "禁用",
    "色谱", "光谱", "滴定", "干燥", "粉碎", "筛分",
    "丸", "散", "膏", "丹", "汤", "颗粒", "片剂", "胶囊", "口服液",
    "人参", "黄芪", "当归", "黄连", "甘草", "川芎", "白芍", "茯苓",
    "麻黄", "何首乌", "丹参", "陈皮", "半夏", "枸杞", "天麻", "麦冬",
    "金银花", "柴胡", "地黄", "五味子", "山药", "桔梗", "百合",
    "决明子", "泽泻", "白术", "西洋参", "赤芍", "黄芩", "红花",
    "桂枝", "附子", "天南星", "酸枣仁", "远志", "柏子仁", "薏苡仁",
    "连翘", "板蓝根", "水蛭", "紫苏", "肉桂", "苏木",
    "补气", "补血", "滋阴", "壮阳", "清热", "解毒", "活血", "化瘀",
    "利水", "渗湿", "安神", "平肝", "息风", "发汗", "解表",
    "健脾", "润肺", "养心", "益胃", "生津", "固表", "托毒",
    "收敛", "固涩", "调经", "敛阴", "止汗", "柔肝",
    "通则", "重金属", "灰分", "水分",
]

# 明显无关关键词 —— 命中任一直接拒绝
OFF_TOPIC_KEYWORDS = [
    "股票", "基金", "理财", "投资", "比特币", "加密货币",
    "游戏", "电竞", "攻略", "通关",
    "编程", "代码", "python", "java", "javascript", "github",
    "足球", "篮球", "棒球", "网球", "奥运",
    "天气", "新闻", "股票行情",
]


def keyword_guard(query: str) -> Tuple[bool, str]:
    """关键词快通道。

    Returns:
        (verdict, reason) — verdict=True 表示放行/拒绝由 reason 说明；
        reason: "keyword_on" / "keyword_off" / "ambiguous"
    """
    q = query.lower()
    for kw in OFF_TOPIC_KEYWORDS:
        if kw in q:
            return False, "keyword_off"
    for kw in ON_TOPIC_KEYWORDS:
        if kw in q:
            return True, "keyword_on"
    return True, "ambiguous"  # 词表无法判定 → 交给 LLM 层


def llm_guard(chat_model, question: str) -> bool:
    """LLM 语义判定（推理类模型需较大 token 预算容纳思考内容）"""
    from prompts import GUARD_PROMPT

    try:
        chain = GUARD_PROMPT | chat_model.bind(temperature=0, max_tokens=512)
        result = chain.invoke({"question": question}).content.strip()
        return "无关" not in result
    except Exception as e:
        logger.warning(f"LLM 守卫判断失败: {e}，默认放行")
        return True
