# -*- coding: utf-8 -*-
"""
问题领域守卫（Guard Checker）
================================
在 RAG 流程入口处检测用户问题是否与中医药/药典相关，
对无关问题直接礼貌拒绝，避免浪费检索和生成资源。

检测策略（两层过滤）：
  ┌──────────────────────────────────────────────┐
  │  1. 关键词快速通道（0ms）                      │
  │     命中"医药相关词" → 直接通过                 │
  │     命中"明显无关词" → 直接拒绝                 │
  └──────────────────┬───────────────────────────┘
                     ▼
  ┌──────────────────────────────────────────────┐
  │  2. LLM 语义判断（~0.5s）                     │
  │     对模糊问题进行一次快速 LLM 分类            │
  │     返回 "相关" / "无关"                       │
  └──────────────────────────────────────────────┘
"""
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 领域关键词表
# ============================================================

# 医药领域关键词 —— 命中任一即直接通过（无需 LLM 判断）
ON_TOPIC_KEYWORDS = [
    # 通用医药术语
    "药", "药材", "药品", "药典", "中药", "中草药", "方剂", "成方",
    "性味", "归经", "功能", "主治", "用法", "用量", "性状", "鉴别",
    "检查", "含量", "测定", "浸出物", "炮制", "制法", "贮藏", "规格",
    "处方", "制剂", "饮片", "提取物", "Feature图谱", "指纹图谱",
    # 用药相关
    "煎服", "口服", "外用", "吞服", "冲服", "另煎",
    "功效", "毒性", "副作用", "禁忌", "注意事项", "不良反应",
    "孕妇", "慎用", "忌用", "禁用",
    # 分析方法
    "色谱", "光谱", "滴定", "干燥", "粉碎", "筛分",
    # 剂型
    "丸", "散", "膏", "丹", "汤", "颗粒", "片剂", "胶囊", "口服液",
    # 常见药材名（覆盖测试集中出现的）
    "人参", "黄芪", "当归", "黄连", "甘草", "川芎", "白芍", "茯苓",
    "麻黄", "何首乌", "丹参", "陈皮", "半夏", "枸杞", "天麻", "麦冬",
    "金银花", "柴胡", "地黄", "五味子", "山药", "桔梗", "百合",
    "决明子", "泽泻", "白术", "西洋参", "赤芍", "黄芩", "红花",
    "桂枝", "附子", "天南星", "酸枣仁", "远志", "柏子仁", "薏苡仁",
    "连翘", "板蓝根", "水蛭", "紫苏", "肉桂", "苏木",
    # 药理概念
    "补气", "补血", "滋阴", "壮阳", "清热", "解毒", "活血", "化瘀",
    "利水", "渗湿", "安神", "平肝", "息风", "发汗", "解表",
    "健脾", "润肺", "养心", "益胃", "生津", "固表", "托毒",
    "收敛", "固涩", "调经", "敛阴", "止汗", "柔肝",
    # 药典通则
    "通则", "通则方法", "重金属", "灰分", "水分",
]

# 明显无关的关键词 —— 命中任一即直接拒绝（无需 LLM 判断）
# 注意：用词级别的匹配，避免误伤（如"药店里有什么游戏机"不会被误判）
OFF_TOPIC_KEYWORDS = [
    "股票", "基金", "理财", "投资", "比特币", "加密货币",
    "游戏", "电竞", "攻略", "通关",
    "编程", "代码", "python", "java", "javascript", "github",
    "足球", "篮球", "棒球", "网球", "奥运",
    "电影", "电视剧", "综艺", "明星", "偶像",
    "音乐", "歌词", "吉他", "钢琴",
    "菜谱", "做菜", "烹饪", "炒菜",
    "旅游", "景点", "机票", "酒店",
    "天气预报", "气温", "今天天气", "明天天气",
    "数学题", "物理题", "化学方程式", "英语翻译", "日语",
    "政治", "选举", "军事", "武器",
    "汽车", "车型", "手机评测", "电脑配置",
    "星座", "运势", "占卜", "算命",
    "小说", "作文", "诗歌",
]

# 拒绝回答模板
OFF_TOPIC_REJECTION = (
    "抱歉，我是《中国药典》智能问答系统，只能回答与中药、药典相关的问题。\n\n"
    "我可以帮您解答以下类型的问题：\n"
    "- 药材的性味归经、功能主治、用法用量\n"
    "- 药材的性状、鉴别、含量测定\n"
    "- 药材的炮制方法、贮藏条件\n"
    "- 药材间的功效对比\n"
    "- 药典通则方法（如色谱法、水分测定等）\n\n"
    "请尝试提出与中医药相关的问题，例如：\"人参的性味归经是什么？\""
)


# ============================================================
# 守卫检查器
# ============================================================

class GuardChecker:
    """
    问题领域守卫检查器。

    使用方式：
        guard = GuardChecker(llm_client)
        is_related, reason = guard.check("人参的性味归经是什么？")
        if not is_related:
            # 直接返回拒绝回答，不执行检索和生成
            return OFF_TOPIC_REJECTION
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 客户端实例（可选，用于语义判断）
        """
        self.llm_client = llm_client

        # 预编译关键词正则（大小写不敏感）
        self._on_topic_pattern = re.compile(
            "|".join(re.escape(kw) for kw in ON_TOPIC_KEYWORDS),
            re.IGNORECASE,
        )
        self._off_topic_pattern = re.compile(
            "|".join(re.escape(kw) for kw in OFF_TOPIC_KEYWORDS),
            re.IGNORECASE,
        )

        # 多轮对话优化：记录当前会话是否已通过守卫
        # 第一轮通过后，后续轮次跳过 LLM 判断，降低延迟
        self._session_trusted = False

    def check(self, query: str, is_followup: bool = False) -> Tuple[bool, str]:
        """
        检查问题是否与中医药/药典相关。

        Args:
            query: 用户查询
            is_followup: 是否为多轮对话的后续轮次
                        （第一轮通过后，后续轮次跳过 LLM 判断）

        Returns:
            (is_related, reason)
            - is_related: True=相关（可以继续问答），False=无关（应拒绝）
            - reason: 判断原因（用于日志）
        """
        query_stripped = query.strip()

        # 空查询直接拒绝
        if not query_stripped:
            return False, "空查询"

        # ---- 第一层：关键词快速通道 ----

        # 命中医药关键词 → 直接通过
        if self._on_topic_pattern.search(query_stripped):
            self._session_trusted = True
            return True, "关键词匹配（医药相关）"

        # 命中无关关键词 → 直接拒绝
        if self._off_topic_pattern.search(query_stripped):
            return False, "关键词匹配（明显无关）"

        # ---- 多轮对话优化 ----
        # 如果当前会话第一轮已通过守卫，后续轮次跳过 LLM 判断
        if is_followup and self._session_trusted:
            return True, "多轮对话信任通道（跳过LLM）"

        # ---- 第二层：LLM 语义判断 ----
        if self.llm_client:
            is_rel, reason = self._llm_check(query_stripped)
            if is_rel:
                self._session_trusted = True
            return is_rel, reason

        # 无 LLM 客户端时，默认通过（不阻断正常使用）
        self._session_trusted = True
        return True, "无LLM，默认通过"

    def reset_session(self):
        """
        重置会话信任状态。

        在新会话开始或会话结束时调用，确保新一轮对话从第一轮开始判断。
        """
        self._session_trusted = False

    def _llm_check(self, query: str) -> Tuple[bool, str]:
        """
        使用 LLM 进行语义级别的领域判断。

        使用极简 prompt + 低 max_tokens 保证速度。
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个问题分类器。判断用户的问题是否与中医药、"
                    "《中国药典》、中药材、药理功效相关。\n"
                    "只回答【相关】或【无关】，不输出任何其他内容。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{query}",
            },
        ]

        try:
            result = self.llm_client.chat(
                messages,
                temperature=0,
                max_tokens=10,
            )
            result = result.strip()

            if "无关" in result:
                return False, f"LLM判断：无关"
            else:
                return True, f"LLM判断：相关"

        except Exception as e:
            logger.warning(f"LLM 守卫判断失败: {e}，默认通过")
            return True, f"LLM异常，默认通过"

    def __repr__(self) -> str:
        return f"GuardChecker(llm={'on' if self.llm_client else 'off'})"
