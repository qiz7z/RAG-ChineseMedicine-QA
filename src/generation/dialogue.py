# -*- coding: utf-8 -*-
"""
多轮对话管理模块
=================
管理多轮对话上下文，支持：

1. 对话历史保持：维护最近 N 轮对话
2. 指代消解：通过 LLM 将"它"、"这个药"等指代词替换为具体药品名
3. 上下文窗口控制：防止对话历史超出 token 限制

使用方式：
    dm = DialogueManager(llm_client)
    dm.add_user_message("人参的性味归经是什么？")
    dm.add_assistant_message("人参的性味为甘、微苦...")
    resolved = dm.resolve_query("那它的功能主治呢？")
    # resolved => "那人参的功能主治呢？"
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from config import (
    DIALOGUE_MAX_HISTORY, DIALOGUE_CONTEXT_WINDOW,
    ENABLE_MEMORY_PERSIST, MEMORY_RESTORE_TURNS,
    ENABLE_MEMORY_SUMMARY, MEMORY_SUMMARY_MIN_TURNS, MEMORY_SUMMARY_INTERVAL,
)

logger = logging.getLogger(__name__)

# 摘要生成放到后台线程，避免多一次 LLM 调用（2~5s）拖慢回答。
# 单个会话同一时刻最多一个任务在跑，2 个 worker 足够；失败只记日志。
_SUMMARY_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='mem-summary')


@dataclass
class DialogueTurn:
    """单轮对话"""
    role: str           # "user" 或 "assistant"
    content: str        # 消息内容
    query: str = ""     # 原始查询（仅 user 角色有）


class DialogueManager:
    """
    多轮对话管理器。

    负责维护对话历史、执行指代消解、控制上下文窗口。

    使用方式：
        dm = DialogueManager(llm_client)
        # 第一轮
        resolved_q = dm.resolve_query("人参的性味归经是什么？")
        dm.add_user_message(resolved_q)
        dm.add_assistant_message(answer)
        # 第二轮
        resolved_q = dm.resolve_query("那它的功能主治呢？")
        # resolved_q => "那人参的功能主治呢？"
    """

    def __init__(
        self,
        llm_client=None,
        max_history: int = None,
        session_id: str = None,
        store=None,
        drug_names: List[str] = None,
    ):
        """
        Args:
            llm_client: LLM 客户端实例（用于指代消解）。如不提供则跳过指代消解。
            max_history: 保留的最大对话轮数
            session_id: 会话 ID；提供且开启持久化时，会从库里恢复最近几轮
            store: 记忆存储实例（默认取全局单例）
            drug_names: 药典药名清单，用于确定性地跟踪「当前讨论的药」
        """
        self.llm_client = llm_client
        self.max_history = max_history or DIALOGUE_MAX_HISTORY
        self.session_id = session_id
        self.history: List[DialogueTurn] = []

        # 短期记忆的状态槽：当前焦点药品。
        # 「它」的指向由这个变量决定，而不是每轮让 LLM 从历史文本里重新推断。
        self.current_drug: Optional[str] = None
        self.drug_names: List[str] = sorted(drug_names, key=len, reverse=True) if drug_names else []

        # 三槽摘要：focus / asked_sections / open_questions
        self.summary: Optional[Dict] = None
        self._last_summary_turn = 0
        # 累计轮数：不随 _trim_history 减少，用于摘要节流
        # （若用 turn_count，超过 max_history 后恒为 5，摘要永远不再刷新）
        self._total_turns = 0

        # 记忆持久化：失败即退化为纯内存，绝不影响对话主流程
        self.store = None
        if ENABLE_MEMORY_PERSIST:
            try:
                from memory.store import get_memory_store
                self.store = store or get_memory_store()
            except Exception as e:
                logger.warning(f"记忆存储初始化失败，退化为纯内存: {e}")

        if self.store and self.session_id:
            self.store.touch_session(self.session_id)
            self._restore()
            self.summary = self.store.load_summary(self.session_id)
        # 恢复会话时同步累计轮数，避免节流从 0 重来
        self._total_turns = self.turn_count

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------

    def _restore(self):
        """从库里恢复最近 max_history 轮对话"""
        try:
            rows = self.store.load_turns(self.session_id, self.max_history)
        except Exception as e:
            logger.warning(f"恢复对话历史失败: {e}")
            return
        self.history = [
            DialogueTurn(
                role=r.get('role', ''),
                content=r.get('content', ''),
                query=r.get('resolved_query') or '',
            )
            for r in rows
            if r.get('content')
        ]
        # 恢复焦点槽：取最近一条用户消息里的实体（药名或表外话题）。
        # turn.query 是消解后的文本（"黄麻能补血吗"），比 content（"它能补血吗"）
        # 更适合反推实体，两个都试。
        for turn in reversed(self.history):
            if turn.role == 'user':
                matched = (
                    self._match_drug(turn.query) or self._match_drug(turn.content)
                    or self._extract_topic_entity(turn.query)
                    or self._extract_topic_entity(turn.content)
                )
                if matched:
                    self.current_drug = matched
                    break
        if self.history:
            logger.info(f"已从库恢复会话 {self.session_id} 的 {len(self.history)} 条消息")

    def bind_session(self, session_id: str):
        """切换到指定会话（不同会话历史相互隔离），并加载其历史"""
        if not session_id or session_id == self.session_id:
            return
        self.session_id = session_id
        self.history.clear()
        if self.store:
            self.store.touch_session(session_id)
            self._restore()

    def _persist_turn(self, role: str, content: str, resolved_query: str = ''):
        if not (self.store and self.session_id):
            return
        self.store.touch_session(self.session_id)
        self.store.append_turn(self.session_id, role, content, resolved_query)

    # ----------------------------------------------------------
    # 会话摘要（三槽：关注点 / 已问章节 / 未解决问题）
    # ----------------------------------------------------------

    def get_summary_text(self) -> str:
        """返回可直接注入 prompt 的摘要文本；无摘要时返回空串"""
        from memory.summarizer import format_summary
        return format_summary(self.summary)

    def schedule_summary(self):
        """按轮次间隔触发后台摘要。

        放在 add_assistant_message 之后调用：此时一轮对话才完整。
        不阻塞回答，失败静默——摘要只是增强，不能影响主流程。
        """
        if not (ENABLE_MEMORY_SUMMARY and self.llm_client and self.session_id):
            return
        if self._total_turns < MEMORY_SUMMARY_MIN_TURNS:
            return
        # 已有摘要才按间隔节流；还没有摘要时允许下一轮立刻重试，
        # 否则一次失败（超时/格式不对）会让摘要永久缺失
        if self.summary and self._total_turns - self._last_summary_turn < MEMORY_SUMMARY_INTERVAL:
            return

        self._last_summary_turn = self._total_turns
        try:
            # 传副本：后台线程读历史时，主线程可能继续追加
            _SUMMARY_EXECUTOR.submit(
                self._refresh_summary, list(self.history), self.session_id
            )
        except Exception as e:
            logger.warning(f"提交摘要任务失败（不影响对话）: {e}")

    def _refresh_summary(self, history: List[DialogueTurn], session_id: str):
        """后台线程：调 LLM 生成摘要并落库"""
        try:
            from memory.summarizer import extract_summary
            summary = extract_summary(self.llm_client, history)
            if not summary:
                return
            self.summary = summary
            if self.store and session_id:
                self.store.save_summary(session_id, summary)
            logger.info(f"会话 {session_id} 摘要已更新: {summary.get('focus', '')}")
        except Exception as e:
            logger.warning(f"刷新会话摘要失败（不影响对话）: {e}")

    # ----------------------------------------------------------
    # 对话历史管理
    # ----------------------------------------------------------

    def add_user_message(self, content: str, original_query: str = ""):
        """添加用户消息到对话历史"""
        self.history.append(DialogueTurn(
            role="user",
            content=content,
            query=original_query or content,
        ))
        self._trim_history()
        # content 是消解后的查询，原始问题一并落库供回溯
        self._persist_turn('user', content, original_query or content)
        # 更新焦点槽：优先取用户原话里的实体（药典药名，其次表外话题实体）
        matched = (
            self._match_drug(original_query or "") or self._match_drug(content)
            or self._extract_topic_entity(original_query or "")
            or self._extract_topic_entity(content)
        )
        if matched:
            self.current_drug = matched

    def add_assistant_message(self, content: str):
        """添加助手消息到对话历史"""
        self.history.append(DialogueTurn(
            role="assistant",
            content=content,
        ))
        self._trim_history()
        self._persist_turn('assistant', content)
        self._total_turns += 1
        # 一轮完整对话后再考虑刷新摘要（内部按轮次间隔节流）
        self.schedule_summary()

    def clear(self):
        """清空对话历史（同时清除持久化轮次）"""
        self.history.clear()
        if self.store and self.session_id:
            try:
                self.store.clear_turns(self.session_id)
            except Exception as e:
                logger.warning(f"清空持久化轮次失败: {e}")

    def _trim_history(self):
        """裁剪对话历史，保留最近 max_history 轮"""
        max_turns = self.max_history * 2  # 每轮 = user + assistant
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

    # ----------------------------------------------------------
    # 获取对话历史（OpenAI 格式）
    # ----------------------------------------------------------

    def get_history_messages(self) -> List[Dict]:
        """
        获取对话历史（OpenAI messages 格式）。

        用于注入到 LLM 的 messages 列表中。

        Returns:
            OpenAI 格式的消息列表
        """
        # 控制上下文窗口大小
        messages = []
        total_chars = 0

        # 从最近的对话开始，向前累加，直到达到窗口限制
        for turn in reversed(self.history):
            if total_chars + len(turn.content) > DIALOGUE_CONTEXT_WINDOW:
                break
            messages.insert(0, {
                "role": turn.role,
                "content": turn.content,
            })
            total_chars += len(turn.content)

        return messages

    # ----------------------------------------------------------
    # 指代消解
    # ----------------------------------------------------------

    def resolve_query(self, current_query: str) -> str:
        """
        对当前查询执行指代消解。

        优先走「焦点槽」（`current_drug`）——这是短期记忆的状态，
        指代对象由它决定，不依赖模型每轮重新推断；
        只有焦点为空时才回退 LLM（并在 prompt 里带上会话摘要）。

        Args:
            current_query: 用户当前查询

        Returns:
            消解后的查询（指代词替换为具体药品名）
        """
        # 1. 当前问题自带实体 → 它就是新的焦点，无需消解
        #    包括药典药名，以及药典外的话题实体（如「黄麻」——药典只有「麻黄」）
        matched = self._match_drug(current_query) or self._extract_topic_entity(current_query)
        if matched:
            self.current_drug = matched
            return current_query

        # 2. 不含指代词 → 原样返回
        if not self._has_coreference(current_query):
            return current_query

        # 3. 焦点槽里有药 → 直接替换，不调模型
        if self.current_drug:
            resolved = self._replace_coreference(current_query, self.current_drug)
            if resolved != current_query:
                logger.info(f"指代消解(焦点槽): '{current_query}' -> '{resolved}'")
                return resolved

        # 4. 焦点为空：有历史时交给 LLM，结合摘要兜底
        if not self.history or not self.llm_client:
            return current_query

        try:
            from generation.prompts import build_coreference_prompt

            history_messages = self.get_history_messages()
            # 摘要一并喂给消解：对话超过 5 轮后，"人参"这类实体已滚出历史窗口，
            # 只靠最近几轮会把"它"还原成后来提到的别的药。
            messages = build_coreference_prompt(
                current_query, history_messages, self.get_summary_text()
            )

            resolved = self.llm_client.chat(
                messages,
                temperature=0.0,   # 指代消解需要确定性输出
                # 预算需容纳推理类模型的思考内容；对非推理模型无影响
                max_tokens=1024,
            )

            resolved = resolved.strip()

            # 简单校验：消解结果不应为空，且长度不应超过原查询的3倍
            if resolved and len(resolved) < len(current_query) * 3:
                logger.info(f"指代消解: '{current_query}' -> '{resolved}'")
                return resolved
            else:
                logger.warning("指代消解结果异常，使用原始查询")
                return current_query

        except Exception as e:
            logger.warning(f"指代消解失败: {e}，使用原始查询")
            return current_query

    # 指代词表：长的排在前面，替换时避免「它」先把「它们」截断
    COREFERENCE_WORDS = (
        "前面提到的", "刚才说的", "这种药材", "这个药", "这种药",
        "该药材", "该药", "它们", "上述", "它",
    )

    def _has_coreference(self, query: str) -> bool:
        """
        快速检查查询中是否包含可能的指代词。

        Args:
            query: 用户查询

        Returns:
            是否包含指代词
        """
        # 「其他 / 其它」里的「它」不是指代，先剔掉再判断
        text = query.replace("其他", "").replace("其它", "")
        return any(word in text for word in self.COREFERENCE_WORDS)

    def _match_drug(self, text: str) -> Optional[str]:
        """从文本里找出药典药名（按长度降序匹配，优先长名，如「人参叶」优于「人参」）"""
        if not text or not self.drug_names:
            return None
        plain = text.replace(" ", "")
        for name in self.drug_names:
            if name and name in plain:
                return name
        return None

    # 话题实体抽取的排除词：延续问句的主语不是实体，不许抢焦点。
    # 属性/章节词（「味道怎么样」「禁忌呢」）、人群/条件词（「孕妇能吃吗」）、
    # 泛指/疑问词都在这里。
    _TOPIC_NON_ENTITIES = frozenset({
        # 属性 / 药典章节
        "味道", "口味", "气味", "颜色", "性状", "禁忌", "副作用", "用法", "用量",
        "价格", "价钱", "性味", "归经", "功效", "功能", "主治", "作用", "疗效",
        "效果", "药效", "毒性", "不良反应", "人群", "服用", "吃法", "服法",
        "保存", "存放", "产地", "成分", "含量", "营养", "药性", "药理", "配伍",
        "注意", "事项", "宜忌", "适应症", "疗程", "剂量", "规格", "鉴别",
        "检查", "炮制", "贮藏", "来源", "类别", "品种", "药材", "药品", "药物",
        "中药", "西药", "补品", "东西", "方法", "说明书", "包装", "保质期",
        # 剂型（「口服液说明书」是延续问句，不是新话题）
        "口服液", "胶囊", "颗粒", "冲剂", "糖浆", "药片", "丸剂", "制剂",
        # 人群 / 条件 / 状态
        "孕妇", "产妇", "哺乳", "经期", "月经", "儿童", "小孩", "孩子", "婴儿",
        "老人", "成人", "空腹", "饭前", "饭后", "睡前", "体质", "过敏", "有毒",
        "小时", "分钟",
        # 疑问 / 泛指
        "什么", "怎么", "如何", "多少", "哪个", "哪些", "这个", "那个", "这种",
        "其他", "其它", "前面", "刚才", "上述", "为什么", "是不是", "能不能",
        "有没有", "可以",
    })

    # 句首客套前缀：剥掉后才找话题实体（「请问黄麻什么口味」→「黄麻」）
    _TOPIC_FILLER_PREFIXES = (
        "请问", "你好", "您好", "我想问", "想问一下", "问一下", "麻烦", "请教",
    )

    # 句首封闭类词：这些不是话题，跳过它们找第一个名词
    # （代词「那个」、数量词「3克」、助词/连词/标点等）
    _TOPIC_SKIP_FLAGS = frozenset("rmyqxcupdt")

    def _extract_topic_entity(self, text: str) -> Optional[str]:
        """
        从文本里启发式抽取「话题实体」，可能是药典外的（如「黄麻」——
        药典里只有「麻黄」，`_match_drug` 认不出它）。

        规则：剥掉客套前缀后，跳过句首封闭类词（「那个」「3克」），
        取第一个名词——
        - 名词且不在排除词表 → 视作新话题实体（「黄麻什么口味」→「黄麻」，
          「黄麻叶怎么吃」→「黄麻叶」）；
        - 落在排除词表里，或第一个实词不是名词 → 这是延续问句
          （「孕妇能吃吗」「味道怎么样」「保存方法」），返回 None，焦点保持不变。

        确定性规则，不调模型。
        """
        if not text:
            return None
        # 含指代词的句子不是引入新话题（"它"指什么由焦点槽回答）
        if self._has_coreference(text):
            return None
        plain = text.replace(" ", "")
        for prefix in self._TOPIC_FILLER_PREFIXES:
            if plain.startswith(prefix):
                plain = plain[len(prefix):]
                break
        if not plain:
            return None
        try:
            import jieba.posseg as pseg
        except ImportError:
            return None

        candidate = ""
        for word, flag in pseg.cut(plain):
            w = word.strip()
            if not w:
                continue
            if not candidate:
                # 还没找到实体：跳过句首封闭类词与单字碎片（「3克」的「克」）
                if flag[0] in self._TOPIC_SKIP_FLAGS or len(w) < 2:
                    continue
                # 第一个实词不是名词 → 动词/形容词开头，是延续问句
                if not flag.startswith("n"):
                    return None
                if w in self._TOPIC_NON_ENTITIES:
                    return None
                candidate = w
                continue
            # 已有候选：相邻名词合并（「黄麻」+「叶」→「黄麻叶」）
            if flag.startswith("n") and w not in self._TOPIC_NON_ENTITIES \
                    and len(candidate) + len(w) <= 6:
                candidate += w
            else:
                break
        return candidate or None

    def _replace_coreference(self, query: str, drug: str) -> str:
        """把第一个指代词替换成具体药名（确定性，不调模型）"""
        for word in self.COREFERENCE_WORDS:
            if word in query:
                return query.replace(word, drug, 1)
        return query

    # ----------------------------------------------------------
    # 状态查询
    # ----------------------------------------------------------

    def get_last_drug_name(self) -> Optional[str]:
        """
        从对话历史中提取最近提到的药品名。

        用于辅助检索过滤。

        Returns:
            最近的药品名（如无则返回 None）
        """
        # 从最近的用户查询中查找药品名
        for turn in reversed(self.history):
            if turn.role == "user":
                # 简单策略：从查询文本中提取引号内的内容或已知药品名
                # 更精确的药品名提取由 QueryParser 完成
                return turn.query
        return None

    @property
    def turn_count(self) -> int:
        """当前对话轮数"""
        return len(self.history) // 2

    def __repr__(self) -> str:
        return f"DialogueManager(turns={self.turn_count}, max_history={self.max_history})"
