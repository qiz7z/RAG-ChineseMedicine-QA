# -*- coding: utf-8 -*-
"""
会话管理器
===========
管理多轮对话的会话（session），支持：
  - 创建/获取/删除会话
  - 每个会话维护独立的对话历史
  - 会话自动过期清理

设计说明：
  - 每个会话 ID 对应一个 DialogueManager 实例
  - 会话对象存在内存中；开启 ENABLE_MEMORY_PERSIST 时，会话与轮次
    同时写入 metadata.db，服务重启后可按 session_id 恢复
"""
import uuid
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from config import ENABLE_MEMORY_PERSIST

logger = logging.getLogger(__name__)


class SessionInfo:
    """单个会话的信息"""

    def __init__(self, session_id: str, dialogue_manager, guard=None,
                 user_id: str = 'default'):
        from generation.guard import GuardChecker

        self.session_id = session_id
        self.dialogue_manager = dialogue_manager
        # 守卫持有「本会话是否已通过」的信任状态，必须按会话独立
        self.guard = guard or GuardChecker()
        # 会话归属：后续长期记忆按 user_id 聚合（跨会话）
        self.user_id = user_id or 'default'
        self.created_at = datetime.now()
        self.last_active = datetime.now()

    def update_active(self):
        """更新最后活跃时间"""
        self.last_active = datetime.now()

    def is_expired(self, timeout_hours: int = 24) -> bool:
        """检查会话是否过期"""
        return datetime.now() - self.last_active > timedelta(hours=timeout_hours)

    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        history = []
        for turn in self.dialogue_manager.history:
            history.append({
                "role": turn.role,
                "content": turn.content[:200],  # 截断显示
            })
        return history

    @property
    def turn_count(self) -> int:
        return self.dialogue_manager.turn_count


class SessionManager:
    """
    会话管理器。

    管理所有活跃会话，支持创建、获取、删除和自动过期清理。

    使用方式：
        sm = SessionManager(generator)
        session_id = sm.create_session()
        generator = sm.get_generator(session_id)
        sm.delete_session(session_id)
    """

    # 会话过期时间（小时）
    SESSION_TIMEOUT_HOURS = 24

    # 过期清理的最小间隔（秒），避免每次创建会话都全量扫描
    CLEANUP_INTERVAL_SECONDS = 600

    def __init__(self, generator_factory=None):
        """
        Args:
            generator_factory: 生成 Generator 实例的工厂函数。
                              每个新会话会通过此工厂创建一个独立的 Generator
                              （含独立的 DialogueManager）。
                              如果为 None，则使用默认创建方式。
        """
        self.sessions: Dict[str, SessionInfo] = {}
        self.generator_factory = generator_factory
        self._last_cleanup = time.time()

    def _maybe_cleanup(self) -> None:
        """距上次清理超过 CLEANUP_INTERVAL_SECONDS 时清理一次过期会话。"""
        if time.time() - self._last_cleanup < self.CLEANUP_INTERVAL_SECONDS:
            return
        self._last_cleanup = time.time()
        try:
            self.cleanup_expired()
        except Exception as e:
            logger.warning(f"清理过期会话失败: {e}")

    def create_session(self, user_id: str = 'default') -> str:
        """
        创建新会话。

        Args:
            user_id: 会话归属用户（前端通过 X-User-Id 传入；不传则为 default）

        Returns:
            会话 ID
        """
        self._maybe_cleanup()
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        # 先按真实 user_id 落库，再创建 dm（dm 构造时还会 touch 一次，只更新活跃时间）
        if ENABLE_MEMORY_PERSIST:
            try:
                from memory.store import get_memory_store
                get_memory_store().touch_session(session_id, user_id)
            except Exception as e:
                logger.warning(f"会话落库失败（不影响对话）: {e}")

        # 创建 DialogueManager（不绑定 LLM，指代消解由主 Generator 负责）
        dm = self._new_dialogue_manager(session_id)

        session_info = SessionInfo(session_id, dm, user_id=user_id)
        self.sessions[session_id] = session_info

        logger.info(f"创建会话: {session_id}（user={user_id}）")
        return session_id

    def _new_dialogue_manager(self, session_id: str):
        """创建绑定了 session_id 的对话管理器（开启持久化时会自动恢复历史）"""
        from generation.dialogue import DialogueManager
        return DialogueManager(llm_client=None, session_id=session_id)

    def _restore_session(self, session_id: str, user_id: str = 'default') -> Optional[SessionInfo]:
        """服务重启后，按库里的 session_id 重建内存会话。

        历史由 DialogueManager 构造时自动恢复，这里只负责把对象放回内存表。
        会话归属不一致时拒绝恢复——session_id 可猜，不能靠它跨用户读别人的历史。
        """
        try:
            from memory.store import get_memory_store
            owner = get_memory_store().get_session_user_id(session_id)
        except Exception:
            owner = None

        if owner and owner != user_id:
            logger.warning(f"会话 {session_id} 属于用户 {owner}，拒绝用户 {user_id} 恢复")
            return None

        try:
            dm = self._new_dialogue_manager(session_id)
        except Exception as e:
            logger.warning(f"恢复会话 {session_id} 失败: {e}")
            return None
        if not dm.history:
            return None
        session_info = SessionInfo(session_id, dm, user_id=owner or user_id)
        self.sessions[session_id] = session_info
        logger.info(f"已恢复会话: {session_id}（{len(dm.history)} 条消息）")
        return session_info

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        获取会话信息。

        Args:
            session_id: 会话 ID

        Returns:
            SessionInfo 对象，如不存在则返回 None
        """
        session = self.sessions.get(session_id)
        if session:
            session.update_active()
        return session

    def get_or_create(self, session_id: Optional[str], user_id: str = 'default') -> str:
        """
        获取或创建会话。

        如果 session_id 为 None、不存在、或归属其他用户，则创建新会话。

        Args:
            session_id: 会话 ID（可为 None）
            user_id: 当前请求的用户标识（X-User-Id）

        Returns:
            有效的会话 ID
        """
        user_id = user_id or 'default'

        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if session.user_id == user_id:
                session.update_active()
                return session_id
            # 归属不一致：不把别人的会话交给当前用户，另开一个
            logger.warning(
                f"会话 {session_id} 属于用户 {session.user_id}，为 {user_id} 新建会话"
            )
            return self.create_session(user_id)

        # 服务重启后内存表为空，但库里可能还有该会话 → 恢复而不是新建
        if session_id and ENABLE_MEMORY_PERSIST:
            try:
                from memory.store import get_memory_store
                if get_memory_store().session_exists(session_id, user_id=user_id):
                    restored = self._restore_session(session_id, user_id)
                    if restored:
                        return session_id
            except Exception as e:
                logger.warning(f"按 session_id 恢复失败，改为新建: {e}")

        return self.create_session(user_id)

    def get_dialogue_manager(self, session_id: str):
        """
        获取指定会话的对话管理器。

        Args:
            session_id: 会话 ID

        Returns:
            DialogueManager 实例，如不存在则返回 None
        """
        session = self.get_session(session_id)
        if session:
            return session.dialogue_manager
        return None

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话。

        Args:
            session_id: 会话 ID

        Returns:
            是否删除成功
        """
        existed = session_id in self.sessions
        if existed:
            del self.sessions[session_id]

        # 库里软删除（轮次保留，便于排查）
        if ENABLE_MEMORY_PERSIST:
            try:
                from memory.store import get_memory_store
                get_memory_store().delete_session(session_id)
            except Exception as e:
                logger.warning(f"软删除会话失败: {e}")

        if existed:
            logger.info(f"删除会话: {session_id}")
        return existed

    def list_sessions(self) -> List[str]:
        """列出所有活跃会话 ID"""
        return list(self.sessions.keys())

    def cleanup_expired(self) -> int:
        """
        清理过期会话。

        Returns:
            清理的会话数量
        """
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.SESSION_TIMEOUT_HOURS)
        ]
        for sid in expired_ids:
            del self.sessions[sid]
        if expired_ids:
            logger.info(f"清理过期会话: {len(expired_ids)} 个")
        return len(expired_ids)

    @property
    def active_count(self) -> int:
        """活跃会话数"""
        return len(self.sessions)
