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
  - 会话存储在内存中（适用于单机部署）
  - 生产环境可替换为 Redis 存储
"""
import uuid
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SessionInfo:
    """单个会话的信息"""

    def __init__(self, session_id: str, dialogue_manager):
        self.session_id = session_id
        self.dialogue_manager = dialogue_manager
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

    def create_session(self) -> str:
        """
        创建新会话。

        Returns:
            会话 ID
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        # 为新会话创建独立的 Generator（含独立 DialogueManager）
        if self.generator_factory:
            generator = self.generator_factory()
        else:
            # 默认方式：复用已有的 Generator 但创建新的对话管理器
            # 注意：这里需要特殊处理，因为 Generator 内部有自己的 DialogueManager
            # 最佳实践是每个 session 持有独立的 DialogueManager
            generator = None

        # 创建 DialogueManager（不绑定 LLM，指代消解由主 Generator 负责）
        from generation.dialogue import DialogueManager
        dm = DialogueManager(llm_client=None)

        session_info = SessionInfo(session_id, dm)
        self.sessions[session_id] = session_info

        logger.info(f"创建会话: {session_id}")
        return session_id

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

    def get_or_create(self, session_id: Optional[str]) -> str:
        """
        获取或创建会话。

        如果 session_id 为 None 或不存在，则创建新会话。

        Args:
            session_id: 会话 ID（可为 None）

        Returns:
            有效的会话 ID
        """
        if session_id and session_id in self.sessions:
            self.sessions[session_id].update_active()
            return session_id
        return self.create_session()

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
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False

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
