# -*- coding: utf-8 -*-
"""
记忆持久化（P0：会话与对话轮次落库）
=====================================

背景：原先 DialogueManager 的 history 只存在进程内存里，服务重启、
多 worker 部署都会丢失上下文。这里把「会话」与「每轮对话」落到
metadata.db（与切片元数据共用一个 SQLite 文件，但两张独立的表）。

暂不涉及长期记忆（摘要槽 / 事实库），那是 P1/P2。

表结构：
    sessions        会话一行，记录 user、创建时间、最后活跃时间
    dialogue_turns  每轮一行（user / assistant 各一行），resolved_query
                    保存指代消解后的查询，便于回溯「那它的用法呢」
                    到底被还原成了什么

设计约束（与 DeerFlow 对齐的三条）：
    1. 失败不影响主流程——所有写入异常只记日志，不抛给上层；
    2. 软删除——会话只标记 status，不物理删除，便于排查；
    3. 只存文本，不存检索结果/表格，控制体积。
"""
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def _now() -> str:
    """统一时间格式，SQLite 里按字符串比较即等价于按时间比较"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class MemoryStore:
    """会话与对话轮次的 SQLite 存储。

    与 MetadataStore 共用 metadata.db，但独立连接（避免互相干扰事务）。
    """

    def __init__(self, db_path: str = None):
        from config import SQLITE_DB_PATH
        self.db_path = db_path or SQLITE_DB_PATH
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    # ----------------------------------------------------------
    # 建表
    # ----------------------------------------------------------

    def _init_schema(self):
        with self._lock:
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL DEFAULT 'default',
                    created_at   TEXT NOT NULL,
                    last_active  TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'active',
                    summary_json TEXT
                );

                CREATE TABLE IF NOT EXISTS dialogue_turns (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id     TEXT NOT NULL,
                    role           TEXT NOT NULL,
                    content        TEXT NOT NULL,
                    resolved_query TEXT,
                    created_at     TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON dialogue_turns(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_sessions_active
                    ON sessions(status, last_active);
            ''')
            self.conn.commit()

    # ----------------------------------------------------------
    # 会话
    # ----------------------------------------------------------

    def touch_session(self, session_id: str, user_id: str = 'default') -> None:
        """会话存在则刷新活跃时间，不存在则创建"""
        if not session_id:
            return
        now = _now()
        try:
            with self._lock:
                cur = self.conn.execute(
                    'UPDATE sessions SET last_active = ? WHERE session_id = ?',
                    (now, session_id),
                )
                if cur.rowcount == 0:
                    self.conn.execute(
                        'INSERT INTO sessions (session_id, user_id, created_at, last_active) '
                        'VALUES (?, ?, ?, ?)',
                        (session_id, user_id, now, now),
                    )
                self.conn.commit()
        except Exception as e:
            logger.warning(f"刷新会话失败（不影响对话）: {e}")

    def session_exists(self, session_id: str, user_id: str = None) -> bool:
        """会话是否存在；传 user_id 时额外校验归属（防跨用户复用 session_id）"""
        sql = 'SELECT 1 FROM sessions WHERE session_id = ? AND status = ?'
        params = [session_id, 'active']
        if user_id is not None:
            sql += ' AND user_id = ?'
            params.append(user_id)
        return self.conn.execute(sql, params).fetchone() is not None

    def get_session_user_id(self, session_id: str) -> Optional[str]:
        """会话归属的用户；不存在返回 None"""
        row = self.conn.execute(
            'SELECT user_id FROM sessions WHERE session_id = ?', (session_id,)
        ).fetchone()
        return row['user_id'] if row else None

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        rows = self.conn.execute(
            'SELECT s.session_id, s.user_id, s.created_at, s.last_active, '
            '       (SELECT COUNT(*) FROM dialogue_turns t WHERE t.session_id = s.session_id) AS turns '
            'FROM sessions s WHERE s.status = ? ORDER BY s.last_active DESC LIMIT ?',
            ('active', limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> int:
        """软删除：标记 status，轮次保留可查"""
        with self._lock:
            cur = self.conn.execute(
                'UPDATE sessions SET status = ? WHERE session_id = ?',
                ('deleted', session_id),
            )
            self.conn.commit()
            return cur.rowcount

    def cleanup_expired(self, timeout_hours: int = 24) -> int:
        """超过 timeout_hours 未活跃的会话标记为 expired"""
        deadline = (datetime.now() - timedelta(hours=timeout_hours)).strftime('%Y-%m-%d %H:%M:%S')
        with self._lock:
            cur = self.conn.execute(
                'UPDATE sessions SET status = ? WHERE status = ? AND last_active < ?',
                ('expired', 'active', deadline),
            )
            self.conn.commit()
            return cur.rowcount

    def save_summary(self, session_id: str, summary: Dict) -> None:
        """P1 预留：三槽摘要落库（当前不使用）"""
        try:
            with self._lock:
                self.conn.execute(
                    'UPDATE sessions SET summary_json = ? WHERE session_id = ?',
                    (json.dumps(summary, ensure_ascii=False), session_id),
                )
                self.conn.commit()
        except Exception as e:
            logger.warning(f"保存会话摘要失败: {e}")

    def load_summary(self, session_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            'SELECT summary_json FROM sessions WHERE session_id = ?', (session_id,)
        ).fetchone()
        if not row or not row['summary_json']:
            return None
        try:
            return json.loads(row['summary_json'])
        except (ValueError, TypeError):
            return None

    # ----------------------------------------------------------
    # 对话轮次
    # ----------------------------------------------------------

    def append_turn(self, session_id: str, role: str, content: str,
                    resolved_query: str = '') -> Optional[int]:
        """追加一轮对话；失败返回 None，不抛异常"""
        if not session_id or not content:
            return None
        try:
            with self._lock:
                cur = self.conn.execute(
                    'INSERT INTO dialogue_turns (session_id, role, content, resolved_query, created_at) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (session_id, role, content, resolved_query or None, _now()),
                )
                self.conn.commit()
                return cur.lastrowid
        except Exception as e:
            logger.warning(f"写入对话轮次失败（不影响对话）: {e}")
            return None

    def load_turns(self, session_id: str, max_turns: int = 5) -> List[Dict]:
        """载入最近 max_turns 轮（1 轮 = user + assistant 共 2 条）"""
        if not session_id:
            return []
        limit = max(1, max_turns) * 2
        rows = self.conn.execute(
            'SELECT role, content, resolved_query FROM dialogue_turns '
            'WHERE session_id = ? ORDER BY id DESC LIMIT ?',
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def count_turns(self, session_id: str) -> int:
        row = self.conn.execute(
            'SELECT COUNT(*) AS n FROM dialogue_turns WHERE session_id = ?', (session_id,)
        ).fetchone()
        return int(row['n']) if row else 0

    def clear_turns(self, session_id: str) -> int:
        with self._lock:
            cur = self.conn.execute(
                'DELETE FROM dialogue_turns WHERE session_id = ?', (session_id,)
            )
            self.conn.commit()
            return cur.rowcount

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


_default_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """全局单例（延迟初始化），避免每个 DialogueManager 各开一个连接"""
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store
