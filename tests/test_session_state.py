# -*- coding: utf-8 -*-
"""会话状态隔离测试：守卫信任状态必须按会话独立。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import memory.store as store_mod                 # noqa: E402
from api.session import SessionManager           # noqa: E402
from memory.store import MemoryStore             # noqa: E402


def _with_temp_store(tmp_path):
    """把全局单例替换成临时库，避免污染 metadata.db"""
    saved = store_mod._default_store
    store_mod._default_store = MemoryStore(str(tmp_path / "memory.db"))
    return saved


def _restore(saved):
    if saved is None:
        store_mod._default_store = None
    else:
        store_mod._default_store = saved


def test_guard_is_per_session(tmp_path):
    saved = _with_temp_store(tmp_path)
    try:
        sm = SessionManager()
        s1 = sm.create_session()
        s2 = sm.create_session()

        g1 = sm.get_session(s1).guard
        g2 = sm.get_session(s2).guard
        assert g1 is not g2, "守卫实例不应被多个会话共享"

        ok, _ = g1.check("人参的性味归经")
        assert ok and g1._session_trusted
        assert not g2._session_trusted, "一个会话通过守卫不应影响另一个会话"
    finally:
        _restore(saved)


def test_guard_resets_after_restore(tmp_path):
    """服务重启恢复会话时，守卫应是全新实例（信任状态不持久化）"""
    saved = _with_temp_store(tmp_path)
    try:
        sm1 = SessionManager()
        sid = sm1.create_session()
        dm = sm1.get_dialogue_manager(sid)
        dm.add_user_message("人参的性味归经", "人参的性味归经")
        dm.add_assistant_message("甘、微苦，微温。")
        sm1.get_session(sid).guard.check("人参的性味归经")

        sm2 = SessionManager()          # 模拟重启
        assert sm2.get_or_create(sid) == sid
        session = sm2.get_session(sid)
        assert session is not None
        assert session.guard._session_trusted is False
    finally:
        _restore(saved)


def test_dialogue_manager_not_shared(tmp_path):
    saved = _with_temp_store(tmp_path)
    try:
        sm = SessionManager()
        s1 = sm.create_session()
        s2 = sm.create_session()
        assert sm.get_session(s1).dialogue_manager is not sm.get_session(s2).dialogue_manager

        dm1 = sm.get_dialogue_manager(s1)
        dm1.add_user_message("A")
        assert sm.get_dialogue_manager(s2).history == []
    finally:
        _restore(saved)
