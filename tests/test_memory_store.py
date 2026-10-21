# -*- coding: utf-8 -*-
"""记忆持久化（P0）回归测试：会话与对话轮次落库、按会话恢复。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory.store import MemoryStore          # noqa: E402
from generation.dialogue import DialogueManager  # noqa: E402


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memory.db"))


def test_append_and_load_turns(tmp_path):
    s = _store(tmp_path)
    s.touch_session("sess_1")
    s.append_turn("sess_1", "user", "人参的性味归经", "人参的性味归经")
    s.append_turn("sess_1", "assistant", "甘、微苦，微温。")

    rows = s.load_turns("sess_1", max_turns=5)
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["resolved_query"] == "人参的性味归经"
    assert s.count_turns("sess_1") == 2


def test_dialogue_restores_after_restart(tmp_path):
    """模拟重启：新建同 session_id 的 manager，历史应自动恢复"""
    s = _store(tmp_path)
    dm = DialogueManager(session_id="sess_a", store=s)
    dm.add_user_message("人参的性味归经", "人参的性味归经")
    dm.add_assistant_message("甘、微苦，微温。")

    dm2 = DialogueManager(session_id="sess_a", store=s)
    assert dm2.turn_count == 1
    assert dm2.history[0].content == "人参的性味归经"
    assert dm2.history[1].role == "assistant"


def test_sessions_are_isolated(tmp_path):
    s = _store(tmp_path)
    d1 = DialogueManager(session_id="s1", store=s)
    d1.add_user_message("A")
    d1.add_assistant_message("a")

    d2 = DialogueManager(session_id="s2", store=s)
    d2.add_user_message("B")
    d2.add_assistant_message("b")

    d3 = DialogueManager(session_id="s1", store=s)
    assert [t.content for t in d3.history] == ["A", "a"]


def test_bind_session_switches_history(tmp_path):
    s = _store(tmp_path)
    d1 = DialogueManager(session_id="s1", store=s)
    d1.add_user_message("A")
    d1.add_assistant_message("a")

    d2 = DialogueManager(session_id="s2", store=s)
    d2.add_user_message("B")
    d2.add_assistant_message("b")

    d2.bind_session("s1")
    assert d2.session_id == "s1"
    assert [t.content for t in d2.history] == ["A", "a"]


def test_memory_trim_does_not_delete_persisted(tmp_path):
    """内存只保留最近 max_history 轮，库里保留全部轮次"""
    s = _store(tmp_path)
    dm = DialogueManager(session_id="s3", max_history=2, store=s)
    for i in range(4):
        dm.add_user_message(f"q{i}")
        dm.add_assistant_message(f"a{i}")

    assert dm.turn_count == 2
    assert s.count_turns("s3") == 8


def test_clear_removes_persisted_turns(tmp_path):
    s = _store(tmp_path)
    dm = DialogueManager(session_id="s4", store=s)
    dm.add_user_message("A")
    dm.add_assistant_message("a")
    dm.clear()

    assert dm.history == []
    assert s.count_turns("s4") == 0


def test_session_lifecycle(tmp_path):
    s = _store(tmp_path)
    s.touch_session("sess_x")
    assert s.session_exists("sess_x")

    assert s.delete_session("sess_x") == 1
    assert not s.session_exists("sess_x")

    sessions = s.list_sessions()
    assert all(x["session_id"] != "sess_x" for x in sessions)


def test_list_sessions_title_and_user_filter(tmp_path):
    """会话列表：按用户过滤，title 取首条用户提问"""
    s = _store(tmp_path)
    s.touch_session("sess_a", "alice")
    s.append_turn("sess_a", "user", "人参的性味归经是什么")
    s.append_turn("sess_a", "assistant", "甘、微苦，微温。")
    s.touch_session("sess_b", "bob")
    s.append_turn("sess_b", "user", "黄芪怎么吃")

    alice = s.list_sessions(user_id="alice")
    assert [x["session_id"] for x in alice] == ["sess_a"]
    assert alice[0]["title"] == "人参的性味归经是什么"
    assert alice[0]["turns"] == 2

    all_sessions = s.list_sessions()
    assert {x["session_id"] for x in all_sessions} == {"sess_a", "sess_b"}

    # 空会话也进列表，title 为 None
    s.touch_session("sess_c", "alice")
    titles = {x["session_id"]: x["title"] for x in s.list_sessions(user_id="alice")}
    assert titles["sess_c"] is None


def test_session_manager_restores_by_id(tmp_path):
    """API 层：新进程拿到旧 session_id 时应恢复会话，而不是新建"""
    import memory.store as store_mod
    from api.session import SessionManager

    saved = store_mod._default_store
    store_mod._default_store = _store(tmp_path)
    try:
        sm1 = SessionManager()
        sid = sm1.create_session()
        dm = sm1.get_dialogue_manager(sid)
        dm.add_user_message("人参的性味归经", "人参的性味归经")
        dm.add_assistant_message("甘、微苦，微温。")

        # 模拟重启：新的 SessionManager，内存表为空
        sm2 = SessionManager()
        assert sm2.get_or_create(sid) == sid
        restored = sm2.get_dialogue_manager(sid)
        assert restored is not None
        assert restored.turn_count == 1
        assert restored.history[0].content == "人参的性味归经"
    finally:
        store_mod._default_store = saved
