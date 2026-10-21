# -*- coding: utf-8 -*-
"""会话摘要（P1）测试：解析、格式化、节流与落库恢复。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory.store import MemoryStore                        # noqa: E402
from memory.summarizer import parse_summary, format_summary  # noqa: E402
from generation.dialogue import DialogueManager             # noqa: E402


class _FakeLLM:
    """只回固定文本的假模型，避免测试触发真实 API"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return self.reply


VALID = '{"focus":"人参 · 性味与归经","asked_sections":["性味与归经","用法与用量"],"open_questions":[]}'


def test_parse_summary_plain_json():
    data = parse_summary(VALID)
    assert data["focus"] == "人参 · 性味与归经"
    assert data["asked_sections"] == ["性味与归经", "用法与用量"]


def test_parse_summary_with_code_fence():
    data = parse_summary(f"```json\n{VALID}\n```")
    assert data and data["focus"].startswith("人参")


def test_parse_summary_rejects_garbage():
    assert parse_summary("抱歉，我无法总结") is None
    assert parse_summary("") is None
    assert parse_summary('{"focus":"","asked_sections":[],"open_questions":[]}') is None


def test_format_summary():
    text = format_summary({
        "focus": "人参 · 性味与归经",
        "asked_sections": ["性味与归经", "用法与用量"],
        "open_questions": ["含量测定限度未收录"],
    })
    assert "当前关注：人参 · 性味与归经" in text
    assert "已问章节：性味与归经、用法与用量" in text
    assert "未解决：含量测定限度未收录" in text
    assert format_summary(None) == ""


def test_summary_persisted_and_restored(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    dm = DialogueManager(session_id="s1", store=store)
    dm.llm_client = _FakeLLM(VALID)

    dm.add_user_message("人参的性味归经", "人参的性味归经")
    dm.add_assistant_message("甘、微苦，微温。")
    dm._refresh_summary(list(dm.history), "s1")       # 直接同步执行，避免等线程

    assert "人参" in dm.get_summary_text()

    dm2 = DialogueManager(session_id="s1", store=store)
    assert "人参" in dm2.get_summary_text(), "重启后摘要应随会话恢复"


def test_build_prompt_includes_summary():
    from generation.prompts import build_prompt

    msgs = build_prompt(
        "人参的用法用量",
        [{"drug_name": "人参", "section": "性味与归经", "content": "甘、微苦"}],
        memory_summary="当前关注：人参 · 性味与归经",
    )
    content = msgs[-1]["content"]
    assert "用户记忆" in content
    assert "人参 · 性味与归经" in content
    assert content.index("用户记忆") < content.index("药典参考资料")


def test_total_turns_survives_history_trim(tmp_path):
    """超过 max_history 后累计轮数仍增长——否则摘要永远不再刷新"""
    store = MemoryStore(str(tmp_path / "mem.db"))
    dm = DialogueManager(session_id="s5", max_history=2, store=store)
    dm.llm_client = _FakeLLM(VALID)
    dm._refresh_summary = lambda *a, **k: None       # 只验节流，不跑线程

    for i in range(6):
        dm.add_user_message(f"q{i}")
        dm.add_assistant_message(f"a{i}")

    assert dm.turn_count == 2, "内存历史按 max_history 裁剪"
    assert dm._total_turns == 6, "累计轮数不应被裁剪"
    assert dm._last_summary_turn >= 2, "摘要应持续被触发"


def test_summary_retries_immediately_when_missing(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    dm = DialogueManager(session_id="s6", store=store)
    dm.llm_client = _FakeLLM(VALID)
    dm._refresh_summary = lambda *a, **k: None       # 模拟失败，summary 保持空

    for i in range(3):
        dm.add_user_message(f"q{i}")
        dm.add_assistant_message(f"a{i}")
    assert dm._last_summary_turn == 3
    assert dm.summary is None

    dm.add_user_message("q3")
    dm.add_assistant_message("a3")
    assert dm._last_summary_turn == 4, "没有摘要时不应等间隔，应立刻重试"


def test_coreference_prompt_includes_summary():
    """超长对话中，只存在于摘要里的药品也要能用于指代消解"""
    from generation.prompts import build_coreference_prompt

    msgs = build_coreference_prompt(
        "那它的禁忌呢？",
        [{"role": "user", "content": "黄芪的性味"}],
        "当前关注：人参 · 性味与归经",
    )
    content = msgs[-1]["content"]
    assert "人参 · 性味与归经" in content
    assert "黄芪的性味" in content


def test_schedule_throttled_by_turn_count(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    dm = DialogueManager(session_id="s2", store=store)
    dm.llm_client = _FakeLLM(VALID)

    # 不足 MIN_TURNS（3 轮）不应触发
    dm.add_user_message("q1")
    dm.add_assistant_message("a1")
    assert dm._last_summary_turn == 0

    dm.add_user_message("q2")
    dm.add_assistant_message("a2")
    dm.add_user_message("q3")
    dm.add_assistant_message("a3")
    assert dm._last_summary_turn == 3, "达到轮次阈值后应记录触发点"

    # 间隔不足 INTERVAL（2 轮）不再触发
    dm.add_user_message("q4")
    dm.add_assistant_message("a4")
    assert dm._last_summary_turn == 3
