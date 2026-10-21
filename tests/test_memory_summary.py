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
