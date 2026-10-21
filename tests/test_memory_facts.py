# -*- coding: utf-8 -*-
"""长期记忆（P2）用户事实库测试：抽取解析、合并纠错、保护位、容量淘汰、跨会话。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory.store import MemoryStore                    # noqa: E402
from memory.facts import (                              # noqa: E402
    has_fact_signal, parse_facts, format_facts, extract_facts,
)
from generation.dialogue import DialogueManager        # noqa: E402

FACTS_JSON = (
    '{"facts": [{"type": "allergy", "content": "对人参过敏", "remember": true}], '
    '"retract": ["对三七过敏"]}'
)


class _FakeLLM:
    """只回固定文本的假模型"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return self.reply


class _SyncExec:
    """同步执行器：submit 立即执行，消除后台线程竞态"""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return None


# ----------------------------------------------------------
# 关键词信号过滤
# ----------------------------------------------------------

def test_fact_signal_words():
    assert has_fact_signal("我对人参过敏")
    assert has_fact_signal("我正在吃华法林")
    assert has_fact_signal("是给6岁孩子问的")
    assert has_fact_signal("记住我对它过敏")
    assert not has_fact_signal("人参的性味归经是什么")
    assert not has_fact_signal("它能补血吗")
    assert not has_fact_signal("")


# ----------------------------------------------------------
# 解析与格式化
# ----------------------------------------------------------

def test_parse_facts_json():
    result = parse_facts(FACTS_JSON)
    assert result is not None
    assert result["facts"][0]["type"] == "allergy"
    assert result["facts"][0]["content"] == "对人参过敏"
    assert result["facts"][0]["remember"] is True
    assert result["retract"] == ["对三七过敏"]


def test_parse_facts_rejects_garbage():
    assert parse_facts("这不是JSON") is None
    assert parse_facts("") is None
    assert parse_facts('{"facts": [], "retract": []}') is None, "无产出应返回 None"
    assert parse_facts('{"facts": [{"content": ""}]}') is None, "空内容不算事实"


def test_format_facts():
    text = format_facts([
        {"fact_type": "allergy", "content": "对人参过敏"},
        {"type": "population", "content": "孕期"},
    ])
    assert "过敏史：对人参过敏" in text
    assert "特殊人群：孕期" in text
    assert format_facts([]) == ""


# ----------------------------------------------------------
# store 层：去重 / 保护位 / 淘汰
# ----------------------------------------------------------

def test_fact_upsert_dedup_and_protect():
    store = MemoryStore(":memory:") if False else MemoryStore(_tmp("f1"))
    store.upsert_fact("u1", "allergy", "对人参过敏")
    store.upsert_fact("u1", "allergy", "对人参过敏")   # 重复只刷新时间
    assert store.count_facts("u1") == 1

    store.upsert_fact("u1", "allergy", "对人参过敏", protected=True)
    facts = store.load_facts("u1")
    assert facts[0]["protected"] == 1, "保护位应被置上"

    # 保护位只升不降
    store.upsert_fact("u1", "allergy", "对人参过敏", protected=False)
    assert store.load_facts("u1")[0]["protected"] == 1

    # 不同用户互不影响
    store.upsert_fact("u2", "allergy", "对三七过敏")
    assert store.count_facts("u2") == 1


def test_fact_eviction_keeps_protected():
    store = MemoryStore(_tmp("f2"))
    store.upsert_fact("u1", "allergy", "重要事实", protected=True)
    for i in range(25):
        store.upsert_fact("u1", "other", f"普通事实{i}")

    removed = store.evict_facts("u1", keep=20)
    assert removed >= 5
    facts = store.load_facts("u1")
    assert len(facts) == 20
    contents = [f["content"] for f in facts]
    assert "重要事实" in contents, "保护位事实永不淘汰"


def test_fact_delete_bypasses_protect():
    store = MemoryStore(_tmp("f3"))
    store.upsert_fact("u1", "allergy", "对人参过敏", protected=True)
    assert store.delete_fact("u1", "对人参过敏") == 1
    assert store.count_facts("u1") == 0, "显式删除（用户纠错）不受保护位拦截"


# ----------------------------------------------------------
# DialogueManager 集成
# ----------------------------------------------------------

def _dm(tmp_path, user_id="u1", name="s1", reply=FACTS_JSON):
    dm = DialogueManager(
        session_id=name,
        store=MemoryStore(str(tmp_path / "m.db")),
        drug_names=["人参", "黄芪"],
        user_id=user_id,
    )
    dm.llm_client = _FakeLLM(reply)
    dm._mem_executor = _SyncExec()
    return dm


def test_facts_not_extracted_without_signal(tmp_path):
    dm = _dm(tmp_path)
    dm.add_user_message("人参的性味归经是什么？", "人参的性味归经是什么？")
    dm.add_assistant_message("甘、微苦。")
    assert dm.llm_client.calls == 0, "无事实信号词不应调 LLM"


def test_facts_extracted_and_injected(tmp_path):
    dm = _dm(tmp_path)
    dm.add_user_message("我对人参过敏", "我对人参过敏")
    dm.add_assistant_message("了解，请避免使用人参。")

    assert dm.llm_client.calls == 1
    text = dm.get_facts_text()
    assert "过敏史：对人参过敏" in text

    # 注入 prompt
    from generation.prompts import build_prompt
    msgs = build_prompt("黄芪能吃吗", [{"drug_name": "黄芪", "content": "甘"}],
                        user_facts=text)
    assert "用户情况（长期事实" in msgs[-1]["content"]
    assert "对人参过敏" in msgs[-1]["content"]


def test_fact_retract_removes_old(tmp_path):
    store = MemoryStore(str(tmp_path / "m.db"))
    store.upsert_fact("u1", "allergy", "对三七过敏", protected=True)

    dm = DialogueManager(session_id="s2", store=store,
                         drug_names=["人参"], user_id="u1")
    dm.llm_client = _FakeLLM(FACTS_JSON)   # retract: 对三七过敏；facts: 对人参过敏
    dm._mem_executor = _SyncExec()
    dm.add_user_message("说错了，我不过敏", "说错了，我不过敏")
    dm.add_assistant_message("好的，已更正。")

    contents = [f["content"] for f in store.load_facts("u1")]
    assert "对三七过敏" not in contents, "被否认的旧事实应删除（保护位也拦不住）"
    assert "对人参过敏" in contents


def test_facts_cross_session_same_user(tmp_path):
    """长期事实按 user_id 跨会话生效"""
    store = MemoryStore(str(tmp_path / "m.db"))
    dm1 = DialogueManager(session_id="sa", store=store,
                          drug_names=["人参"], user_id="u1")
    dm1.llm_client = _FakeLLM(FACTS_JSON)
    dm1._mem_executor = _SyncExec()
    dm1.add_user_message("我对人参过敏", "我对人参过敏")
    dm1.add_assistant_message("好的。")

    dm2 = DialogueManager(session_id="sb", store=store,
                          drug_names=["人参"], user_id="u1")
    assert "对人参过敏" in dm2.get_facts_text(), "新会话应看到用户的历史事实"

    dm3 = DialogueManager(session_id="sc", store=store,
                          drug_names=["人参"], user_id="u2")
    assert dm3.get_facts_text() == "", "不同用户事实隔离"


def test_extract_facts_failure_is_silent(tmp_path):
    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("boom")

    result = extract_facts(_Boom(), [{"role": "user", "content": "我对人参过敏"}])
    assert result is None


def _tmp(name: str) -> str:
    import tempfile, os
    d = tempfile.mkdtemp()
    return os.path.join(d, name + ".db")
