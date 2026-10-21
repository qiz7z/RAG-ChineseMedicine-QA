# -*- coding: utf-8 -*-
"""短期记忆的焦点槽：指代消解应确定性跟随「用户最近提到的药」。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory.store import MemoryStore        # noqa: E402
from generation.dialogue import DialogueManager  # noqa: E402

DRUGS = ["人参", "人参叶", "黄芪", "当归"]


class _CountingLLM:
    """记录调用次数，用来证明焦点槽路径不依赖模型"""

    def __init__(self, reply: str = "兜底药"):
        self.reply = reply
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return self.reply


def _dm(tmp_path, name: str = "s") -> DialogueManager:
    return DialogueManager(
        session_id=name,
        store=MemoryStore(str(tmp_path / "m.db")),
        drug_names=DRUGS,
    )


def test_focus_set_from_user_message_and_reused(tmp_path):
    dm = _dm(tmp_path)
    dm.llm_client = _CountingLLM()

    dm.add_user_message("人参的性味归经是什么？", "人参的性味归经是什么？")
    dm.add_assistant_message("甘、微苦，微温。")
    assert dm.current_drug == "人参"

    assert dm.resolve_query("它能补血吗？") == "人参能补血吗？"
    assert dm.resolve_query("那它的禁忌呢？") == "那人参的禁忌呢？"
    assert dm.llm_client.calls == 0, "焦点槽命中时不应调用模型"


def test_focus_switches_to_latest_drug(tmp_path):
    dm = _dm(tmp_path)
    dm.llm_client = _CountingLLM()

    dm.add_user_message("人参的性味归经", "人参的性味归经")
    dm.add_assistant_message("…")
    dm.add_user_message("黄芪的功能主治", "黄芪的功能主治")
    dm.add_assistant_message("…")

    assert dm.current_drug == "黄芪"
    assert dm.resolve_query("它能补气吗？") == "黄芪能补气吗？"


def test_query_with_drug_updates_focus_without_resolution(tmp_path):
    dm = _dm(tmp_path)
    dm.llm_client = _CountingLLM()
    dm.add_user_message("人参的性味", "人参的性味")
    dm.add_assistant_message("…")

    text = "当归的用法用量是什么？"
    assert dm.resolve_query(text) == text, "自带药名的问题不应被改写"
    assert dm.current_drug == "当归"


def test_longest_drug_name_wins(tmp_path):
    dm = _dm(tmp_path)
    dm.llm_client = _CountingLLM()
    dm.add_user_message("人参叶的性味", "人参叶的性味")
    assert dm.current_drug == "人参叶", "应优先匹配更长的药名"


def test_other_is_not_coreference(tmp_path):
    dm = _dm(tmp_path)
    assert dm._has_coreference("其他药材有什么禁忌") is False
    assert dm._has_coreference("其它功效呢") is False
    assert dm._has_coreference("它有什么禁忌") is True


def test_fallback_to_llm_when_focus_empty(tmp_path):
    dm = _dm(tmp_path)
    llm = _CountingLLM("兜底药")
    dm.llm_client = llm

    dm.add_user_message("为什么要这样配伍？", "人参为什么要这样配伍？")
    dm.add_assistant_message("…")
    dm.current_drug = None            # 模拟焦点丢失

    resolved = dm.resolve_query("它有什么禁忌？")
    assert llm.calls == 1, "焦点为空时应回退模型"
    assert "兜底药" in resolved


def test_focus_restored_from_history(tmp_path):
    store = MemoryStore(str(tmp_path / "m.db"))
    dm1 = DialogueManager(session_id="s9", store=store, drug_names=DRUGS)
    dm1.add_user_message("黄芪的性味", "黄芪的性味")
    dm1.add_assistant_message("…")

    dm2 = DialogueManager(session_id="s9", store=store, drug_names=DRUGS)
    assert dm2.current_drug == "黄芪", "重启恢复后焦点应随之恢复"


# ----------------------------------------------------------
# 表外实体：药典没有的名字（如「黄麻」——药典只有「麻黄」）也要能成为焦点
# ----------------------------------------------------------

def test_off_catalog_entity_becomes_focus(tmp_path):
    """用户场景：人参问了几轮后切到「黄麻」，"它"必须指黄麻，不是人参"""
    dm = _dm(tmp_path)
    dm.llm_client = _CountingLLM()
    dm._refresh_summary = lambda *a, **k: None   # 关掉后台摘要，避免污染调用计数

    dm.add_user_message("人参的性味归经是什么？", "人参的性味归经是什么？")
    dm.add_assistant_message("甘、微苦，微温。")
    dm.add_user_message("它能补血吗？", "人参能补血吗？")
    dm.add_assistant_message("…")
    dm.add_user_message("人参一般用多少克？", "人参一般用多少克？")
    dm.add_assistant_message("…")

    # 切到药典外的「黄麻」
    dm.add_user_message("黄麻什么口味？", "黄麻什么口味？")
    assert dm.current_drug == "黄麻", "表外实体应能成为焦点"

    assert dm.resolve_query("它能补血吗？") == "黄麻能补血吗？"
    assert dm.resolve_query("那它的禁忌呢？") == "那黄麻的禁忌呢？"
    assert dm.llm_client.calls == 0, "焦点槽命中时不应调用模型"


def test_off_catalog_entity_variants(tmp_path):
    dm = _dm(tmp_path)
    dm.llm_client = _CountingLLM()

    for text, expected in [
        ("黄麻叶怎么吃？", "黄麻叶"),       # 相邻名词合并
        ("那个黄麻什么口味？", "黄麻"),      # 跳过句首代词
        ("请问黄麻什么口味？", "黄麻"),      # 剥掉客套前缀
        ("3克黄麻怎么吃？", "黄麻"),        # 跳过数量词
    ]:
        assert dm._extract_topic_entity(text) == expected, text


def test_continuation_question_keeps_focus(tmp_path):
    """延续问句（主语是属性/人群词）不许抢焦点"""
    dm = _dm(tmp_path)
    dm.llm_client = _CountingLLM()
    dm.add_user_message("黄麻什么口味？", "黄麻什么口味？")
    dm.add_assistant_message("…")
    assert dm.current_drug == "黄麻"

    for text in [
        "孕妇能吃吗？", "药效怎么样？", "味道怎么样？", "保存方法？",
        "补血吗？", "一次吃多少？", "口服液说明书？", "适合孕妇吗？",
        "其他药材有什么禁忌？",
    ]:
        assert dm._extract_topic_entity(text) is None, text
        dm.resolve_query(text)          # 不改写
        assert dm.current_drug == "黄麻", f"'{text}' 不应改变焦点"

    # 延续问句之后，"它"仍指黄麻
    assert dm.resolve_query("它有毒吗？") == "黄麻有毒吗？"


def test_off_catalog_focus_restored_from_history(tmp_path):
    """重启后表外实体的焦点也要恢复"""
    store = MemoryStore(str(tmp_path / "m.db"))
    dm1 = DialogueManager(session_id="s10", store=store, drug_names=DRUGS)
    dm1.add_user_message("黄麻什么口味？", "黄麻什么口味？")
    dm1.add_assistant_message("…")

    dm2 = DialogueManager(session_id="s10", store=store, drug_names=DRUGS)
    assert dm2.current_drug == "黄麻"
    assert dm2.resolve_query("它能补血吗？") == "黄麻能补血吗？"
