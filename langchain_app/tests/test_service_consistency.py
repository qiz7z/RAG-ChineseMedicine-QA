# -*- coding: utf-8 -*-
"""service 两条路径（同步 / 流式）行为一致性回归测试

对应两个已修缺陷（见 docs/13）：
  缺陷 23：守卫在同步路径跑两遍 —— service 里一遍，链里 `_guard_step` 又一遍，
           且链里那遍不认"会话信任通道"。
  缺陷 24：流式路径的历史靠 `if history:` 写 —— 会话未预先创建时（客户端不调
           /sessions 直接调 /chat/stream）历史永不写入、多轮失效；返回给客户端的
           session_id 还是请求里的（可能为空串），客户端拿不到会话号。

用假检索器 + 假模型驱动：不加载真索引、不调真 API、不产生 token 消耗。
"""
import sys
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from chains import build_chat_chain
from guard import keyword_guard
from prompts import GUARD_REJECTION, NO_RESULT_ANSWER
from service import ChatService

AMBIGUOUS = "帮我写一首诗"        # 词表判不出 → 需 LLM 判定
OFF_TOPIC = "今天天气怎么样？"    # 词表直接拒
ON_TOPIC = "人参的性味归经是什么？"


class CountingFakeLLM(GenericFakeChatModel):
    """假模型 + 调用计数（用于证明"守卫只跑一遍"）"""

    counter: ClassVar[dict] = {"n": 0}

    @classmethod
    def reset(cls):
        cls.counter["n"] = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        CountingFakeLLM.counter["n"] += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _fake_llm(text: str = "人参味甘、微苦，微温，归脾、肺经。"):
    return CountingFakeLLM(messages=iter([AIMessage(content=text)] * 50))


def _doc(drug="人参", section="性味与归经", content="味甘、微苦，微温。"):
    return Document(page_content=content, metadata={
        "chunk_id": "c1", "drug_name": drug, "section": section,
        "category": "药材和饮片", "chunk_id_src": None, "score": 0.9,
    })


class FakeRetriever:
    """只实现 invoke；docs=[] 可模拟"检索未覆盖" """

    def __init__(self, docs=None):
        self.docs = [_doc()] if docs is None else docs
        self.queries = []

    def invoke(self, query, drug_filter=None):
        self.queries.append(query)
        return list(self.docs)


def _service(docs=None):
    CountingFakeLLM.reset()
    retriever = FakeRetriever(docs)
    return ChatService(retriever=retriever, llm=_fake_llm()), retriever


class TestGuardSingleEntry:
    """守卫只在一个地方（service._guard），且带会话信任通道"""

    def test_ambiguous_needs_llm_then_trusted_skips(self):
        svc, _ = _service()
        assert keyword_guard(AMBIGUOUS)[1] == "ambiguous"      # 测试前提
        assert svc._guard(AMBIGUOUS, "s1", is_followup=False) is True
        assert CountingFakeLLM.counter["n"] == 1               # 调了一次 LLM 判定
        assert svc._guard(AMBIGUOUS, "s1", is_followup=True) is True
        assert CountingFakeLLM.counter["n"] == 1               # 信任通道：不再调 LLM

    def test_trusted_still_rejects_off_topic(self):
        """信任通道 ≠ 放行一切：明显无关的问题仍被词表拒掉"""
        svc, _ = _service()
        svc._guard(AMBIGUOUS, "s1", is_followup=False)          # 先让该会话建立信任
        assert svc._guard(OFF_TOPIC, "s1", is_followup=True) is False

    def test_on_topic_needs_no_llm(self):
        svc, _ = _service()
        assert svc._guard(ON_TOPIC, "s1", is_followup=False) is True
        assert CountingFakeLLM.counter["n"] == 0

    def test_sync_path_guards_exactly_once(self):
        """同步路径的 LLM 调用应为 2 次：守卫 1 次 + 生成 1 次。

        缺陷 23 之前是 3 次（链里 `_guard_step` 又判一遍，且不认信任通道）。
        """
        svc, _ = _service()
        out = svc.answer(AMBIGUOUS)
        assert out["answer"]
        assert CountingFakeLLM.counter["n"] == 2, (
            f"共 {CountingFakeLLM.counter['n']} 次 LLM 调用（1 守卫 + 1 生成 = 2 才对）"
        )

    def test_chain_itself_has_no_guard(self):
        """链是纯流程：直接调链时，无关问题也不由链拒答（守卫已归 service）"""
        chain = build_chat_chain(retriever=FakeRetriever(), llm=_fake_llm())
        out = chain.invoke(
            {"question": OFF_TOPIC},
            config={"configurable": {"session_id": "x"}},
        )
        assert out["answer"] != GUARD_REJECTION
        assert "guard_rejected" not in out


class TestStreamSessionConsistency:
    """流式路径的会话与历史，必须与同步路径一致"""

    def test_stream_creates_session_and_returns_id(self):
        """不传 session_id 时，服务端生成的会话号要回传，且历史要建立"""
        svc, _ = _service()
        events = list(svc.answer_stream(ON_TOPIC))
        meta = events[-1]["metadata"]
        assert meta["session_id"]
        assert len(svc._histories[meta["session_id"]].messages) == 2

    def test_stream_remembers_multi_turn(self):
        """缺陷 24：此前这里 dialogue_turn 恒为 0、历史永不写入"""
        svc, _ = _service()
        sid = list(svc.answer_stream(ON_TOPIC))[-1]["metadata"]["session_id"]
        meta = list(svc.answer_stream("它有什么副作用？", session_id=sid))[-1]["metadata"]
        assert meta["dialogue_turn"] == 2
        assert len(svc._histories[sid].messages) == 4

    def test_stream_no_result_also_records_turn(self):
        """检索为空时也要记这一轮（与同步路径一致）"""
        svc, _ = _service(docs=[])
        events = list(svc.answer_stream(ON_TOPIC))
        assert any(e.get("content") == NO_RESULT_ANSWER for e in events)
        sid = events[-1]["metadata"]["session_id"]
        assert len(svc._histories[sid].messages) == 2

    def test_stream_rejection_carries_session_id(self):
        svc, _ = _service()
        events = list(svc.answer_stream(OFF_TOPIC))
        assert events[0]["rejected"] is True
        assert events[0]["session_id"]

    def test_both_paths_leave_history(self):
        svc1, _ = _service()
        out = svc1.answer(ON_TOPIC)
        assert len(svc1._histories[out["session_id"]].messages) == 2

        svc2, _ = _service()
        ev = list(svc2.answer_stream(ON_TOPIC))
        assert len(svc2._histories[ev[-1]["metadata"]["session_id"]].messages) == 2


class TestStreamApiContract:
    """SSE 事件契约：done 事件必须回传**服务端**的 session_id（缺陷 24 的 API 侧）"""

    def test_done_event_carries_server_side_session_id(self, monkeypatch):
        from fastapi.testclient import TestClient

        import api as api_module

        class _FakeService:
            def answer_stream(self, question, session_id=None, temperature=None, max_tokens=None):
                yield {"content": "人参味甘。"}
                yield {"metadata": {"session_id": "srv-1", "latency_ms": 1, "dialogue_turn": 1}}

        monkeypatch.setattr(api_module, "get_service", lambda: _FakeService())
        client = TestClient(api_module.app)

        resp = client.post("/api/v1/chat/stream", json={"question": "人参的性味"})
        assert resp.status_code == 200
        done = [ln for ln in resp.text.splitlines() if '"done"' in ln]
        assert done, resp.text
        assert '"session_id": "srv-1"' in done[-1]     # 不是请求里的空串

