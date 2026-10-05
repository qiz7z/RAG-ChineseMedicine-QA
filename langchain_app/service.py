# -*- coding: utf-8 -*-
"""
服务层（标准版）
================
面向 API 的 ChatService：
  - answer()          走声明式主链（chains.build_chat_chain）
  - answer_stream()   复用同一组构建块的流式路径（SSE 事件格式与主项目一致）
  - search()          独立检索（支持手动药品过滤）
  - 会话管理 / 统计

流式与同步两条路径共用同一批构建块（check_guard / resolve_question / postprocess），
且**守卫与会话历史都收口在本类**（`_guard` / `_ensure_history`），保证两条路径行为一致：

  - 守卫只做一次、含会话级信任通道——链里不再有守卫（缺陷 23）
  - 会话历史统一经 `_ensure_history` 建立——流式路径不再丢多轮上下文（缺陷 24）
"""
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from llm import build_chat_model
from retrievers import build_hybrid_retriever
from chains import build_chat_chain, check_guard, resolve_question
from prompts import RAG_PROMPT, GUARD_REJECTION, NO_RESULT_ANSWER
from postprocess import postprocess, docs_to_sources
from query_understanding import QueryAnalyzer

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, retriever=None, llm=None):
        self.retriever = retriever or build_hybrid_retriever()
        self.llm = llm or build_chat_model()
        # 声明式主链（同步问答走这条链）
        self.chain = build_chat_chain(retriever=self.retriever, llm=self.llm)
        self._histories = self.chain._histories
        # 守卫信任态：按会话隔离（首轮通过后，追问跳过 LLM 语义判定）
        self._guard_trusted: set = set()
        self._stats_cache: Optional[dict] = None

    # ----------------------------------------------------------
    # 两条路径共用的构建块（守卫 / 会话历史）
    # ----------------------------------------------------------

    def _ensure_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """取会话历史，不存在则建立。

        同步路径的历史由 `RunnableWithMessageHistory` 自动创建；流式路径没有它，
        早先直接 `if history:` 判断 —— 会话若未预先创建（客户端不调 /sessions 直接
        调 /chat/stream），历史永远不写、多轮上下文失效（缺陷 24）。两条路径统一走这里。
        """
        if session_id not in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        return self._histories[session_id]

    def _guard(self, question: str, session_id: str, is_followup: bool) -> bool:
        """领域守卫（**唯一入口**，同步与流式共用）。

        信任通道：该会话此前已通过守卫时，追问场景跳过 LLM 判定（省一次调用）；
        但"明显无关"仍会被词表拒掉。判据见 `chains.check_guard(trusted=...)`。

        早先这段逻辑在 `answer()` 与 `answer_stream()` 里各写一份，且链里还有第三份
        （`_guard_step`，不认信任通道）——同步路径因此守卫跑两遍（缺陷 23）。
        """
        trusted = is_followup and session_id in self._guard_trusted
        ok = check_guard(self.llm, question, trusted=trusted)
        if ok:
            self._guard_trusted.add(session_id)
        return ok

    # ----------------------------------------------------------
    # 同步问答（声明式主链）
    # ----------------------------------------------------------

    def answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        session_id = session_id or f"lcsess_{time.time_ns()}"
        history = self._histories.get(session_id)
        chat_history = history.messages if history else []

        t0 = time.time()

        # ---- 守卫（唯一入口；信任态与流式路径共用同一实现）----
        if not self._guard(question, session_id, is_followup=len(chat_history) > 0):
            return {
                "answer": GUARD_REJECTION,
                "citations": [],
                "consistency_issues": [],
                "sources": [],
                "resolved_query": question,
                "guard_rejected": True,
                "session_id": session_id,
                "latency": time.time() - t0,
                "dialogue_turn": len(chat_history) // 2,
            }

        # ---- 走声明式主链（历史注入由 RunnableWithMessageHistory 完成）----
        # 链只做"问题 → 答案"，守卫与拒答已在上面处理完（缺陷 23）
        out = self.chain.invoke(
            {"question": question, "chat_history": chat_history},
            config={"configurable": {"session_id": session_id}},
        )

        # 用户消息与 AI 回复由 RunnableWithMessageHistory 按
        # input/output_messages_key 自动写入历史，这里不再手动追加
        out["session_id"] = session_id
        out["latency"] = time.time() - t0
        history = self._ensure_history(session_id)
        out["dialogue_turn"] = len(history.messages) // 2
        return out

    # ----------------------------------------------------------
    # 流式问答（SSE 事件格式与主项目一致）
    # ----------------------------------------------------------

    def answer_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        session_id = session_id or f"lcsess_{time.time_ns()}"
        history = self._histories.get(session_id)
        chat_history = history.messages if history else []
        total_start = time.time()

        # ---- 守卫（与同步路径同一实现）----
        if not self._guard(question, session_id, is_followup=len(chat_history) > 0):
            # 带上真实 session_id：客户端不传 id 时也能拿到服务端生成的会话号（缺陷 24）
            yield {"rejected": True, "content": GUARD_REJECTION, "session_id": session_id}
            return

        # ---- 指代消解 ----
        resolved = resolve_question(self.llm, question, chat_history)

        # ---- 检索 ----
        docs = self.retriever.invoke(resolved)
        if not docs:
            # 与同步路径保持一致：这一轮也要进历史（同步路径由 RunnableWithMessageHistory 写入）
            history = self._ensure_history(session_id)
            history.add_message(HumanMessage(content=question))
            history.add_message(AIMessage(content=NO_RESULT_ANSWER))
            yield {"content": NO_RESULT_ANSWER}
            yield {"metadata": {
                "session_id": session_id,
                "resolved_query": resolved, "sources": [], "citations": [],
                "consistency_issues": [],
                "latency_ms": int((time.time() - total_start) * 1000),
                "dialogue_turn": len(history.messages) // 2,
            }}
            return

        # ---- 流式生成 ----
        model = self.llm
        if temperature is not None or max_tokens is not None:
            overrides = {}
            if temperature is not None:
                overrides["temperature"] = temperature
            if max_tokens is not None:
                overrides["max_tokens"] = max_tokens
            model = model.bind(**overrides)

        from postprocess import format_docs
        chain = RAG_PROMPT | model | StrOutputParser()
        raw = ""
        for chunk in chain.stream(
            {
                "context": format_docs(docs),
                "question": question,
                "chat_history": chat_history,
            }
        ):
            raw += chunk
            yield {"content": chunk}

        # ---- 后处理 + 历史更新 ----
        pp = postprocess(raw, docs)
        # 统一经 _ensure_history：会话未预先创建时也要能记住这一轮（缺陷 24）
        history = self._ensure_history(session_id)
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=pp["answer"]))

        yield {"metadata": {
            "session_id": session_id,
            "resolved_query": resolved,
            "sources": docs_to_sources(docs),
            "citations": pp["citations"],
            "consistency_issues": pp["consistency_issues"],
            "latency_ms": int((time.time() - total_start) * 1000),
            "dialogue_turn": len(history.messages) // 2,
        }}

    # ----------------------------------------------------------
    # 独立检索
    # ----------------------------------------------------------

    def search(self, query: str, top_k: int = 5, drug_filter: Optional[str] = None) -> List[dict]:
        docs = self.retriever.invoke(query, drug_filter=drug_filter)
        return docs_to_sources(docs)[:top_k]

    # ----------------------------------------------------------
    # 会话管理
    # ----------------------------------------------------------

    def create_session(self) -> str:
        sid = f"lcsess_{time.time_ns()}"
        self._histories[sid] = InMemoryChatMessageHistory()
        return sid

    def list_sessions(self) -> List[str]:
        return list(self._histories.keys())

    def session_info(self, session_id: str) -> Optional[dict]:
        history = self._histories.get(session_id)
        if not history:
            return None
        return {
            "session_id": session_id,
            "turn_count": len(history.messages) // 2,
            "history": [
                {"role": "user" if m.type == "human" else "assistant",
                 "content": m.content[:200]}
                for m in history.messages
            ],
        }

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._histories:
            del self._histories[session_id]
            self._guard_trusted.discard(session_id)
            return True
        return False

    # ----------------------------------------------------------
    # 统计（从 FAISS docstore 元数据汇总，启动时缓存一次）
    # ----------------------------------------------------------

    def stats(self) -> dict:
        if self._stats_cache is None:
            by_category: Dict[str, int] = {}
            by_chunk_type: Dict[str, int] = {}
            drugs = set()
            for doc in self.retriever.vectorstore.docstore._dict.values():
                m = doc.metadata
                by_category[m.get("category", "")] = by_category.get(m.get("category", ""), 0) + 1
                by_chunk_type[m.get("chunk_type", "")] = by_chunk_type.get(m.get("chunk_type", ""), 0) + 1
                if m.get("drug_name"):
                    drugs.add(m["drug_name"])
            self._stats_cache = {
                "total_chunks": self.retriever.vectorstore.index.ntotal,
                "total_drugs": len(drugs),
                "by_category": by_category,
                "by_chunk_type": by_chunk_type,
            }
        return self._stats_cache
