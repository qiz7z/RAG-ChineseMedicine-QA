# -*- coding: utf-8 -*-
"""
服务层（标准版）
================
面向 API 的 ChatService：
  - answer()          走声明式主链（chains.build_chat_chain）
  - answer_stream()   复用同一组构建块的流式路径（SSE 事件格式与主项目一致）
  - search()          独立检索（支持手动药品过滤）
  - 会话管理 / 统计

流式与同步两条路径共用 check_guard / resolve_question / postprocess 等构建块，
保证行为一致。
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
from guard import keyword_guard, llm_guard
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

        # ---- 守卫（信任态：已通过守卫的会话，追问不再走 LLM 判定）----
        is_followup = len(chat_history) > 0
        ok, reason = keyword_guard(question)
        if reason == "keyword_off":
            ok = False
        elif reason == "ambiguous" and not (is_followup and session_id in self._guard_trusted):
            ok = llm_guard(self.llm, question)
        if not ok:
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
        self._guard_trusted.add(session_id)

        # ---- 走声明式主链（历史注入由 RunnableWithMessageHistory 完成）----
        out = self.chain.invoke(
            {"question": question, "chat_history": chat_history, "guard_rejected": False},
            config={"configurable": {"session_id": session_id}},
        )

        # 历史已由 RunnableWithMessageHistory 自动写入用户消息；这里补写助手回复
        history = self._histories.get(session_id)
        if history:
            history.add_message(AIMessage(content=out["answer"]))

        out["session_id"] = session_id
        out["latency"] = time.time() - t0
        out["dialogue_turn"] = len(history.messages) // 2 if history else 0
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

        # ---- 守卫 ----
        is_followup = len(chat_history) > 0
        ok, reason = keyword_guard(question)
        if reason == "keyword_off":
            ok = False
        elif reason == "ambiguous" and not (is_followup and session_id in self._guard_trusted):
            ok = llm_guard(self.llm, question)
        if not ok:
            yield {"rejected": True, "content": GUARD_REJECTION}
            return
        self._guard_trusted.add(session_id)

        # ---- 指代消解 ----
        resolved = resolve_question(self.llm, question, chat_history)

        # ---- 检索 ----
        docs = self.retriever.invoke(resolved)
        if not docs:
            yield {"content": NO_RESULT_ANSWER}
            yield {"metadata": {
                "resolved_query": resolved, "sources": [], "citations": [],
                "consistency_issues": [],
                "latency_ms": int((time.time() - total_start) * 1000),
                "dialogue_turn": len(chat_history) // 2,
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
        history = self._histories.get(session_id)
        if history:
            history.add_message(HumanMessage(content=question))
            history.add_message(AIMessage(content=pp["answer"]))

        yield {"metadata": {
            "resolved_query": resolved,
            "sources": docs_to_sources(docs),
            "citations": pp["citations"],
            "consistency_issues": pp["consistency_issues"],
            "latency_ms": int((time.time() - total_start) * 1000),
            "dialogue_turn": len(history.messages) // 2 if history else 0,
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
