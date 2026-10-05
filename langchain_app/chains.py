# -*- coding: utf-8 -*-
"""
主程序：会话式 RAG 链（全声明式 LCEL）
======================================
本项目 LangChain 版的"粘合层"。整条链由官方 Runnable 组合而成：

  [多轮] 历史感知问题改写（condense question）
    → PharmacopoeiaRetriever（Ensemble RRF + CrossEncoder 重排）
    → RAG_PROMPT | ChatOpenAI | StrOutputParser()
    → 引用标注节点
  外层 RunnableWithMessageHistory 注入按 session_id 隔离的对话历史。

**链是无状态的纯流程：只负责"问题 → 答案"，不做守卫、不判会话。**
守卫（含会话级信任通道）由 service.ChatService._guard 统一做一次，同步与流式共用。
早先链里还有一道 `_guard_step`（调本模块的 check_guard），使同步路径的守卫跑两遍，
且链里那遍不认信任通道（缺陷 23，已移除）——检查站只留在"前台"（service）。

链的输出为 dict：{answer, citations, sources, resolved_query, retrieval_latency}。
流式路径见 service.answer_stream（复用本模块的构建块，逐步 yield）。
"""
import sys
import time
from pathlib import Path
from typing import List

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from llm import build_chat_model
from retrievers import build_hybrid_retriever
from guard import keyword_guard, llm_guard
from prompts import RAG_PROMPT, CONDENSE_PROMPT, NO_RESULT_ANSWER
from postprocess import postprocess, format_docs, docs_to_sources


# ============================================================
# 构建块（声明式链与流式路径共用，保证两条路径行为一致）
# ============================================================

def check_guard(llm, question: str, trusted: bool = False) -> bool:
    """两层守卫：关键词快通道 →（词表无法判定时）LLM 语义判定

    Args:
        trusted: 该会话**此前已通过过守卫**（追问场景）。此时：
                 - 词表命中"无关" → 仍然拒绝（信任不等于放行任何话）
                 - 词表判不出（ambiguous） → **跳过 LLM 判定**，省一次调用
                 这就是"会话级信任通道"，由 service 持有 `_guard_trusted` 决定是否传 True。

    只此一份实现：链不再自带守卫（缺陷 23 已移除链里的 `_guard_step`），
    避免"同步路径守卫跑两遍、且链里那遍不认信任通道"。
    """
    ok, reason = keyword_guard(question)
    if reason == "ambiguous":
        return True if trusted else llm_guard(llm, question)
    return ok


def resolve_question(llm, question: str, chat_history: List) -> str:
    """多轮指代消解：有历史时把追问改写为独立问题（condense question 模式）"""
    if not chat_history:
        return question
    chain = CONDENSE_PROMPT | llm.bind(temperature=0, max_tokens=1024) | StrOutputParser()
    try:
        resolved = chain.invoke(
            {"question": question, "chat_history": chat_history}
        ).strip()
        if resolved and len(resolved) < len(question) * 3:
            return resolved
        return question
    except Exception:
        return question


def retrieval_step(retriever):
    """返回"独立问题 → Documents"的 Runnable"""

    def _run(standalone_question: str):
        return retriever.invoke(standalone_question)

    return RunnableLambda(_run)


def _no_result_output(resolved: str) -> dict:
    """检索为空时的统一输出。

    注意：这里**不再有 `guard_rejected`**——拒答已完全归 service（前台）负责，
    链只产出"答得出"或"资料里没有"两种结果。
    """
    return {
        "answer": NO_RESULT_ANSWER,
        "citations": [],
        "consistency_issues": [],
        "sources": [],
        "resolved_query": resolved,
    }


# ============================================================
# 声明式主链
# ============================================================

def build_chat_chain(retriever=None, llm=None):
    """构建完整会话式 RAG 链。

    用法:
        chain = build_chat_chain()
        out = chain.invoke(
            {"question": "人参的用法用量？"},
            config={"configurable": {"session_id": "abc"}},
        )
    """
    retriever = retriever or build_hybrid_retriever()
    llm = llm or build_chat_model()

    generate_chain = RAG_PROMPT | llm | StrOutputParser()

    def _core(state: dict) -> dict:
        """RAG 核心：改写 → 检索 → 生成 → 引用标注"""
        question = state["question"]
        chat_history = state.get("chat_history") or []

        t0 = time.time()
        resolved = resolve_question(llm, question, chat_history)

        docs = retriever.invoke(resolved)
        if not docs:
            return _no_result_output(resolved)

        raw = generate_chain.invoke(
            {
                "context": format_docs(docs),
                "question": question,   # 回答用原始问题（更自然），检索用改写后问题
                "chat_history": chat_history,
            }
        )
        pp = postprocess(raw, docs)
        return {
            "answer": pp["answer"],
            "citations": pp["citations"],
            "consistency_issues": pp["consistency_issues"],
            "sources": docs_to_sources(docs),
            "resolved_query": resolved,
            "retrieval_latency": time.time() - t0,
        }

    # 链 = 纯流程。守卫与拒答由 service 在调用本链**之前**完成（缺陷 23），
    # 所以这里不再有 _guard_step / 守卫短路分支。
    session_chain = RunnableLambda(_core)

    _histories: dict = {}

    def _get_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in _histories:
            _histories[session_id] = InMemoryChatMessageHistory()
        return _histories[session_id]

    chain = RunnableWithMessageHistory(
        session_chain,
        _get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",  # 链输出为 dict：按 key 自动保存 AI 回复到历史
    )
    # 会话存储随链返回（API 层需要做会话列表/删除）
    chain._histories = _histories  # noqa: SLF001 — 有意暴露给服务层
    return chain
