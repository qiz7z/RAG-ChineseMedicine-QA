# -*- coding: utf-8 -*-
"""
LangChain 版生成引擎
====================
用 LCEL 组合 RAG 生成链，行为与手撕版 Generator 逐段对齐（保证对比实验同口径）：

  守卫 → 指代消解 → 混合检索 → Prompt 构建 → LLM → 后处理

与手撕版的关键差异：
  - 会话状态（对话历史 / 守卫信任态）按 session_id 隔离，
    修复手撕版"全局单例被所有用户共享"的并发缺陷
  - LLM 调用经 langchain_openai.ChatOpenAI，管线以 LCEL 组合
  - 复用手撕版零耦合模块：prompts / PostProcessor / GuardChecker /
    DialogueManager / GenerationResponse / SearchResult

对 GuardChecker、DialogueManager 的复用方式：这两个类只依赖 duck-typed
的 llm_client.chat(messages, ...) -> str，提供一个薄适配器把 ChatOpenAI
包装成该协议即可，无需修改原类。
"""
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))                    # retriever/llm/embeddings 平铺导入
sys.path.insert(0, str(APP_DIR.parent / "src"))     # config / src 各模块

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from config import RERANKER_TOP_K
from generation.dialogue import DialogueManager
from generation.guard import GuardChecker, OFF_TOPIC_REJECTION
from generation.postprocessor import PostProcessor
from generation.prompts import build_prompt, build_no_result_prompt, build_context_section
from generation.generator import GenerationResponse
from retrieval.retriever import RetrievalResponse, SearchResult

from retriever import HybridRetriever, doc_to_result_dict
from llm import build_chat_model

logger = logging.getLogger(__name__)


def to_lc_messages(messages: List[Dict]) -> List:
    """OpenAI dict 消息格式 → LangChain 消息对象"""
    mapping = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    lc_msgs = []
    for m in messages:
        cls = mapping.get(m.get("role"), HumanMessage)
        lc_msgs.append(cls(content=m.get("content", "")))
    return lc_msgs


class _ChatOpenAIAdapter:
    """把 ChatOpenAI 包装成手撕版 llm_client 的 duck-typed 协议。

    GuardChecker / DialogueManager 只需要 .chat(messages, temperature,
    max_tokens) -> str 和 .model 属性，通过本适配器即可直接复用。
    """

    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.model = chat_model.model_name

    def chat(self, messages, temperature=None, max_tokens=None):
        model = self.chat_model
        overrides = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        if overrides:
            model = model.bind(**overrides)
        return model.invoke(to_lc_messages(messages)).content


class LCGenerator:
    """LangChain 版生成引擎（对外接口与手撕版 Generator 一致）"""

    def __init__(
        self,
        retriever: HybridRetriever = None,
        enable_reranker: bool = None,
        enable_dialogue: bool = True,
        enable_postprocess: bool = True,
    ):
        self.retriever = retriever or HybridRetriever.create(
            enable_reranker=enable_reranker
        )
        self.chat_model = build_chat_model()
        self.llm_adapter = _ChatOpenAIAdapter(self.chat_model)
        self.enable_dialogue = enable_dialogue
        self.enable_postprocess = enable_postprocess
        self.postprocessor = PostProcessor()

        # per-session 状态：每个会话独立的对话历史与守卫信任态
        self._sessions: Dict[str, dict] = {}

    # ----------------------------------------------------------
    # 会话管理
    # ----------------------------------------------------------

    def _get_session(self, session_id: Optional[str]):
        """获取或创建会话（每会话独立 DialogueManager + GuardChecker）"""
        if not session_id:
            session_id = f"lcsess_{uuid.uuid4().hex[:12]}"
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "dialogue": DialogueManager(llm_client=self.llm_adapter),
                "guard": GuardChecker(llm_client=self.llm_adapter),
            }
        return session_id, self._sessions[session_id]

    def reset_session(self, session_id: str):
        """重置会话状态"""
        self._sessions.pop(session_id, None)

    @property
    def llm_model_name(self) -> str:
        return self.chat_model.model_name

    # ----------------------------------------------------------
    # 检索（与手撕版 Retriever.search 对齐，供 /api/v1/search 与评估器使用）
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = None,
        drug_filter: Optional[str] = None,
    ) -> RetrievalResponse:
        top_k = top_k or RERANKER_TOP_K
        docs, parsed, latency, comp = self.retriever.hybrid_search(
            query, top_k=top_k, drug_filter=drug_filter
        )
        results = [SearchResult(**doc_to_result_dict(d)) for d in docs]
        result_dicts = [r.to_dict() for r in results]
        return RetrievalResponse(
            query=query,
            parsed=parsed,
            results=results,
            context=build_context_section(result_dicts) if result_dicts else "",
            latency=latency,
            component_latency=comp,
        )

    # ----------------------------------------------------------
    # 同步问答（LCEL 管线）
    # ----------------------------------------------------------

    def answer(
        self,
        query: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResponse:
        total_start = time.time()
        component_latency = {}
        session_id, sess = self._get_session(session_id)
        dm, guard = sess["dialogue"], sess["guard"]

        # ============ 0. 问题领域守卫 ============
        t0 = time.time()
        is_followup = self.enable_dialogue and dm.turn_count > 0
        is_related, guard_reason = guard.check(query, is_followup=is_followup)
        component_latency["guard_check"] = time.time() - t0

        if not is_related:
            logger.info(f"问题被守卫拦截: {query} ({guard_reason})")
            return GenerationResponse(
                query=query,
                resolved_query=query,
                answer=OFF_TOPIC_REJECTION,
                retrieval=None,
                raw_answer=OFF_TOPIC_REJECTION,
                citations=[],
                consistency_issues=[],
                latency=time.time() - total_start,
                component_latency=component_latency,
                llm_model=self.llm_model_name,
                dialogue_turn=dm.turn_count,
            )

        # ============ 1. 指代消解 ============
        t0 = time.time()
        resolved_query = query
        if self.enable_dialogue and dm.turn_count > 0:
            resolved_query = dm.resolve_query(query)
        component_latency["coreference_resolution"] = time.time() - t0

        # ============ 2. 检索 ============
        t0 = time.time()
        retrieval_response = self.search(resolved_query)
        component_latency["retrieval"] = time.time() - t0

        # ============ 3. Prompt 构建 ============
        t0 = time.time()
        retrieval_results = [r.to_dict() for r in retrieval_response.results]
        if retrieval_results:
            messages = build_prompt(
                user_query=resolved_query,
                retrieval_results=retrieval_results,
                chat_history=dm.get_history_messages() if self.enable_dialogue else None,
            )
        else:
            messages = build_no_result_prompt(resolved_query)
        component_latency["prompt_building"] = time.time() - t0

        # ============ 4. LLM 生成（LCEL: messages | model | parser） ============
        t0 = time.time()
        model = self.chat_model
        if temperature is not None or max_tokens is not None:
            overrides = {}
            if temperature is not None:
                overrides["temperature"] = temperature
            if max_tokens is not None:
                overrides["max_tokens"] = max_tokens
            model = model.bind(**overrides)
        chain = to_lc_messages | model | StrOutputParser()
        raw_answer = chain.invoke(messages)
        component_latency["llm_generation"] = time.time() - t0

        # ============ 5. 后处理 ============
        final_answer, citations, consistency_issues = self._postprocess(
            raw_answer, retrieval_results
        )
        component_latency["post_processing"] = 0.0

        # ============ 6. 更新对话历史 ============
        if self.enable_dialogue:
            dm.add_user_message(query, resolved_query)
            dm.add_assistant_message(final_answer)

        return GenerationResponse(
            query=query,
            resolved_query=resolved_query,
            answer=final_answer,
            retrieval=retrieval_response,
            raw_answer=raw_answer,
            citations=citations,
            consistency_issues=consistency_issues,
            latency=time.time() - total_start,
            component_latency=component_latency,
            llm_model=self.llm_model_name,
            dialogue_turn=dm.turn_count,
        )

    # ----------------------------------------------------------
    # 流式问答（事件格式与手撕版 Generator.answer_stream 一致）
    # ----------------------------------------------------------

    def answer_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        total_start = time.time()
        session_id, sess = self._get_session(session_id)
        dm, guard = sess["dialogue"], sess["guard"]

        # 0. 守卫
        is_followup = self.enable_dialogue and dm.turn_count > 0
        is_related, guard_reason = guard.check(query, is_followup=is_followup)
        if not is_related:
            logger.info(f"问题被守卫拦截(流式): {query} ({guard_reason})")
            yield {"rejected": True, "content": OFF_TOPIC_REJECTION}
            return

        # 1. 指代消解
        resolved_query = query
        if self.enable_dialogue and dm.turn_count > 0:
            resolved_query = dm.resolve_query(query)

        # 2. 检索
        retrieval_response = self.search(resolved_query)
        retrieval_results = [r.to_dict() for r in retrieval_response.results]

        # 3. Prompt 构建
        if retrieval_results:
            messages = build_prompt(
                user_query=resolved_query,
                retrieval_results=retrieval_results,
                chat_history=dm.get_history_messages() if self.enable_dialogue else None,
            )
        else:
            messages = build_no_result_prompt(resolved_query)

        # 4. 流式生成
        model = self.chat_model
        if temperature is not None or max_tokens is not None:
            overrides = {}
            if temperature is not None:
                overrides["temperature"] = temperature
            if max_tokens is not None:
                overrides["max_tokens"] = max_tokens
            model = model.bind(**overrides)
        chain = to_lc_messages | model | StrOutputParser()

        raw_answer = ""
        for chunk in chain.stream(messages):
            raw_answer += chunk
            yield {"content": chunk}

        # 5. 后处理（流式结束后执行，与手撕版一致）
        final_answer, citations, consistency_issues = self._postprocess(
            raw_answer, retrieval_results
        )

        # 6. 更新对话历史
        if self.enable_dialogue:
            dm.add_user_message(query, resolved_query)
            dm.add_assistant_message(final_answer)

        # 7. 元数据事件
        sources = []
        for r in retrieval_response.results:
            sources.append({
                "chunk_id": r.chunk_id,
                "drug_name": r.drug_name,
                "section": r.section,
                "category": r.category,
                "content": r.content[:500],
                "score": r.score,
                "rerank_score": r.rerank_score,
                "sources": r.sources,
            })

        yield {
            "metadata": {
                "resolved_query": resolved_query,
                "sources": sources,
                "citations": citations,
                "consistency_issues": consistency_issues,
                "latency_ms": int((time.time() - total_start) * 1000),
                "dialogue_turn": dm.turn_count,
            }
        }

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------

    def _postprocess(self, raw_answer: str, retrieval_results: List[Dict]):
        """后处理（与手撕版语义一致）"""
        if self.enable_postprocess and retrieval_results:
            pp = self.postprocessor.process(raw_answer, retrieval_results)
            return pp.answer, pp.citations, pp.consistency_issues
        return raw_answer, [], []

    def simple_answer(self, query: str) -> str:
        """单轮快速问答（无会话状态）"""
        resp = self.answer(query)
        return resp.answer
