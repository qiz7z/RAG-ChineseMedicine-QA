# -*- coding: utf-8 -*-
"""
FastAPI 应用主模块
===================
基于 FastAPI 构建药典智能问答 RESTful API。

API 端点：
  POST /api/v1/chat          - 智能问答（支持多轮对话）
  POST /api/v1/chat/stream   - 流式问答（SSE 实时输出）
  GET  /api/v1/search        - 检索查询
  GET  /api/v1/drugs         - 药品列表查询
  GET  /api/v1/stats         - 系统统计信息
  GET  /api/v1/health        - 健康检查
  GET  /api/v1/sessions      - 会话列表
  POST /api/v1/sessions      - 创建会话
  DELETE /api/v1/sessions/{id} - 删除会话
  GET  /docs                 - Swagger UI 文档

启动方式：
  python scripts/run/run_api.py
  或
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# 确保 src 目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import (
    ChatRequest, ChatResponse, SourceItem,
    SearchRequest, SearchResultItem, SearchResponse,
    DrugListResponse,
    HealthResponse, StatsResponse,
    SessionCreateResponse, SessionInfo, SessionListResponse,
    ErrorResponse,
)
from api.session import SessionManager
from generation.generator import Generator
from retrieval.retriever import Retriever
from indexing.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

# ============================================================
# 全局单例（延迟初始化）
# ============================================================

_generator: Optional[Generator] = None
_session_manager: Optional[SessionManager] = None
_meta_store: Optional[MetadataStore] = None


def get_generator() -> Generator:
    """获取全局 Generator 单例"""
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def get_session_manager() -> SessionManager:
    """获取全局会话管理器"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_meta_store() -> MetadataStore:
    """获取全局元数据存储"""
    global _meta_store
    if _meta_store is None:
        _meta_store = MetadataStore()
    return _meta_store


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="中国药典智能问答系统 API",
    description="""
基于 RAG 架构的《中国药典》2020年版一部智能问答系统。

## 功能
- **智能问答**：基于药典原文的自然语言问答
- **多轮对话**：支持上下文理解和指代消解
- **检索查询**：支持按药品名、章节过滤的精确检索
- **药品列表**：查询药典收录的所有药品

## 技术架构
- 检索：FAISS 向量检索 + BM25 关键词检索 + RRF 融合 + BGE 重排
- 生成：美团 LongCat 2.0 大语言模型
- 后处理：引用标注 + 一致性校验 + 用药安全提醒
    """,
    version="1.0.0",
    contact={
        "name": "Chinese Medicine RAG",
        "url": "https://github.com/meituan-longcat",
    },
)

# CORS 中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 健康检查 & 统计
# ============================================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    try:
        gen = get_generator()
        store = get_meta_store()
        return HealthResponse(
            status="ok",
            version="1.0.0",
            model=gen.llm_client.model,
            chunks_count=gen.retriever.vector_index.count(),
            drug_count=len(gen.retriever.query_parser.drug_names),
        )
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail=f"服务不可用: {e}")


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["系统"])
async def get_stats():
    """获取系统统计信息"""
    store = get_meta_store()
    stats = store.stats()
    gen = get_generator()
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        total_drugs=stats["total_drugs"],
        by_category=stats["by_category"],
        by_chunk_type=stats["by_chunk_type"],
        index_status={
            "vector_index": gen.retriever.vector_index.count(),
            "bm25_index": gen.retriever.bm25_index.count(),
            "metadata_store": store.count(),
        },
    )


# ============================================================
# 智能问答
# ============================================================

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["问答"])
async def chat(request: ChatRequest):
    """
    智能问答接口

    支持多轮对话，传入 session_id 可保持上下文。
    首次对话不传 session_id，系统会自动创建新会话。
    """
    gen = get_generator()
    sm = get_session_manager()

    # 获取或创建会话
    session_id = sm.get_or_create(request.session_id)
    session = sm.get_session(session_id)

    # 新会话时重置守卫信任状态
    if not request.session_id or not session.dialogue_manager.history:
        gen.guard.reset_session()

    # 将会话的对话历史同步到 Generator 的对话管理器
    # （Generator 内部有自己的 DialogueManager，我们需要把历史注入）
    if session and session.dialogue_manager.history:
        gen.dialogue_manager.history = session.dialogue_manager.history

    try:
        # 执行问答
        response = gen.answer(
            request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 将更新后的对话历史同步回 session
        if session:
            session.dialogue_manager.history = gen.dialogue_manager.history

        # 构建响应
        sources = []
        if request.return_sources and response.retrieval:
            for r in response.retrieval.results:
                sources.append(SourceItem(
                    chunk_id=r.chunk_id,
                    drug_name=r.drug_name,
                    section=r.section,
                    category=r.category,
                    content=r.content[:500],  # 截断内容
                    score=r.score,
                    rerank_score=r.rerank_score,
                    sources=r.sources,
                ))

        return ChatResponse(
            answer=response.answer,
            session_id=session_id,
            sources=sources,
            citations=response.citations,
            resolved_query=response.resolved_query,
            latency_ms=int(response.latency * 1000),
            component_latency=response.component_latency,
            dialogue_turn=response.dialogue_turn,
            consistency_issues=response.consistency_issues,
        )

    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@app.post("/api/v1/chat/stream", tags=["问答"])
async def chat_stream(request: ChatRequest):
    """
    流式问答接口（SSE）

    逐 token 返回 LLM 生成内容，适用于实时显示场景。
    使用 Server-Sent Events (SSE) 格式。
    """
    gen = get_generator()
    sm = get_session_manager()

    session_id = sm.get_or_create(request.session_id)
    session = sm.get_session(session_id)

    # 新会话时重置守卫信任状态
    if not request.session_id or not session.dialogue_manager.history:
        gen.guard.reset_session()

    if session and session.dialogue_manager.history:
        gen.dialogue_manager.history = session.dialogue_manager.history

    async def event_stream():
        try:
            for event in gen.answer_stream(
                request.question,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                # 非医药问题拒绝
                if "rejected" in event:
                    data = json.dumps({
                        "content": event["content"],
                        "rejected": True,
                        "session_id": session_id,
                    }, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    return

                # 流式内容片段
                if "content" in event:
                    data = json.dumps({"content": event["content"]}, ensure_ascii=False)
                    yield f"data: {data}\n\n"

                # 元数据（来源、引用等）——流式结束后一次性返回
                if "metadata" in event:
                    meta = event["metadata"]
                    # 同步对话历史到 session
                    if session:
                        session.dialogue_manager.history = gen.dialogue_manager.history

                    data = json.dumps({
                        "done": True,
                        "session_id": session_id,
                        "sources": meta["sources"],
                        "citations": meta["citations"],
                        "consistency_issues": meta["consistency_issues"],
                        "latency_ms": meta["latency_ms"],
                        "dialogue_turn": meta["dialogue_turn"],
                        "resolved_query": meta["resolved_query"],
                    }, ensure_ascii=False)
                    yield f"data: {data}\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 检索查询
# ============================================================

@app.get("/api/v1/search", response_model=SearchResponse, tags=["检索"])
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="查询关键词"),
    drug: Optional[str] = Query(None, description="药品名过滤"),
    section: Optional[str] = Query(None, description="章节过滤"),
    top_k: int = Query(5, ge=1, le=30, description="返回结果数"),
):
    """
    检索查询接口

    支持按药品名、章节过滤，返回检索结果。
    """
    gen = get_generator()

    try:
        t0 = time.time()
        response = gen.retriever.search(q, top_k=top_k, drug_filter=drug)
        latency_ms = int((time.time() - t0) * 1000)

        results = []
        for r in response.results:
            # 如果指定了章节过滤，在 Python 层面过滤
            if section and r.section != section:
                continue
            results.append(SearchResultItem(
                chunk_id=r.chunk_id,
                drug_name=r.drug_name,
                section=r.section,
                category=r.category,
                content=r.content[:500],
                score=r.score,
                rerank_score=r.rerank_score,
                sources=r.sources,
            ))

        return SearchResponse(
            query=q,
            results=results,
            total=len(results),
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


# ============================================================
# 药品列表
# ============================================================

@app.get("/api/v1/drugs", response_model=DrugListResponse, tags=["药品"])
async def list_drugs(
    keyword: Optional[str] = Query(None, description="搜索关键词（前缀匹配）"),
    category: Optional[str] = Query(None, description="分类过滤"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制"),
):
    """
    获取药品列表

    支持按关键词搜索和分类过滤。
    """
    store = get_meta_store()

    try:
        all_drugs = store.get_all_drug_names()

        # 关键词过滤
        if keyword:
            all_drugs = [d for d in all_drugs if keyword in d]

        # 分类过滤（需要查 SQLite）
        if category:
            # 先按分类查出该分类下的所有药品名
            chunks = store.filter(category=category)
            category_drugs = set(c["drug_name"] for c in chunks)
            all_drugs = [d for d in all_drugs if d in category_drugs]

        # 限制数量
        all_drugs = all_drugs[:limit]

        return DrugListResponse(
            total=len(all_drugs),
            drugs=all_drugs,
        )

    except Exception as e:
        logger.error(f"药品列表查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# ============================================================
# 会话管理
# ============================================================

@app.post("/api/v1/sessions", response_model=SessionCreateResponse, tags=["会话"])
async def create_session():
    """创建新会话"""
    sm = get_session_manager()
    session_id = sm.create_session()
    return SessionCreateResponse(
        session_id=session_id,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.get("/api/v1/sessions", response_model=SessionListResponse, tags=["会话"])
async def list_sessions():
    """列出所有活跃会话"""
    sm = get_session_manager()
    sessions = sm.list_sessions()
    return SessionListResponse(
        total=len(sessions),
        sessions=sessions,
    )


@app.get("/api/v1/sessions/{session_id}", response_model=SessionInfo, tags=["会话"])
async def get_session_info(session_id: str):
    """获取会话详情"""
    sm = get_session_manager()
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return SessionInfo(
        session_id=session_id,
        turn_count=session.turn_count,
        history=session.get_history(),
    )


@app.delete("/api/v1/sessions/{session_id}", tags=["会话"])
async def delete_session(session_id: str):
    """删除会话"""
    sm = get_session_manager()
    if sm.delete_session(session_id):
        return {"message": "会话已删除", "session_id": session_id}
    raise HTTPException(status_code=404, detail="会话不存在")


# ============================================================
# 根路由
# ============================================================

@app.get("/", tags=["根"])
async def root():
    """根路由 - 重定向到 API 文档"""
    return {
        "name": "中国药典智能问答系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /api/v1/chat",
            "chat_stream": "POST /api/v1/chat/stream",
            "search": "GET /api/v1/search",
            "drugs": "GET /api/v1/drugs",
            "health": "GET /api/v1/health",
            "stats": "GET /api/v1/stats",
            "sessions": "GET /api/v1/sessions",
        },
    }
