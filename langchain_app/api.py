# -*- coding: utf-8 -*-
"""
FastAPI 应用（LangChain 标准版）
================================
端点与请求/响应 Schema 与主项目完全一致（schemas 为独立副本），
现有 Streamlit 前端零改动即可切换引擎。

启动方式：
  python scripts/run/run_lc_api.py   # 默认端口 8001，可与主项目 8000 并行
"""
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from schemas import (
    ChatRequest, ChatResponse, SourceItem,
    SearchResultItem, SearchResponse,
    DrugListResponse,
    HealthResponse, StatsResponse,
    SessionCreateResponse, SessionInfo, SessionListResponse,
)

from service import ChatService

logger = logging.getLogger(__name__)

_service: Optional[ChatService] = None


def get_service() -> ChatService:
    """获取全局服务单例（首次调用时加载索引与模型）"""
    global _service
    if _service is None:
        _service = ChatService()
    return _service


app = FastAPI(
    title="中国药典智能问答系统 API（LangChain 标准版）",
    description="""
基于 LangChain 1.x 标准组件搭建的《中国药典》RAG 问答系统（完全独立实现）：

- 检索：FAISS + BM25Retriever → EnsembleRetriever(RRF) → CrossEncoder 重排
- 生成：LCEL 声明式链 + RunnableBranch 守卫 + RunnableWithMessageHistory 会话记忆
- 端点与主项目同构，可无缝切换
    """,
    version="2.0.0-lc",
    contact={
        "name": "Chinese Medicine RAG (LangChain Standard)",
        "url": "https://github.com/qiz7z/RAG-ChineseMedicine-QA",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["系统"])
def health_check():
    try:
        svc = get_service()
        return HealthResponse(
            status="ok",
            version="2.0.0-lc",
            model=svc.llm.model_name,
            chunks_count=svc.retriever.vectorstore.index.ntotal,
            drug_count=len(svc.retriever.analyzer.drug_names),
        )
    except Exception as e:
        logger.error(f"健康检查失败: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"服务不可用: {e}")


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["系统"])
def get_stats():
    svc = get_service()
    stats = svc.stats()
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        total_drugs=stats["total_drugs"],
        by_category=stats["by_category"],
        by_chunk_type=stats["by_chunk_type"],
        index_status={
            "vector_index": svc.retriever.vectorstore.index.ntotal,
            "bm25_index": len(svc.retriever.bm25.docs) if svc.retriever.bm25 else 0,
            "engine": "langchain-standard",
        },
    )


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["问答"])
def chat(request: ChatRequest):
    """智能问答接口（LangChain 标准版，支持多轮对话）"""
    svc = get_service()
    try:
        out = svc.answer(
            request.question,
            session_id=request.session_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        sources = out.get("sources", []) if request.return_sources else []
        return ChatResponse(
            answer=out["answer"],
            session_id=out["session_id"],
            sources=[SourceItem(**s) for s in sources],
            citations=out.get("citations", []),
            resolved_query=out.get("resolved_query", request.question),
            latency_ms=int(out.get("latency", 0) * 1000),
            component_latency={"retrieval": out.get("retrieval_latency", 0.0)},
            dialogue_turn=out.get("dialogue_turn", 0),
            consistency_issues=out.get("consistency_issues", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@app.post("/api/v1/chat/stream", tags=["问答"])
def chat_stream(request: ChatRequest):
    """流式问答接口（SSE）"""
    svc = get_service()

    def event_stream():
        try:
            for event in svc.answer_stream(
                request.question,
                session_id=request.session_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                if "rejected" in event:
                    yield "data: " + json.dumps(
                        {"content": event["content"], "rejected": True,
                         "session_id": request.session_id or ""},
                        ensure_ascii=False,
                    ) + "\n\n"
                    return
                if "content" in event:
                    yield "data: " + json.dumps(
                        {"content": event["content"]}, ensure_ascii=False
                    ) + "\n\n"
                if "metadata" in event:
                    meta = event["metadata"]
                    yield "data: " + json.dumps(
                        {"done": True, "session_id": request.session_id or "", **meta},
                        ensure_ascii=False,
                    ) + "\n\n"
        except Exception as e:
            logger.error(f"流式问答失败: {e}", exc_info=True)
            yield "data: " + json.dumps({"error": str(e)}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/search", response_model=SearchResponse, tags=["检索"])
def search(
    q: str = Query(..., min_length=1, max_length=500),
    drug: Optional[str] = Query(None, description="药品名过滤"),
    section: Optional[str] = Query(None, description="章节过滤"),
    top_k: int = Query(5, ge=1, le=30),
):
    """检索查询接口（标准混合检索，section 超量召回后过滤）"""
    svc = get_service()
    try:
        t0 = time.time()
        fetch_k = min(top_k * 4, 30) if section else top_k
        items = svc.search(q, top_k=fetch_k, drug_filter=drug)
        if section:
            items = [it for it in items if it["section"] == section]
        items = items[:top_k]
        return SearchResponse(
            query=q,
            results=[SearchResultItem(**it) for it in items],
            total=len(items),
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.get("/api/v1/drugs", response_model=DrugListResponse, tags=["药品"])
def list_drugs(
    keyword: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """药品列表（来自索引元数据）"""
    svc = get_service()
    stats = svc.stats()
    # 药品名清单来自查询理解器加载的 drug_names.json
    all_drugs = list(svc.retriever.analyzer.drug_names)
    if keyword:
        all_drugs = [d for d in all_drugs if keyword in d]
    if category:
        # 按分类过滤：扫描 docstore 元数据
        cat_drugs = {
            doc.metadata.get("drug_name")
            for doc in svc.retriever.vectorstore.docstore._dict.values()
            if doc.metadata.get("category") == category
        }
        all_drugs = [d for d in all_drugs if d in cat_drugs]
    all_drugs = all_drugs[:limit]
    return DrugListResponse(total=len(all_drugs), drugs=all_drugs)


@app.post("/api/v1/sessions", response_model=SessionCreateResponse, tags=["会话"])
def create_session():
    svc = get_service()
    sid = svc.create_session()
    return SessionCreateResponse(
        session_id=sid, created_at=time.strftime("%Y-%m-%d %H:%M:%S")
    )


@app.get("/api/v1/sessions", response_model=SessionListResponse, tags=["会话"])
def list_sessions():
    svc = get_service()
    sessions = svc.list_sessions()
    return SessionListResponse(total=len(sessions), sessions=sessions)


@app.get("/api/v1/sessions/{session_id}", response_model=SessionInfo, tags=["会话"])
def get_session_info(session_id: str):
    info = get_service().session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="会话不存在")
    return SessionInfo(**info)


@app.delete("/api/v1/sessions/{session_id}", tags=["会话"])
def delete_session(session_id: str):
    if get_service().delete_session(session_id):
        return {"message": "会话已删除", "session_id": session_id}
    raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/", tags=["根"])
def root():
    return {
        "name": "中国药典智能问答系统 API（LangChain 标准版）",
        "version": "2.0.0-lc",
        "docs": "/docs",
        "engine": "LangChain 1.x standard: FAISS+BM25 Ensemble(RRF) + CrossEncoder rerank + LCEL",
    }
