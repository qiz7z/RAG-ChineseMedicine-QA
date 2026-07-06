# -*- coding: utf-8 -*-
"""
API 数据模型（Pydantic Schemas）
=================================
定义所有 API 接口的请求体和响应体结构。
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ============================================================
# 通用模型
# ============================================================

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态: ok / error")
    version: str = Field(..., description="系统版本")
    model: str = Field(..., description="LLM 模型名称")
    chunks_count: int = Field(..., description="索引中的 chunk 总数")
    drug_count: int = Field(..., description="药品总数")


class StatsResponse(BaseModel):
    """系统统计信息响应"""
    total_chunks: int = Field(..., description="chunk 总数")
    total_drugs: int = Field(..., description="药品总数")
    by_category: Dict[str, int] = Field(..., description="按分类统计")
    by_chunk_type: Dict[str, int] = Field(..., description="按 chunk 类型统计")
    index_status: Dict[str, Any] = Field(..., description="各索引状态")


# ============================================================
# 问答接口
# ============================================================

class ChatRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID（多轮对话用），不传则创建新会话")
    return_sources: bool = Field(True, description="是否返回检索来源")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM 温度参数")
    max_tokens: Optional[int] = Field(None, ge=100, le=8192, description="LLM 最大输出 token 数")


class SourceItem(BaseModel):
    """检索来源项"""
    chunk_id: str = Field(..., description="chunk ID")
    drug_name: str = Field(..., description="药品名")
    section: str = Field(..., description="章节")
    category: str = Field(..., description="分类")
    content: str = Field(..., description="chunk 内容")
    score: float = Field(..., description="相关性分数")
    rerank_score: Optional[float] = Field(None, description="重排分数")
    sources: List[str] = Field(default_factory=list, description="命中来源（vector/bm25）")


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str = Field(..., description="回答文本")
    session_id: str = Field(..., description="会话 ID")
    sources: List[SourceItem] = Field(default_factory=list, description="检索来源")
    citations: List[str] = Field(default_factory=list, description="引用标注")
    resolved_query: str = Field(..., description="指代消解后的查询")
    latency_ms: int = Field(..., description="总耗时（毫秒）")
    component_latency: Dict[str, float] = Field(default_factory=dict, description="各阶段耗时（秒）")
    dialogue_turn: int = Field(..., description="对话轮次")
    consistency_issues: List[str] = Field(default_factory=list, description="一致性问题")


# ============================================================
# 检索接口
# ============================================================

class SearchRequest(BaseModel):
    """检索请求"""
    q: str = Field(..., min_length=1, max_length=500, description="查询关键词")
    drug: Optional[str] = Field(None, description="药品名过滤")
    section: Optional[str] = Field(None, description="章节过滤")
    top_k: int = Field(5, ge=1, le=30, description="返回结果数")


class SearchResultItem(BaseModel):
    """检索结果项"""
    chunk_id: str
    drug_name: str
    section: str
    category: str
    content: str
    score: float
    rerank_score: Optional[float] = None
    sources: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """检索响应"""
    query: str
    results: List[SearchResultItem]
    total: int
    latency_ms: int


# ============================================================
# 药品列表接口
# ============================================================

class DrugListResponse(BaseModel):
    """药品列表响应"""
    total: int = Field(..., description="药品总数")
    drugs: List[str] = Field(..., description="药品名列表")


# ============================================================
# 会话管理接口
# ============================================================

class SessionCreateResponse(BaseModel):
    """创建会话响应"""
    session_id: str = Field(..., description="会话 ID")
    created_at: str = Field(..., description="创建时间")


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    turn_count: int = Field(..., description="对话轮次")
    history: List[Dict[str, str]] = Field(default_factory=list, description="对话历史")


class SessionListResponse(BaseModel):
    """会话列表响应"""
    total: int
    sessions: List[str] = Field(..., description="会话 ID 列表")


# ============================================================
# 错误响应
# ============================================================

class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str
    error_code: Optional[str] = None
