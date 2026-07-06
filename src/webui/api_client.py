# -*- coding: utf-8 -*-
"""
API HTTP 客户端
================
封装与后端 FastAPI 的所有 HTTP 通信，提供：
  - 健康检查
  - 智能问答（同步 + 流式 SSE）
  - 检索查询
  - 药品列表
  - 系统统计
  - 会话管理

所有方法返回 Python 字典或生成器，供 Streamlit 前端直接使用。
"""
import json
import logging
import requests
from typing import Dict, List, Optional, Generator, Any

logger = logging.getLogger(__name__)


class APIClient:
    """后端 API HTTP 客户端"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        Args:
            base_url: API 基地址（不含末尾斜杠）
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        # 超时设置：连接 5s，读取 120s（LLM 生成可能较慢）
        self.timeout = (5, 120)

    # ========================================================
    # 系统接口
    # ========================================================

    def health(self) -> Dict[str, Any]:
        """健康检查"""
        r = self.session.get(
            f"{self.base_url}/api/v1/health",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def stats(self) -> Dict[str, Any]:
        """系统统计信息"""
        r = self.session.get(
            f"{self.base_url}/api/v1/stats",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ========================================================
    # 智能问答
    # ========================================================

    def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
        return_sources: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        同步问答接口。

        Args:
            question: 用户问题
            session_id: 会话 ID（多轮对话），None 则创建新会话
            return_sources: 是否返回检索来源
            temperature: LLM 温度
            max_tokens: LLM 最大 token

        Returns:
            API 响应字典，包含 answer, sources, citations 等
        """
        payload: Dict[str, Any] = {
            "question": question,
            "return_sources": return_sources,
        }
        if session_id:
            payload["session_id"] = session_id
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        r = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式问答接口（SSE）。

        逐 chunk 生成内容，适用于实时显示场景。

        Yields:
            解析后的 SSE 事件字典：
              - {"content": "..."}  — 文本片段
              - {"done": True, "session_id": "..."}  — 结束信号
              - {"error": "..."}  — 错误信息
        """
        payload: Dict[str, Any] = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        r = self.session.post(
            f"{self.base_url}/api/v1/chat/stream",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            # SSE 格式：data: {...}
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning(f"无法解析 SSE 数据: {data_str}")

    # ========================================================
    # 检索查询
    # ========================================================

    def search(
        self,
        q: str,
        drug: Optional[str] = None,
        section: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        检索查询接口。

        Args:
            q: 查询关键词
            drug: 药品名过滤
            section: 章节过滤
            top_k: 返回结果数

        Returns:
            检索响应字典
        """
        params: Dict[str, Any] = {"q": q, "top_k": top_k}
        if drug:
            params["drug"] = drug
        if section:
            params["section"] = section

        r = self.session.get(
            f"{self.base_url}/api/v1/search",
            params=params,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ========================================================
    # 药品列表
    # ========================================================

    def list_drugs(
        self,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        获取药品列表。

        Args:
            keyword: 搜索关键词（包含匹配）
            category: 分类过滤
            limit: 返回数量限制

        Returns:
            {"total": int, "drugs": [...]}
        """
        params: Dict[str, Any] = {"limit": limit}
        if keyword:
            params["keyword"] = keyword
        if category:
            params["category"] = category

        r = self.session.get(
            f"{self.base_url}/api/v1/drugs",
            params=params,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ========================================================
    # 会话管理
    # ========================================================

    def create_session(self) -> Dict[str, Any]:
        """创建新会话"""
        r = self.session.post(
            f"{self.base_url}/api/v1/sessions",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def list_sessions(self) -> Dict[str, Any]:
        """列出所有活跃会话"""
        r = self.session.get(
            f"{self.base_url}/api/v1/sessions",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """获取会话详情（含对话历史）"""
        r = self.session.get(
            f"{self.base_url}/api/v1/sessions/{session_id}",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """删除会话"""
        r = self.session.delete(
            f"{self.base_url}/api/v1/sessions/{session_id}",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
