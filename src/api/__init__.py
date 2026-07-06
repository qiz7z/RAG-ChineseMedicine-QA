# -*- coding: utf-8 -*-
"""
API 服务模块（阶段五）
======================
基于 FastAPI 构建 RESTful API，对外提供药典智能问答服务。

模块组成：
  - schemas:  请求/响应数据模型（Pydantic）
  - session:  会话管理器（多轮对话 session 管理）
  - main:     FastAPI 应用与路由定义

API 端点：
  POST /api/v1/chat          - 智能问答（支持多轮对话）
  POST /api/v1/chat/stream   - 流式问答（SSE）
  GET  /api/v1/search        - 检索查询
  GET  /api/v1/drugs         - 药品列表查询
  GET  /api/v1/stats         - 系统统计信息
  GET  /api/v1/health        - 健康检查
  GET  /docs                 - Swagger UI（自动生成文档）
"""
