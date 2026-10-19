# -*- coding: utf-8 -*-
"""
API 服务启动脚本
==================
启动 FastAPI 应用，提供药典智能问答 API 服务。

用法:
  python scripts/run/run_api.py [--host 0.0.0.0] [--port 8000] [--reload]

环境变量（也可写在项目根目录 .env）:
  LLM_API_KEY        - LLM API Key（必需）
  LLM_BASE_URL       - OpenAI 兼容端点
  LLM_MODEL          - 模型名称
"""
import sys
import io
import os
import argparse
from pathlib import Path

# 修复 Windows 控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description="启动药典智能问答 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认: 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认: 8000）")
    parser.add_argument("--reload", action="store_true", help="开发模式（热重载）")
    args = parser.parse_args()

    # 检查 LLM 配置（导入 config 时自动加载项目根目录 .env）
    from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    missing = [
        name for name, val in (
            ("LLM_API_KEY", LLM_API_KEY),
            ("LLM_BASE_URL", LLM_BASE_URL),
            ("LLM_MODEL", LLM_MODEL),
        ) if not val
    ]
    if missing:
        print("=" * 60)
        print(f"❌ 错误: LLM 配置缺失: {', '.join(missing)}")
        print("=" * 60)
        print("\n请在项目根目录 .env 中补齐（任意 OpenAI 兼容服务均可）：")
        print("  LLM_API_KEY=your_api_key_here")
        print("  LLM_BASE_URL=https://your-host/v1")
        print("  LLM_MODEL=your-model-name")
        print()
        sys.exit(1)

    # 导入并启动
    print("=" * 60)
    print("  药典 RAG 系统 - API 服务启动")
    print("=" * 60)
    print(f"  监听地址: http://{args.host}:{args.port}")
    print(f"  API 文档: http://{args.host}:{args.port}/docs")
    print(f"  LLM 模型: {LLM_MODEL or '未配置'}")
    print("=" * 60)
    print("\n正在初始化模型和索引（首次启动需要 10-15 秒）...\n")

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == '__main__':
    main()
