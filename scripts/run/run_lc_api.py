# -*- coding: utf-8 -*-
"""
LangChain 标准版 API 服务启动脚本
==================================
启动 LangChain 标准版 FastAPI 应用（默认端口 8001），
可与手撕版（8000）并行运行，共享数据资产与 .env 配置。

用法:
  python scripts/run/run_lc_api.py [--host 0.0.0.0] [--port 8001] [--reload]
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
# 标准版完全独立：只加 langchain_app 与项目根，不加 src/
# （否则 src/config.py 会抢先进入 sys.modules，污染 langchain_app 的配置加载）
sys.path.insert(0, str(PROJECT_ROOT / 'langchain_app'))
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="启动药典智能问答 API 服务（LangChain 标准版）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认: 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8001, help="监听端口（默认: 8001）")
    parser.add_argument("--reload", action="store_true", help="开发模式（热重载）")
    args = parser.parse_args()

    # 检查 API Key（langchain_app/config.py 会自动加载项目根目录 .env）
    from config import LLM_API_KEY
    if not LLM_API_KEY:
        print("=" * 60)
        print("❌ 错误: 未设置 LONGCAT_API_KEY！")
        print("=" * 60)
        print("\n请通过以下任一方式设置 LLM API Key：")
        print("\n  方式一 — 环境变量 (PowerShell):")
        print('    $env:LONGCAT_API_KEY="your_api_key_here"')
        print("\n  方式二 — 项目根目录 .env 文件（推荐，已被 git 忽略）:")
        print("    创建 .env 文件，写入一行: LONGCAT_API_KEY=your_api_key_here")
        print("\n获取 API Key: https://longcat.chat/platform/api_keys")
        print()
        sys.exit(1)

    print("=" * 60)
    print("  药典 RAG 系统 - API 服务启动（LangChain 标准版）")
    print("=" * 60)
    print(f"  监听地址: http://{args.host}:{args.port}")
    print(f"  API 文档: http://{args.host}:{args.port}/docs")
    print("  引擎: LangChain 1.x 标准组件")
    print("        FAISS + BM25Retriever -> Ensemble(RRF) -> CrossEncoder 重排")
    print("        LCEL 主链 + RunnableBranch 守卫 + 会话记忆")
    print(f"  LLM 模型: {os.environ.get('LONGCAT_MODEL', 'LongCat-2.0')}")
    print("=" * 60)
    print("\n正在初始化索引与模型（首次启动需要 10-20 秒）...\n")

    import uvicorn
    uvicorn.run(
        "langchain_app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == '__main__':
    main()
