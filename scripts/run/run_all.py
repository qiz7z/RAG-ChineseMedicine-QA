# -*- coding: utf-8 -*-
"""
一键启动脚本
=============
同时启动 API 后端服务和 Streamlit Web UI 前端。

用法:
  python scripts/run/run_all.py

环境变量:
  LONGCAT_API_KEY   - 美团 LongCat API Key（必需）
  API_BASE_URL      - API 地址（默认 http://127.0.0.1:8000）
  API_PORT          - API 端口（默认 8000）
  UI_PORT           - Web UI 端口（默认 8501）
"""
import sys
import os
import io
import time
import signal
import subprocess
from pathlib import Path

# 修复 Windows 控制台编码
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

# 路径配置
API_SCRIPT = PROJECT_ROOT / "scripts" / "run" / "run_api.py"
WEBUI_SCRIPT = PROJECT_ROOT / "src" / "webui" / "app.py"
API_PORT = os.environ.get("API_PORT", "8000")
UI_PORT = os.environ.get("UI_PORT", "8501")

# 存储子进程
processes = []


def print_banner():
    print()
    print("=" * 60)
    print("  中国药典智能问答系统 — 一键启动")
    print("=" * 60)
    print(f"  项目目录: {PROJECT_ROOT}")
    print(f"  API 服务: http://localhost:{API_PORT}")
    print(f"  API 文档: http://localhost:{API_PORT}/docs")
    print(f"  Web UI:  http://localhost:{UI_PORT}")
    print(f"  LLM 模型: {os.environ.get('LONGCAT_MODEL', 'LongCat-2.0')}")
    print("=" * 60)
    print()


def check_api_key():
    """检查 API Key 是否设置"""
    api_key = os.environ.get("LONGCAT_API_KEY", "")
    if not api_key:
        print("[ERROR] 未设置 LONGCAT_API_KEY 环境变量！")
        print()
        print("请先设置美团 LongCat API Key：")
        print()
        print("  PowerShell:")
        print('    $env:LONGCAT_API_KEY="your_api_key_here"')
        print()
        print("  CMD:")
        print('    set LONGCAT_API_KEY=your_api_key_here')
        print()
        print("获取 API Key: https://longcat.chat/platform/api_keys")
        print()
        sys.exit(1)
    print("[OK] LONGCAT_API_KEY 已设置")


def check_indexes():
    """检查索引文件是否存在"""
    chroma_dir = PROJECT_ROOT / "data" / "vectorstore" / "chroma"
    bm25_path = PROJECT_ROOT / "data" / "vectorstore" / "bm25_index.pkl"
    sqlite_path = PROJECT_ROOT / "data" / "vectorstore" / "metadata.db"

    all_ok = True
    if not chroma_dir.exists():
        print(f"[WARN] Chroma 向量库不存在: {chroma_dir}")
        all_ok = False
    if not bm25_path.exists():
        print(f"[WARN] BM25 索引不存在: {bm25_path}")
        all_ok = False
    if not sqlite_path.exists():
        print(f"[WARN] SQLite 元数据库不存在: {sqlite_path}")
        all_ok = False

    if not all_ok:
        print()
        print("[ERROR] 索引文件缺失，请先构建索引：")
        print("  python scripts/build/build_index.py")
        print()
        sys.exit(1)

    print(f"[OK] 索引文件就绪 (Chroma + BM25 + SQLite)")


def start_api():
    """启动 API 后端"""
    print()
    print("[1/2] 启动 API 后端服务...")
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, str(API_SCRIPT), "--host", "0.0.0.0", "--port", API_PORT],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    processes.append(("API", proc))
    print(f"  PID: {proc.pid}, 端口: {API_PORT}")

    # 等待 API 就绪（最多等 60 秒）
    print("  等待 API 初始化...", end="", flush=True)
    import requests
    for i in range(30):
        time.sleep(2)
        try:
            r = requests.get(f"http://127.0.0.1:{API_PORT}/api/v1/health", timeout=2)
            if r.status_code == 200:
                print(" OK!")
                data = r.json()
                print(f"  模型: {data.get('model', 'N/A')}")
                print(f"  索引: {data.get('chunks_count', 0)} chunks")
                print(f"  药品: {data.get('drug_count', 0)} 种")
                return True
        except Exception:
            print(".", end="", flush=True)
    print(" FAILED!")
    print("[ERROR] API 服务启动超时")
    return False


def start_webui():
    """启动 Streamlit Web UI"""
    print()
    print("[2/2] 启动 Streamlit Web UI...")
    env = os.environ.copy()
    env["API_BASE_URL"] = f"http://127.0.0.1:{API_PORT}"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(WEBUI_SCRIPT),
        "--server.port", UI_PORT,
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
        "--server.headless", "true",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    processes.append(("WebUI", proc))
    print(f"  PID: {proc.pid}, 端口: {UI_PORT}")

    # 等待 Streamlit 就绪
    print("  等待 Streamlit 初始化...", end="", flush=True)
    import requests
    for i in range(15):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{UI_PORT}/_stcore/health", timeout=2)
            if r.status_code == 200:
                print(" OK!")
                return True
        except Exception:
            print(".", end="", flush=True)
    print(" FAILED!")
    print("[WARN] Streamlit 可能未完全就绪，请稍后手动访问")
    return False


def cleanup(signum=None, frame=None):
    """清理子进程"""
    print()
    print("正在停止所有服务...")
    for name, proc in reversed(processes):
        if proc.poll() is None:
            print(f"  停止 {name} (PID: {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("所有服务已停止。")
    sys.exit(0)


def main():
    print_banner()

    # 前置检查
    check_api_key()
    check_indexes()

    # 注册信号处理（Ctrl+C 优雅退出）
    signal.signal(signal.SIGINT, cleanup)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, cleanup)

    # 启动服务
    if not start_api():
        sys.exit(1)
    if not start_webui():
        print("[WARN] Web UI 启动失败，API 服务仍在运行")

    # 完成
    print()
    print("=" * 60)
    print("  所有服务已启动！")
    print("=" * 60)
    print(f"  Web UI:  http://localhost:{UI_PORT}")
    print(f"  API 文档: http://localhost:{API_PORT}/docs")
    print()
    print("  按 Ctrl+C 停止所有服务")
    print("=" * 60)
    print()

    # 保持运行，监控子进程
    try:
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"[WARN] {name} 进程已退出 (code={proc.returncode})")
                    # 输出最后几行日志
                    if proc.stdout:
                        output = proc.stdout.read()
                        if output:
                            lines = output.decode('utf-8', errors='replace').strip().split('\n')
                            for line in lines[-10:]:
                                print(f"  [{name}] {line}")
                    cleanup()
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
