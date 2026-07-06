# -*- coding: utf-8 -*-
"""
Web UI 启动脚本
================
启动 Streamlit Web UI 服务。

使用方式：
  python scripts/run/run_webui.py
  或
  streamlit run src/webui/app.py

环境变量：
  API_BASE_URL  — 后端 API 地址（默认 http://127.0.0.1:8000）
"""
import sys
import os
import subprocess
from pathlib import Path

# 设置 UTF-8 编码（Windows 环境）
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT_DIR)

# 将 src 加入 Python 路径
sys.path.insert(0, str(ROOT_DIR / "src"))

# 检查后端 API 是否可用
def check_api():
    """快速检查后端 API 是否运行"""
    try:
        import requests
        base = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
        r = requests.get(f"{base}/api/v1/health", timeout=3)
        if r.status_code == 200:
            print(f"  [OK] 后端 API 已连接: {base}")
            return True
        else:
            print(f"  [WARN] 后端 API 返回非 200: {r.status_code}")
            return False
    except Exception:
        print("  [WARN] 后端 API 未运行，Web UI 可正常打开但功能不可用")
        print("         请先启动后端: python scripts/run/run_api.py")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  中国药典智能问答系统 — Web UI")
    print("=" * 60)

    # 检查后端
    print("\n[1/2] 检查后端 API 服务...")
    check_api()

    # 启动 Streamlit
    print("\n[2/2] 启动 Streamlit Web UI...")
    app_path = ROOT_DIR / "src" / "webui" / "app.py"
    print(f"  应用路径: {app_path}")
    print(f"  访问地址: http://localhost:8501")
    print(f"\n{'=' * 60}")
    print("  按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")

    # 使用 streamlit CLI 启动
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
    ]

    try:
        subprocess.run(cmd, cwd=str(ROOT_DIR))
    except KeyboardInterrupt:
        print("\n\nWeb UI 已停止。")
    except Exception as e:
        print(f"\n启动失败: {e}")
        print("请确保已安装 streamlit: pip install streamlit")
        sys.exit(1)
