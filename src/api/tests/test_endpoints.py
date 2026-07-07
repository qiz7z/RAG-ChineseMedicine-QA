# -*- coding: utf-8 -*-
"""
API 接口自动化测试
==================
对 FastAPI 后端的关键端点进行端到端测试。

运行方式：
    # 方式一：启动 API 后运行 pytest
    pytest src/api/tests/test_endpoints.py -v

    # 方式二：直接运行（需要 API 已启动）
    python src/api/tests/test_endpoints.py

测试前请确保：
    1. API 服务已启动（python scripts/run/run_api.py）
    2. 已安装 pytest 和 requests（pip install pytest requests）
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

import requests

# pytest 为可选依赖：直接 python 运行时不需要
try:
    import pytest
except ImportError:
    pytest = None

# pytest 未安装时提供空操作装饰器，使直接运行不报错
if pytest is None:
    def _noop_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _noop_decorator
    fixture = _noop_decorator
    skip = lambda reason: None
else:
    fixture = pytest.fixture
    skip = pytest.skip

# ============================================================
# 配置
# ============================================================

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = (5, 120)  # 连接 5s，读取 120s

logger = logging.getLogger(__name__)


# ============================================================
# Fixtures
# ============================================================

@fixture(scope="session")
def api_base():
    """返回 API 基地址，并检查服务是否可用"""
    try:
        r = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        r.raise_for_status()
    except requests.ConnectionError:
        skip(f"API 服务未启动（{BASE_URL}），跳过测试")
    return BASE_URL


@fixture(scope="session")
def session(api_base):
    """创建一个测试会话，返回 session_id"""
    r = requests.post(f"{api_base}/api/v1/sessions", timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("session_id")


# ============================================================
# 测试用例
# ============================================================

class TestHealth:
    """健康检查 & 统计"""

    def test_health(self, api_base):
        """GET /api/v1/health 返回正常状态"""
        r = requests.get(f"{api_base}/api/v1/health", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert "chunks_count" in data
        assert data["chunks_count"] > 0

    def test_stats(self, api_base):
        """GET /api/v1/stats 返回系统统计"""
        r = requests.get(f"{api_base}/api/v1/stats", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data["total_chunks"] > 0
        assert data["total_drugs"] > 0


class TestSearch:
    """检索查询接口"""

    def test_search_basic(self, api_base):
        """GET /api/v1/search 基本检索"""
        r = requests.get(
            f"{api_base}/api/v1/search",
            params={"q": "人参", "top_k": 5},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "人参"
        assert data["total"] > 0
        assert len(data["results"]) <= 5

        # 验证结果结构
        for result in data["results"]:
            assert "chunk_id" in result
            assert "drug_name" in result
            assert "content" in result
            assert "score" in result

    def test_search_with_drug_filter(self, api_base):
        """GET /api/v1/search 按药品名过滤"""
        r = requests.get(
            f"{api_base}/api/v1/search",
            params={"q": "性味", "drug": "人参", "top_k": 10},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0
        # 所有结果应该包含"人参"相关
        for result in data["results"]:
            assert "人参" in result.get("drug_name", "")

    def test_search_empty_query(self, api_base):
        """GET /api/v1/search 空查询应返回 422"""
        r = requests.get(
            f"{api_base}/api/v1/search",
            params={"q": ""},
            timeout=TIMEOUT,
        )
        assert r.status_code == 422


class TestChat:
    """智能问答接口"""

    def test_chat_basic(self, api_base):
        """POST /api/v1/chat 基本问答"""
        r = requests.post(
            f"{api_base}/api/v1/chat",
            json={
                "question": "人参的性味归经是什么？",
                "return_sources": True,
            },
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 10
        assert "session_id" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

    def test_chat_with_session(self, api_base, session):
        """POST /api/v1/chat 多轮对话"""
        # 第一轮
        r1 = requests.post(
            f"{api_base}/api/v1/chat",
            json={
                "question": "黄芪的功效是什么？",
                "session_id": session,
                "return_sources": True,
            },
            timeout=TIMEOUT,
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["session_id"] == session

        # 第二轮（测试指代消解）
        r2 = requests.post(
            f"{api_base}/api/v1/chat",
            json={
                "question": "它的用法用量呢？",
                "session_id": session,
                "return_sources": True,
            },
            timeout=TIMEOUT,
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["session_id"] == session
        assert len(d2["answer"]) > 10
        # 指代消解后的查询应该与黄芪相关
        assert d2.get("resolved_query", "") != ""

    def test_chat_guard_off_topic(self, api_base):
        """POST /api/v1/chat 非医药问题被守卫拦截"""
        r = requests.post(
            f"{api_base}/api/v1/chat",
            json={
                "question": "今天天气怎么样？",
                "return_sources": False,
            },
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        # 守卫应该拦截并返回拒绝回答
        assert "抱歉" in data["answer"] or "药典" in data["answer"]
        assert len(data.get("sources", [])) == 0


class TestChatStream:
    """流式问答接口"""

    def test_stream_basic(self, api_base):
        """POST /api/v1/chat/stream 基本流式问答"""
        r = requests.post(
            f"{api_base}/api/v1/chat/stream",
            json={"question": "黄连性味归经是什么？"},
            stream=True,
            timeout=TIMEOUT,
        )
        assert r.status_code == 200

        events = []
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = json.loads(line[6:])
                events.append(data)

        # 应该有至少一个 content 事件和一个 done 事件
        content_events = [e for e in events if "content" in e and "done" not in e]
        done_events = [e for e in events if e.get("done") is True]

        assert len(content_events) > 0
        assert len(done_events) == 1

        # done 事件应该包含来源等元数据
        done = done_events[0]
        assert "session_id" in done
        assert "sources" in done
        assert "latency_ms" in done

        # 验证流式内容拼接后是有效文本
        full_content = "".join(e["content"] for e in content_events)
        assert len(full_content) > 10

    def test_stream_guard_rejection(self, api_base):
        """POST /api/v1/chat/stream 非医药问题流式拦截"""
        r = requests.post(
            f"{api_base}/api/v1/chat/stream",
            json={"question": "帮我写一首诗"},
            stream=True,
            timeout=TIMEOUT,
        )
        assert r.status_code == 200

        events = []
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))

        # 应该有一个 rejected 事件
        rejected_events = [e for e in events if e.get("rejected")]
        assert len(rejected_events) > 0
        assert "抱歉" in rejected_events[0]["content"] or "药典" in rejected_events[0]["content"]


class TestSessions:
    """会话管理接口"""

    def test_create_and_delete_session(self, api_base):
        """POST/DELETE /api/v1/sessions 创建和删除会话"""
        # 创建
        r = requests.post(f"{api_base}/api/v1/sessions", timeout=TIMEOUT)
        assert r.status_code == 200
        sid = r.json()["session_id"]
        assert sid

        # 获取
        r = requests.get(f"{api_base}/api/v1/sessions/{sid}", timeout=TIMEOUT)
        assert r.status_code == 200

        # 删除
        r = requests.delete(f"{api_base}/api/v1/sessions/{sid}", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_list_sessions(self, api_base):
        """GET /api/v1/sessions 列出会话"""
        r = requests.get(f"{api_base}/api/v1/sessions", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data


class TestDrugs:
    """药品列表接口"""

    def test_list_drugs(self, api_base):
        """GET /api/v1/drugs 药品列表"""
        r = requests.get(
            f"{api_base}/api/v1/drugs",
            params={"limit": 10},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0
        assert len(data["drugs"]) <= 10

    def test_search_drugs(self, api_base):
        """GET /api/v1/drugs?keyword=黄 搜索药品"""
        r = requests.get(
            f"{api_base}/api/v1/drugs",
            params={"keyword": "黄", "limit": 20},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0
        for drug in data["drugs"]:
            assert "黄" in drug.get("drug_name", "")


# ============================================================
# 直接运行入口
# ============================================================

def main():
    """直接运行时执行手动测试流程"""
    print("=" * 60)
    print("  药典 RAG 系统 - API 端到端测试")
    print(f"  目标: {BASE_URL}")
    print("=" * 60)

    # 检查服务是否可用
    try:
        r = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        r.raise_for_status()
        print(f"\n[OK] 服务可用: {r.json()['status']}")
    except Exception as e:
        print(f"\n[FAIL] 服务不可用: {e}")
        print("   请先启动 API: python scripts/run/run_api.py")
        sys.exit(1)

    passed = 0
    failed = 0
    failures = []

    def run_test(name, func):
        nonlocal passed, failed
        print(f"\n{'---' * 15}")
        print(f"测试: {name}")
        try:
            func()
            print(f"  [PASS]")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1
            failures.append((name, str(e)))

    # 1. 健康检查
    def test_health():
        r = requests.get(f"{BASE_URL}/api/v1/health", timeout=TIMEOUT)
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert d["chunks_count"] > 0
        print(f"  状态: {d['status']}, chunks: {d['chunks_count']}")
    run_test("健康检查", test_health)

    # 2. 统计信息
    def test_stats():
        r = requests.get(f"{BASE_URL}/api/v1/stats", timeout=TIMEOUT)
        d = r.json()
        assert d["total_chunks"] > 0
        print(f"  chunks: {d['total_chunks']}, drugs: {d['total_drugs']}")
    run_test("统计信息", test_stats)

    # 3. 检索查询
    def test_search():
        r = requests.get(f"{BASE_URL}/api/v1/search",
                         params={"q": "人参", "top_k": 5}, timeout=TIMEOUT)
        d = r.json()
        assert d["total"] > 0
        print(f"  检索到 {d['total']} 条结果")
    run_test("检索查询", test_search)

    # 4. 智能问答
    def test_chat():
        r = requests.post(f"{BASE_URL}/api/v1/chat",
                          json={"question": "人参的性味归经是什么？",
                                "return_sources": True}, timeout=TIMEOUT)
        d = r.json()
        assert len(d["answer"]) > 10
        assert len(d["sources"]) > 0
        print(f"  回答长度: {len(d['answer'])}, 来源: {len(d['sources'])} 条")
    run_test("智能问答", test_chat)

    # 5. 流式问答
    def test_stream():
        r = requests.post(f"{BASE_URL}/api/v1/chat/stream",
                          json={"question": "黄连的功效是什么？"},
                          stream=True, timeout=TIMEOUT)
        content = ""
        sources = []
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = json.loads(line[6:])
                if "content" in data and "done" not in data:
                    content += data["content"]
                if data.get("done"):
                    sources = data.get("sources", [])
        assert len(content) > 10
        assert len(sources) > 0
        print(f"  流式回答长度: {len(content)}, 来源: {len(sources)} 条")
    run_test("流式问答", test_stream)

    # 6. 守卫拦截
    def test_guard():
        r = requests.post(f"{BASE_URL}/api/v1/chat",
                          json={"question": "今天天气怎么样？",
                                "return_sources": False}, timeout=TIMEOUT)
        d = r.json()
        assert "抱歉" in d["answer"] or "药典" in d["answer"]
        print(f"  守卫拦截成功, 回答前20字: {d['answer'][:20]}...")
    run_test("守卫拦截", test_guard)

    # 7. 药品列表
    def test_drugs():
        r = requests.get(f"{BASE_URL}/api/v1/drugs",
                         params={"keyword": "黄", "limit": 10}, timeout=TIMEOUT)
        d = r.json()
        assert d["total"] > 0
        print(f"  匹配 {d['total']} 个药品")
    run_test("药品列表", test_drugs)

    # 8. 会话管理
    def test_session():
        r = requests.post(f"{BASE_URL}/api/v1/sessions", timeout=TIMEOUT)
        sid = r.json()["session_id"]
        r = requests.delete(f"{BASE_URL}/api/v1/sessions/{sid}", timeout=TIMEOUT)
        assert r.status_code == 200
        print(f"  创建 & 删除会话成功")
    run_test("会话管理", test_session)

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"  测试结果: {passed} 通过, {failed} 失败")
    if failures:
        print("  失败项:")
        for name, err in failures:
            print(f"    - {name}: {err}")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
