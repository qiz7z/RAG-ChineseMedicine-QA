# -*- coding: utf-8 -*-
"""API 端到端测试脚本"""
import sys
import io
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("  药典 RAG 系统 - API 端到端测试")
print("=" * 60)

# 1. 健康检查
print("\n[1] 健康检查 GET /api/v1/health")
r = requests.get(f"{BASE}/api/v1/health")
print(f"  状态码: {r.status_code}")
d = r.json()
print(f"  status: {d['status']}, model: {d['model']}, chunks: {d['chunks_count']}")

# 2. 统计信息
print("\n[2] 统计信息 GET /api/v1/stats")
r = requests.get(f"{BASE}/api/v1/stats")
d = r.json()
print(f"  总 chunk: {d['total_chunks']}, 药品数: {d['total_drugs']}")

# 3. 药品列表
print("\n[3] 药品列表 GET /api/v1/drugs?keyword=人参")
r = requests.get(f"{BASE}/api/v1/drugs", params={"keyword": "人参", "limit": 10})
d = r.json()
print(f"  匹配 {d['total']} 个: {d['drugs']}")

# 4. 智能问答（单轮）
print("\n[4] 智能问答 POST /api/v1/chat")
r = requests.post(f"{BASE}/api/v1/chat", json={
    "question": "当归的性味和功效是什么？",
    "return_sources": True,
})
d = r.json()
print(f"  状态码: {r.status_code}")
print(f"  会话ID: {d['session_id']}")
print(f"  对话轮次: {d['dialogue_turn']}")
print(f"  耗时: {d['latency_ms']}ms")
print(f"  回答:\n{d['answer'][:500]}")
print(f"  检索来源数: {len(d['sources'])}")
for i, s in enumerate(d['sources'][:3], 1):
    print(f"    {i}. {s['drug_name']} > {s['section']} (score={s['score']:.4f})")

session_id = d['session_id']

# 5. 多轮对话
print("\n[5] 多轮对话 POST /api/v1/chat (第2轮)")
r = requests.post(f"{BASE}/api/v1/chat", json={
    "question": "那它的用法用量呢？",
    "session_id": session_id,
    "return_sources": False,
})
d = r.json()
print(f"  消解后查询: {d['resolved_query']}")
print(f"  对话轮次: {d['dialogue_turn']}")
print(f"  回答:\n{d['answer'][:300]}")

# 6. 检索查询
print("\n[6] 检索查询 GET /api/v1/search")
r = requests.get(f"{BASE}/api/v1/search", params={"q": "黄芪 功效", "top_k": 3})
d = r.json()
print(f"  结果数: {d['total']}, 耗时: {d['latency_ms']}ms")
for i, s in enumerate(d['results'], 1):
    print(f"    {i}. {s['drug_name']} > {s['section']} (score={s['score']:.4f})")

# 7. 会话管理
print("\n[7] 会话列表 GET /api/v1/sessions")
r = requests.get(f"{BASE}/api/v1/sessions")
d = r.json()
print(f"  活跃会话: {d['total']} 个: {d['sessions']}")

print("\n[8] 会话详情 GET /api/v1/sessions/{id}")
r = requests.get(f"{BASE}/api/v1/sessions/{session_id}")
d = r.json()
print(f"  会话: {d['session_id']}, 轮次: {d['turn_count']}")
for h in d['history'][:2]:
    role = "用户" if h['role'] == 'user' else "助手"
    print(f"    {role}: {h['content'][:60]}...")

print("\n" + "=" * 60)
print("  ✅ API 全部端点测试通过!")
print("=" * 60)
