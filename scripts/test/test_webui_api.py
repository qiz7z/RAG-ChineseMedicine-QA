# -*- coding: utf-8 -*-
"""Web UI API 客户端功能测试"""
import sys
import io
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from webui.api_client import APIClient

api = APIClient("http://127.0.0.1:8000")

print("=" * 60)
print("  Web UI API 客户端功能测试")
print("=" * 60)

# 1. Health check
print("\n[1] Health Check")
h = api.health()
print(f"  status={h['status']}, model={h['model']}")
print(f"  chunks={h['chunks_count']}, drugs={h['drug_count']}")

# 2. Stats
print("\n[2] Stats")
s = api.stats()
print(f"  total_chunks={s['total_chunks']}, total_drugs={s['total_drugs']}")
print(f"  by_category={list(s['by_category'].keys())[:5]}...")

# 3. Drug list
print("\n[3] Drug List (keyword=人参)")
d = api.list_drugs(keyword="人参", limit=5)
print(f"  total={d['total']}, showing: {d['drugs']}")

# 4. Search
print("\n[4] Search (q=黄芪 功效)")
r = api.search(q="黄芪 功效", top_k=3)
print(f"  total={r['total']}, latency={r['latency_ms']}ms")
for i, item in enumerate(r["results"], 1):
    print(f"    {i}. {item['drug_name']} > {item['section']} (score={item['score']:.4f})")

# 5. Chat (sync)
print("\n[5] Chat (sync) - 人参的性味归经")
c = api.chat("人参的性味归经是什么？", return_sources=True)
print(f"  session={c['session_id'][:20]}...")
print(f"  latency={c['latency_ms']}ms, turn={c['dialogue_turn']}")
print(f"  answer={c['answer'][:200]}...")
print(f"  sources={len(c['sources'])} items")
print(f"  citations={c['citations']}")

# 6. Multi-turn chat
print("\n[6] Multi-turn Chat (指代消解)")
c2 = api.chat("那它的功能主治呢？", session_id=c["session_id"], return_sources=False)
print(f"  resolved_query={c2['resolved_query']}")
print(f"  turn={c2['dialogue_turn']}")
print(f"  answer={c2['answer'][:200]}...")

# 7. Chat stream
print("\n[7] Chat Stream - 当归功效")
content = ""
session_id = None
for event in api.chat_stream("当归有什么功效？"):
    if "content" in event:
        content += event["content"]
    elif "done" in event:
        session_id = event.get("session_id")
print(f"  session={session_id[:20] if session_id else 'N/A'}...")
print(f"  streamed_content_length={len(content)}")
print(f"  content_preview={content[:200]}...")

# 8. Session management
print("\n[8] Session Management")
sessions = api.list_sessions()
print(f"  active_sessions={sessions['total']}")
if c["session_id"]:
    info = api.get_session(c["session_id"])
    print(f"  session={info['session_id'][:20]}..., turns={info['turn_count']}")

print("\n" + "=" * 60)
print("  All Web UI API tests passed!")
print("=" * 60)
