# -*- coding: utf-8 -*-
"""
中国药典智能问答系统 — Web UI 主应用
======================================
基于 Streamlit 构建的可视化交互界面。

功能页面：
  1. 智能问答 — 流式对话、多轮上下文、引用展示
  2. 检索查询 — 独立检索引擎、来源查看
  3. 药品浏览 — 药典收录药品列表
  4. 系统统计 — 索引/模型/性能指标
  5. 关于系统 — 架构说明与使用指南

启动方式：
  streamlit run src/webui/app.py
  或
  python scripts/run/run_webui.py
"""
import sys
import os
import time
import uuid
from pathlib import Path

import streamlit as st

# 确保 src 目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webui.styles import GLOBAL_CSS
from webui.api_client import APIClient
from webui.components import (
    ICONS,
    render_header,
    render_hero,
    render_status_card,
    render_source_card,
    render_citations,
    render_disclaimer,
    render_stat_card,
    render_latency_badge,
    render_typing_indicator,
    render_empty_state,
    render_drug_badge,
    render_consistency_warning,
    render_feature_card,
    render_user_message,
    render_assistant_meta,
)

# ============================================================
# 页面配置
# ============================================================

# 静态资源目录（SVG 头像/图标）
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_SVG = ASSETS_DIR / "logo.svg"
AVATAR_AI_SVG = ASSETS_DIR / "avatar-ai.svg"
AVATAR_USER_SVG = ASSETS_DIR / "avatar-user.svg"


def _svg_data_uri(path: Path) -> str:
    """读取 SVG 文件并转为 base64 data URI（用于 HTML <img>，规避静态服务问题）"""
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode()


LOGO_DATA_URI = _svg_data_uri(LOGO_SVG)

st.set_page_config(
    page_title="中国药典智能问答系统",
    page_icon=str(LOGO_SVG),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "基于 RAG 架构的《中国药典》2020年版一部智能问答系统",
    },
)

# 注入全局 CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ============================================================
# 全局初始化
# ============================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def resolve_user_id() -> str:
    """匿名用户标识：URL 参数 ?uid= 优先，没有则生成并写回地址栏。

    ⚠️ 这不是认证，只用于把会话/记忆归到不同身份下：
    同一浏览器刷新仍是同一身份，换无痕窗口则是新的身份。
    """
    uid = st.query_params.get("uid")
    if not uid:
        uid = "u_" + uuid.uuid4().hex[:8]
        st.query_params["uid"] = uid
    return uid


@st.cache_resource
def get_api_client(user_id: str) -> APIClient:
    """获取 API 客户端（按 user_id 缓存，各自持有独立的请求会话）"""
    return APIClient(API_BASE_URL, user_id=user_id)


def init_session_state():
    """初始化 Session State"""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # 聊天历史 [{"role": ..., "content": ..., "sources": ...}]
    if "session_id" not in st.session_state:
        st.session_state.session_id = None  # 后端会话 ID
    if "current_page" not in st.session_state:
        st.session_state.current_page = "智能问答"
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = {}  # {msg_index: bool} 控制每条消息的来源展开
    if "started" not in st.session_state:
        st.session_state.started = False  # 是否已进入对话（用于控制欢迎区）
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "active_key" not in st.session_state:
        st.session_state.active_key = None
    if not st.session_state.conversations:
        _new_conversation()


def _new_conversation():
    """新建本地会话并切换过去；后端会话在首次提问时懒创建"""
    conv = {
        "key": f"c_{uuid.uuid4().hex[:6]}",
        "session_id": None,
        "title": "新会话",
        "messages": [],
    }
    st.session_state.conversations.insert(0, conv)
    st.session_state.active_key = conv["key"]
    st.session_state.messages = conv["messages"]
    st.session_state.session_id = None
    st.session_state.show_sources = {}
    st.session_state.started = False


def get_active_conversation():
    """返回当前会话 dict（不存在则 None）"""
    key = st.session_state.get("active_key")
    for conv in st.session_state.conversations:
        if conv["key"] == key:
            return conv
    return None


def _switch_conversation(key: str):
    """切换到指定会话，恢复其消息与后端 session"""
    for conv in st.session_state.conversations:
        if conv["key"] == key:
            st.session_state.active_key = key
            st.session_state.messages = conv["messages"]
            st.session_state.session_id = conv["session_id"]
            st.session_state.show_sources = {}
            st.session_state.started = bool(conv["messages"])
            return


def _delete_conversation(conv: dict):
    """删除会话（后端 + 本地），必要时回退到其他会话"""
    if conv.get("session_id"):
        try:
            api.delete_session(conv["session_id"])
        except Exception:
            pass
    st.session_state.conversations = [
        c for c in st.session_state.conversations if c["key"] != conv["key"]
    ]
    if st.session_state.get("active_key") == conv["key"]:
        if st.session_state.conversations:
            _switch_conversation(st.session_state.conversations[0]["key"])
        else:
            _new_conversation()


def _mark_started(*_args):
    """chat_input 提交回调：在脚本重跑前标记进入对话，欢迎区当帧即消失"""
    st.session_state.started = True


def _home_submit(*args):
    """
    首页内联输入框的提交回调。

    首页与对话页用的是两个不同位置的 chat_input，重跑后组件实例不同，
    值会丢失，因此在回调里先把内容写入 pending_question 再交给对话页。
    """
    st.session_state.started = True
    value = args[0] if args else None
    if not isinstance(value, str):
        value = st.session_state.get("home_chat_input")
    if isinstance(value, str) and value.strip():
        st.session_state.pending_question = value.strip()


init_session_state()
USER_ID = resolve_user_id()
api = get_api_client(USER_ID)


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">'
        f'<img class="sidebar-logo" src="{LOGO_DATA_URI}" alt="中国药典智能问答系统" />'
        f'<div class="sidebar-title">中国药典</div>'
        f'<div class="sidebar-sub">智能问答系统 · v1.0</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # 当前匿名身份（换无痕窗口即换身份，用于验证会话/记忆隔离）
    st.caption(f"当前身份：{USER_ID}")

    # 页面导航（按钮式：不依赖 radio 的内部 DOM，选中态完全可控）
    NAV_ITEMS = ["智能问答", "检索查询", "药品浏览", "系统统计", "关于系统"]
    NAV_KEYS = {
        "智能问答": "nav_chat",
        "检索查询": "nav_search",
        "药品浏览": "nav_leaf",
        "系统统计": "nav_activity",
        "关于系统": "nav_info",
    }

    with st.container(key="sidenav"):
        for item in NAV_ITEMS:
            is_active = st.session_state.current_page == item
            key = NAV_KEYS[item] + ("_active" if is_active else "")
            if st.button(
                item,
                key=key,
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.current_page = item
                st.rerun()

    page = st.session_state.current_page

    st.markdown("---")

    # API 连接状态卡片
    st.markdown("#### 系统状态")
    try:
        health = api.health()
        render_status_card(health)
    except Exception:
        render_status_card(None)

    st.markdown("---")

    # 会话管理（仅问答页面显示）
    if page == "智能问答":
        st.markdown("#### 会话管理")
        if st.button("＋ 新建会话", use_container_width=True):
            _new_conversation()
            st.rerun()

        active = get_active_conversation()
        for conv in st.session_state.conversations[:8]:
            is_active = active is not None and conv["key"] == active["key"]
            title = (conv.get("title") or "新会话")[:12]
            label = f"● {title}" if is_active else title
            c1, c2 = st.columns([5, 1], gap="small")
            if c1.button(label, key=f"conv_{conv['key']}", use_container_width=True):
                _switch_conversation(conv["key"])
                st.rerun()
            if c2.button("", icon=":material/delete:", key=f"del_{conv['key']}", help="删除会话"):
                _delete_conversation(conv)
                st.rerun()

        if active and active.get("session_id"):
            turns = len(active["messages"]) // 2
            st.caption(f"当前会话 `{active['session_id'][:16]}…` · {turns} 轮")
        else:
            st.caption("新会话：提问后自动创建后端会话")


# ============================================================
# 页面 1：智能问答
# ============================================================

def render_chat_page():
    """渲染智能问答页面"""
    # 保证 messages 始终指向当前会话的消息列表（rerun 后引用可能失效）
    _conv = get_active_conversation()
    if _conv is not None and st.session_state.messages is not _conv["messages"]:
        _conv["messages"] = st.session_state.messages

    pending = st.session_state.pop("pending_question", None)

    # 首页：无历史、未进入对话、且本次没有待发送问题时才展示
    welcome = (
        not st.session_state.messages
        and not st.session_state.started
        and not pending
    )

    if welcome:
        # 首页整体垂直居中：hero → 输入框 → 示例问题，一屏内不滚动
        with st.container(key="home"):
            render_hero(LOGO_DATA_URI)

            # 在容器内调用 → chat_input 内联显示在页面中间，而不是固定底部
            user_input = st.chat_input(
                "给《中国药典》提问…",
                key="home_chat_input",
                on_submit=_home_submit,
            )

            example_questions = [
                "人参的性味归经是什么？",
                "黄芪的功能与主治有哪些？",
                "当归的用法用量是怎样的？",
                "麻黄的注意事项是什么？",
            ]
            # 容器 key → CSS 类 .st-key-example_grid，样式作用域精确可控
            with st.container(key="example_grid"):
                for row_start in range(0, len(example_questions), 2):
                    cols = st.columns(2, gap="small")
                    for i, q in enumerate(example_questions[row_start:row_start + 2]):
                        if cols[i].button(q, key=f"example_{row_start + i}", use_container_width=True):
                            st.session_state.pending_question = q
                            st.session_state.started = True
                            st.rerun()

    # 显示聊天历史：用户消息右对齐、助手消息左对齐卡片
    with st.container(key="chat_thread"):
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                render_user_message(msg["content"])
                continue

            with st.container(key=f"amsg_h_{idx}"):
                st.markdown(msg["content"])

                if msg.get("citations"):
                    render_citations(msg["citations"])

                if msg.get("consistency_issues"):
                    render_consistency_warning(msg["consistency_issues"])

                if msg.get("sources"):
                    with st.expander(f"查看 {len(msg['sources'])} 条药典来源", expanded=False):
                        max_score = max((s.get("score", 0) for s in msg["sources"]), default=0)
                        for j, src in enumerate(msg["sources"], 1):
                            render_source_card(src, j, max_score)

                render_assistant_meta(
                    latency_ms=msg.get("latency_ms", 0),
                    dialogue_turn=msg.get("dialogue_turn", 0),
                    original_query=msg.get("original_query", ""),
                    resolved_query=msg.get("resolved_query", ""),
                )

    # 对话态：输入框固定在页面底部（首页已在上方内联渲染）
    if not welcome:
        # 流式生成期间脚本尚未结束，Streamlit 会保留上一帧的首页 DOM（置灰），
        # 这里主动注入 CSS 立即隐藏，避免提问后 hero / 示例问题残留在页面上方
        st.markdown(
            '<style>[class*="st-key-home"]{display:none !important;}</style>',
            unsafe_allow_html=True,
        )
        user_input = st.chat_input(
            "给《中国药典》提问…",
            key="chat_input",
            on_submit=_mark_started,
        )

    if pending and not user_input:
        user_input = pending
        st.session_state.started = True

    answered = False
    if user_input:
        # 显示用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        render_user_message(user_input)

        # 流式生成回答（左侧卡片气泡，头像由 CSS 伪元素绘制）
        with st.container(key="amsg_live"):
            placeholder = st.empty()
            full_response = ""

            # 显示打字动画
            with placeholder:
                render_typing_indicator()

            try:
                # 流式获取回答
                collected_content = ""
                final_session_id = st.session_state.session_id
                sources = []
                citations = []
                latency_ms = 0
                consistency_issues = []
                resolved_query = user_input
                dialogue_turn = 0

                # 流式节流：每个分片都重排整段 Markdown 会持续改变内容高度，
                # 长回答时滚动位置会被反复拽动，这里每 3 个分片刷新一次。
                stream_tick = 0
                for event in api.chat_stream(
                    user_input,
                    session_id=st.session_state.session_id,
                ):
                    if "content" in event and "rejected" not in event:
                        collected_content += event["content"]
                        stream_tick += 1
                        if stream_tick % 3 == 0:
                            placeholder.markdown(collected_content + "▌")
                    elif "rejected" in event:
                        collected_content = event["content"]
                        placeholder.markdown(collected_content)
                        # 保存拒绝回答到历史，保持消息一致性
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": collected_content,
                            "sources": [],
                            "citations": [],
                            "latency_ms": 0,
                            "dialogue_turn": 0,
                            "resolved_query": user_input,
                            "original_query": user_input,
                            "consistency_issues": [],
                        })
                        # 拒绝回答同样要绑定会话，避免下次提问又新建后端会话
                        st.session_state.session_id = event.get(
                            "session_id", st.session_state.session_id
                        )
                        conv = get_active_conversation()
                        if conv is not None:
                            conv["session_id"] = st.session_state.session_id
                            if not conv.get("title") or conv["title"] == "新会话":
                                conv["title"] = user_input[:12]
                            conv["messages"] = st.session_state.messages
                        return
                    elif "done" in event:
                        final_session_id = event.get("session_id", final_session_id)
                        # 流式结束时获取来源、引用等元数据
                        sources = event.get("sources", [])
                        citations = event.get("citations", [])
                        latency_ms = event.get("latency_ms", 0)
                        consistency_issues = event.get("consistency_issues", [])
                        resolved_query = event.get("resolved_query", user_input)
                        dialogue_turn = event.get("dialogue_turn", 0)
                    elif "error" in event:
                        placeholder.error(f"生成失败: {event['error']}")
                        return

                full_response = collected_content
                placeholder.markdown(full_response)

                # 更新会话 ID，并回写当前会话条目（标题取首问前 12 字）
                st.session_state.session_id = final_session_id
                conv = get_active_conversation()
                if conv is not None:
                    conv["session_id"] = final_session_id
                    if not conv.get("title") or conv["title"] == "新会话":
                        conv["title"] = user_input[:12]
                    conv["messages"] = st.session_state.messages

                # 显示引用
                if citations:
                    render_citations(citations)

                # 显示一致性警告
                if consistency_issues:
                    render_consistency_warning(consistency_issues)

                # 显示来源（可折叠）
                if sources:
                    with st.expander(f"查看 {len(sources)} 条药典来源", expanded=False):
                        max_score = max((s.get("score", 0) for s in sources), default=0)
                        for j, src in enumerate(sources, 1):
                            render_source_card(src, j, max_score)

                # 元信息（耗时 / 轮次 / 指代消解 / 安全提示）
                render_assistant_meta(
                    latency_ms=latency_ms,
                    dialogue_turn=dialogue_turn,
                    original_query=user_input,
                    resolved_query=resolved_query,
                )

                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "citations": citations,
                    "latency_ms": latency_ms,
                    "dialogue_turn": dialogue_turn if latency_ms else len(st.session_state.messages) // 2,
                    "resolved_query": resolved_query,
                    "original_query": user_input,
                    "consistency_issues": consistency_issues,
                })
                answered = True

            except Exception as e:
                # 如果流式失败，尝试同步接口
                try:
                    placeholder.info("流式连接失败，切换到同步模式...")
                    result = api.chat(
                        user_input,
                        session_id=st.session_state.session_id,
                        return_sources=True,
                    )

                    full_response = result["answer"]
                    st.session_state.session_id = result["session_id"]

                    placeholder.markdown(full_response)

                    # 显示来源
                    sources = result.get("sources", [])
                    citations = result.get("citations", [])
                    latency_ms = result.get("latency_ms", 0)
                    consistency_issues = result.get("consistency_issues", [])

                    if citations:
                        render_citations(citations)

                    if consistency_issues:
                        render_consistency_warning(consistency_issues)

                    if sources:
                        with st.expander(f"查看 {len(sources)} 条药典来源", expanded=False):
                            max_score = max((s.get("score", 0) for s in sources), default=0)
                            for j, src in enumerate(sources, 1):
                                render_source_card(src, j, max_score)

                    render_assistant_meta(
                        latency_ms=latency_ms,
                        dialogue_turn=result.get("dialogue_turn", 1),
                        original_query=user_input,
                        resolved_query=result.get("resolved_query", ""),
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                        "citations": citations,
                        "latency_ms": latency_ms,
                        "dialogue_turn": result.get("dialogue_turn", 1),
                        "resolved_query": result.get("resolved_query", user_input),
                        "original_query": user_input,
                        "consistency_issues": consistency_issues,
                    })
                    answered = True

                except Exception as e2:
                    placeholder.error(f"问答失败: {e2}")

    # 回答完成后再 rerun 一次：让侧边栏的会话标题 / 会话 ID / 轮次即时刷新
    if answered:
        st.rerun()


# ============================================================
# 页面 2：检索查询
# ============================================================

def render_search_page():
    """渲染检索查询页面"""
    render_header(
        "检索查询",
        "直接使用混合检索引擎（向量 + BM25 + RRF）查询药典内容",
        eyebrow="SEARCH · HYBRID RETRIEVAL",
    )

    # 查询输入
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        query = st.text_input(
            "查询关键词",
            placeholder="例如：人参 性味归经",
            label_visibility="collapsed",
        )

    with col2:
        drug_filter = st.text_input(
            "药品过滤",
            placeholder="药品名（可选）",
            label_visibility="collapsed",
        )

    with col3:
        top_k = st.number_input(
            "返回数量",
            min_value=1,
            max_value=30,
            value=5,
            label_visibility="collapsed",
        )

    # 章节过滤
    section_options = [
        "", "来源", "性状", "鉴别", "检查", "浸出物", "含量测定",
        "性味与归经", "功能与主治", "用法与用量", "注意", "炮制",
        "贮藏", "规格", "制剂",
    ]
    section_filter = st.selectbox("章节过滤（可选）", section_options, index=0)

    if st.button("检索", type="primary", use_container_width=False):
        if not query.strip():
            st.warning("请输入查询关键词")
            return

        with st.spinner("正在检索..."):
            try:
                result = api.search(
                    q=query,
                    drug=drug_filter if drug_filter else None,
                    section=section_filter if section_filter else None,
                    top_k=top_k,
                )

                st.markdown("---")

                # 结果统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_stat_card(str(result["total"]), "检索结果数")
                with col2:
                    render_stat_card(f"{result['latency_ms']}", "耗时(ms)")
                with col3:
                    avg_score = (
                        sum(r["score"] for r in result["results"]) / len(result["results"])
                        if result["results"] else 0
                    )
                    render_stat_card(f"{avg_score:.4f}", "平均相似度")

                st.markdown("---")

                # 结果列表
                if result["results"]:
                    max_score = max((r.get("score", 0) for r in result["results"]), default=0)
                    for i, r in enumerate(result["results"], 1):
                        render_source_card(r, i, max_score)
                else:
                    render_empty_state("未找到相关结果", "请尝试其他关键词或调整过滤条件")

            except Exception as e:
                st.error(f"检索失败: {e}")


# ============================================================
# 页面 3：药品浏览
# ============================================================

def render_drugs_page():
    """渲染药品浏览页面"""
    render_header(
        "药品浏览",
        "浏览《中国药典》2020年版一部收录的全部药材和饮片",
        eyebrow="PHARMACOPOEIA · 2020",
    )

    # 搜索框
    col1, col2 = st.columns([3, 1])

    with col1:
        keyword = st.text_input(
            "搜索药品",
            placeholder="输入药品名（支持模糊匹配）",
            label_visibility="collapsed",
        )

    with col2:
        limit = st.number_input(
            "显示数量",
            min_value=10,
            max_value=1000,
            value=100,
            step=50,
            label_visibility="collapsed",
        )

    if st.button("查询", type="primary"):
        with st.spinner("正在加载药品列表..."):
            try:
                result = api.list_drugs(
                    keyword=keyword if keyword else None,
                    limit=limit,
                )

                total = result["total"]
                drugs = result["drugs"]

                st.markdown(f"**共找到 {total} 种药品**")
                st.markdown("---")

                if drugs:
                    # 使用网格布局展示药品标签
                    cols_per_row = 6
                    for i in range(0, len(drugs), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, drug in enumerate(drugs[i:i + cols_per_row]):
                            with cols[j]:
                                render_drug_badge(drug)

                    # 如果结果太多，提示
                    if total > limit:
                        st.info(f"仅显示前 {limit} 种，共 {total} 种药品。请增加显示数量或使用搜索过滤。")
                else:
                    render_empty_state("未找到匹配的药品", "请尝试其他关键词")

            except Exception as e:
                st.error(f"查询失败: {e}")
    else:
        # 默认显示提示
        render_empty_state(
            "点击「查询」按钮浏览全部药品",
            "也可以在上方输入关键词进行搜索",
        )


# ============================================================
# 页面 4：系统统计
# ============================================================

def render_stats_page():
    """渲染系统统计页面"""
    render_header(
        "系统统计",
        "药典知识库索引状态与系统运行指标",
        eyebrow="SYSTEM · METRICS",
    )

    try:
        stats = api.stats()
        health = api.health()

        # 核心指标
        st.markdown("### 核心指标")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_stat_card(str(stats["total_chunks"]), "知识切片(Chunks)", "doc")
        with col2:
            render_stat_card(str(stats["total_drugs"]), "收录药品", "leaf")
        with col3:
            render_stat_card(str(health.get("chunks_count", 0)), "向量索引", "database")
        with col4:
            render_stat_card(health.get("model", "N/A"), "LLM 模型", "chip")

        st.markdown("---")

        # 索引状态
        st.markdown("### 索引状态")
        idx_status = stats.get("index_status", {})
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("向量索引", idx_status.get("vector_index", "N/A"))
        with col2:
            st.metric("BM25 索引", idx_status.get("bm25_index", "N/A"))
        with col3:
            st.metric("元数据库", idx_status.get("metadata_store", "N/A"))

        st.markdown("---")

        # 按分类统计
        st.markdown("### 按分类统计")
        by_category = stats.get("by_category", {})
        if by_category:
            cat_data = [{"分类": k, "数量": v} for k, v in sorted(by_category.items(), key=lambda x: -x[1])]
            st.dataframe(cat_data, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 按 Chunk 类型统计
        st.markdown("### 按 Chunk 类型统计")
        by_chunk_type = stats.get("by_chunk_type", {})
        if by_chunk_type:
            type_data = [{"类型": k, "数量": v} for k, v in sorted(by_chunk_type.items(), key=lambda x: -x[1])]
            st.dataframe(type_data, use_container_width=True, hide_index=True)
        else:
            st.info("暂无 Chunk 类型统计数据")

    except Exception as e:
        st.error(f"获取统计信息失败: {e}")
        st.caption("请确保后端 API 服务正在运行")


# ============================================================
# 页面 5：关于系统
# ============================================================

def render_about_page():
    """渲染关于系统页面"""
    render_header(
        "关于系统",
        "中国药典智能问答系统 — 基于 RAG 架构的知识检索与生成",
        eyebrow="ABOUT · RAG PIPELINE",
    )

    # 模型名以实际配置为准（.env 中的 LLM_MODEL），不在页面里硬编码
    try:
        from config import LLM_MODEL as _llm_model
    except ImportError:
        _llm_model = "N/A"

    st.markdown(
        f"""
        ### 系统简介

        本系统基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 架构，
        以《中国药典》2020年版一部为知识来源，为用户提供准确、可溯源的中医药智能问答服务。

        ### 技术架构（五阶段流水线）

        | 阶段 | 模块 | 说明 |
        |------|------|------|
        | **阶段一** | 数据工程 | .docx 解析 → OCR 清洗 → 结构化提取 → 智能切片 |
        | **阶段二** | 索引构建 | BGE 向量化 + Chroma 存储 + BM25 关键词索引 |
        | **阶段三** | 检索引擎 | 向量+BM25 混合检索 → RRF 融合 → BGE 重排 |
        | **阶段四** | 生成引擎 | 指代消解 → Prompt 工程 → {_llm_model} 生成 → 后处理 |
        | **阶段五** | API 服务 | FastAPI RESTful + SSE 流式 + 会话管理 |

        ### 数据来源

        - **知识库**：《中华人民共和国药典》2020年版一部
        - **收录范围**：药材和饮片、植物油脂和提取物、成方制剂和单味制剂
        - **切片策略**：三级切分（药品级 → 章节级 → 滑动窗口级）

        ### 技术栈

        | 组件 | 选型 |
        |------|------|
        | Embedding | BAAI/bge-large-zh-v1.5 |
        | 向量数据库 | Chroma |
        | 关键词检索 | Rank BM25 |
        | 重排模型 | BAAI/bge-reranker-v2-m3 |
        | LLM | {_llm_model} |
        | 后端 | FastAPI + Uvicorn |
        | 前端 | Streamlit |

        ### 使用指南

        1. **智能问答**：在输入框中输入问题，系统会自动检索药典并生成回答
        2. **多轮对话**：系统支持上下文理解，可以追问"它的用法用量呢？"
        3. **检索来源**：点击回答下方的"检索来源"可查看引用的药典原文
        4. **检索查询**：在"检索查询"页面可直接使用检索引擎，无需 LLM 生成
        5. **药品浏览**：在"药品浏览"页面可查看药典收录的所有药品
        """
    )

    # 核心特性卡片网格
    st.markdown("### 核心特性")
    features = [
        ("git-branch", "混合检索", "向量语义检索 + BM25 关键词检索，RRF 算法融合排序"),
        ("spark", "精准重排", "BGE-Reranker 跨编码器对候选结果二次排序"),
        ("history", "多轮对话", "基于 LLM 的指代消解，支持追问与上下文继承"),
        ("bookmark", "引用标注", "回答自动标注药典出处，确保可溯源性"),
        ("check", "一致性校验", "数值/单位一致性检查，降低 LLM 幻觉风险"),
        ("shield", "用药安全", "自动附加用药安全提醒，引导遵医嘱"),
    ]
    for row_start in range(0, len(features), 2):
        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(features[row_start:row_start + 2]):
            with cols[i]:
                render_feature_card(icon, title, desc)

    st.markdown("---")

    st.markdown(
        f"""
        <div class="medical-disclaimer">
            {ICONS['shield']}
            <div>
                <strong>免责声明</strong>：本系统提供的信息仅供参考和学习使用，
                不构成医疗建议。具体用药请遵医嘱，切勿自行用药。
                系统回答基于《中国药典》原文，但可能存在检索或生成偏差。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 页面路由
# ============================================================

PAGES = {
    "智能问答": render_chat_page,
    "检索查询": render_search_page,
    "药品浏览": render_drugs_page,
    "系统统计": render_stats_page,
    "关于系统": render_about_page,
}

# 渲染当前页面
current = st.session_state.current_page
if current in PAGES:
    PAGES[current]()
