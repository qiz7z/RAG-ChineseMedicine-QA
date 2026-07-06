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
from pathlib import Path

import streamlit as st

# 确保 src 目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webui.styles import GLOBAL_CSS, COLORS
from webui.api_client import APIClient
from webui.components import (
    render_header,
    render_source_card,
    render_citations,
    render_disclaimer,
    render_stat_card,
    render_latency_badge,
    render_typing_indicator,
    render_empty_state,
    render_drug_badge,
    render_consistency_warning,
)

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="中国药典智能问答系统",
    page_icon="🌿",
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


@st.cache_resource
def get_api_client() -> APIClient:
    """获取全局 API 客户端单例"""
    return APIClient(API_BASE_URL)


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


init_session_state()
api = get_api_client()


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem; line-height: 1;">🌿</div>
            <h2 style="margin: 0.5rem 0 0 0; color: {COLORS['primary_dark']};">中国药典</h2>
            <p style="color: {COLORS['text_secondary']}; font-size: 0.85rem; margin: 0.25rem 0;">
                智能问答系统 v1.0
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # 页面导航
    page = st.radio(
        "功能导航",
        ["智能问答", "检索查询", "药品浏览", "系统统计", "关于系统"],
        label_visibility="collapsed",
    )
    st.session_state.current_page = page

    st.markdown("---")

    # API 连接状态
    st.markdown("#### 系统状态")
    try:
        health = api.health()
        st.success(
            f"服务正常\n\n"
            f"模型: {health.get('model', 'N/A')}\n"
            f"索引: {health.get('chunks_count', 0)} chunks\n"
            f"药品: {health.get('drug_count', 0)} 种"
        )
    except Exception:
        st.error(f"无法连接 API\n地址: {API_BASE_URL}")
        st.caption("请先启动后端服务：")
        st.code("python scripts/run/run_api.py", language="bash")

    st.markdown("---")

    # 会话管理（仅问答页面显示）
    if page == "智能问答":
        st.markdown("#### 会话管理")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("新建会话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.session_state.show_sources = {}
                st.rerun()
        with col2:
            if st.session_state.session_id:
                if st.button("查看历史", use_container_width=True):
                    try:
                        info = api.get_session(st.session_state.session_id)
                        st.info(f"会话: {info['session_id'][:16]}...\n轮次: {info['turn_count']}")
                    except Exception as e:
                        st.error(f"获取失败: {e}")

        if st.session_state.session_id:
            st.caption(f"当前会话: `{st.session_state.session_id[:20]}...`")
            st.caption(f"对话轮次: {len(st.session_state.messages) // 2}")
        else:
            st.caption("尚未开始对话")


# ============================================================
# 页面 1：智能问答
# ============================================================

def render_chat_page():
    """渲染智能问答页面"""
    render_header(
        "智能问答",
        "基于《中国药典》2020年版一部的 RAG 智能问答，支持多轮对话与流式输出",
    )

    # 示例问题
    if not st.session_state.messages:
        st.markdown("#### 试试这些问题")
        example_questions = [
            "人参的性味归经是什么？",
            "黄芪的功能与主治有哪些？",
            "当归的用法用量是怎样的？",
            "含有黄连素的药材有哪些？",
            "麻黄的注意事项是什么？",
        ]
        cols = st.columns(len(example_questions))
        for i, q in enumerate(example_questions):
            if cols[i].button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    # 显示聊天历史
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍⚕️" if role == "user" else "🌿"):
            st.markdown(msg["content"])

            # 显示引用
            if role == "assistant" and msg.get("citations"):
                render_citations(msg["citations"])

            # 显示一致性警告
            if role == "assistant" and msg.get("consistency_issues"):
                render_consistency_warning(msg["consistency_issues"])

            # 显示来源（可折叠）
            if role == "assistant" and msg.get("sources"):
                with st.expander(f"📎 检索来源（{len(msg['sources'])} 条）", expanded=False):
                    for j, src in enumerate(msg["sources"], 1):
                        render_source_card(src, j)

            # 显示元信息
            if role == "assistant" and msg.get("latency_ms"):
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    render_latency_badge(msg["latency_ms"])
                with col2:
                    if msg.get("dialogue_turn"):
                        st.caption(f"第 {msg['dialogue_turn']} 轮")
                with col3:
                    if msg.get("resolved_query") and msg["resolved_query"] != msg.get("original_query"):
                        st.caption(f"指代消解: {msg['original_query']} → {msg['resolved_query']}")

    # 处理待发送问题（来自示例按钮）
    pending = st.session_state.pop("pending_question", None)

    # 输入框
    user_input = st.chat_input("输入您的问题...")

    # 优先处理 pending 问题
    if pending and not user_input:
        user_input = pending

    if user_input:
        # 显示用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(user_input)

        # 流式生成回答
        with st.chat_message("assistant", avatar="🌿"):
            placeholder = st.empty()
            full_response = ""

            # 显示打字动画
            with placeholder:
                render_typing_indicator()

            try:
                # 流式获取回答
                collected_content = ""
                final_session_id = st.session_state.session_id

                for event in api.chat_stream(
                    user_input,
                    session_id=st.session_state.session_id,
                ):
                    if "content" in event:
                        collected_content += event["content"]
                        # 实时更新显示
                        placeholder.markdown(collected_content + "▌")
                    elif "done" in event:
                        final_session_id = event.get("session_id", final_session_id)
                    elif "error" in event:
                        placeholder.error(f"生成失败: {event['error']}")
                        return

                full_response = collected_content
                placeholder.markdown(full_response)

                # 更新会话 ID
                st.session_state.session_id = final_session_id

                # 获取完整回答信息（含来源、引用等）
                # 流式接口不返回来源，需要额外调用同步接口获取
                # 但为了避免重复调用 LLM，我们仅展示流式结果
                # 来源信息通过非流式重试获取代价太大，这里简化处理
                sources = []
                citations = []
                latency_ms = 0
                consistency_issues = []

                # 如果用户需要来源，可以在这里调用同步接口
                # 但会增加延迟，所以我们提供一个开关
                if st.session_state.get("fetch_sources", True):
                    try:
                        # 使用同步接口获取来源（只获取检索来源，不重新生成）
                        # 实际上这里我们用 search 接口来获取相关来源
                        # 更好的方式是后端在流式结束后返回来源
                        pass  # 流式模式下暂不获取来源，避免重复调用
                    except Exception:
                        pass

                # 显示安全提醒
                render_disclaimer()

                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "citations": citations,
                    "latency_ms": latency_ms,
                    "dialogue_turn": len(st.session_state.messages) // 2,
                    "resolved_query": user_input,
                    "original_query": user_input,
                    "consistency_issues": consistency_issues,
                })

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
                        with st.expander(f"📎 检索来源（{len(sources)} 条）", expanded=False):
                            for j, src in enumerate(sources, 1):
                                render_source_card(src, j)

                    render_disclaimer()

                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        render_latency_badge(latency_ms)
                    with col2:
                        st.caption(f"第 {result.get('dialogue_turn', 1)} 轮")
                    with col3:
                        if result.get("resolved_query") and result["resolved_query"] != user_input:
                            st.caption(f"指代消解: {user_input} → {result['resolved_query']}")

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

                except Exception as e2:
                    placeholder.error(f"问答失败: {e2}")


# ============================================================
# 页面 2：检索查询
# ============================================================

def render_search_page():
    """渲染检索查询页面"""
    render_header(
        "检索查询",
        "直接使用混合检索引擎（向量 + BM25 + RRF + 重排）查询药典内容",
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
                    for i, r in enumerate(result["results"], 1):
                        render_source_card(r, i)
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
    )

    try:
        stats = api.stats()
        health = api.health()

        # 核心指标
        st.markdown("### 核心指标")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_stat_card(str(stats["total_chunks"]), "知识切片(Chunks)", "📄")
        with col2:
            render_stat_card(str(stats["total_drugs"]), "收录药品", "🌿")
        with col3:
            render_stat_card(str(health.get("chunks_count", 0)), "向量索引", "🔢")
        with col4:
            render_stat_card(health.get("model", "N/A"), "LLM 模型", "🤖")

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
    )

    st.markdown(
        """
        ### 系统简介

        本系统基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 架构，
        以《中国药典》2020年版一部为知识来源，为用户提供准确、可溯源的中医药智能问答服务。

        ### 技术架构（五阶段流水线）

        | 阶段 | 模块 | 说明 |
        |------|------|------|
        | **阶段一** | 数据工程 | .docx 解析 → OCR 清洗 → 结构化提取 → 智能切片 |
        | **阶段二** | 索引构建 | BGE 向量化 + Chroma 存储 + BM25 关键词索引 |
        | **阶段三** | 检索引擎 | 向量+BM25 混合检索 → RRF 融合 → BGE 重排 |
        | **阶段四** | 生成引擎 | 指代消解 → Prompt 工程 → LongCat 2.0 生成 → 后处理 |
        | **阶段五** | API 服务 | FastAPI RESTful + SSE 流式 + 会话管理 |

        ### 核心特性

        - **混合检索**：向量语义检索 + BM25 关键词检索，通过 RRF 算法融合
        - **精准重排**：BGE-Reranker 跨编码器对候选结果二次排序
        - **多轮对话**：基于 LLM 的指代消解（"它" → "人参"）
        - **引用标注**：回答自动标注药典出处，确保可溯源性
        - **一致性校验**：数值/单位一致性检查，防止 LLM 幻觉
        - **用药安全**：自动添加用药安全提醒
        - **流式输出**：SSE 实时流式生成，提升用户体验

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
        | LLM | 美团 LongCat 2.0 |
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

    st.markdown("---")

    st.markdown(
        """
        <div class="medical-disclaimer">
            <strong>⚠ 免责声明</strong>：本系统提供的信息仅供参考和学习使用，
            不构成医疗建议。具体用药请遵医嘱，切勿自行用药。
            系统回答基于《中国药典》原文，但可能存在检索或生成偏差。
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
