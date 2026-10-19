# -*- coding: utf-8 -*-
"""
可复用 UI 组件
================
封装 Streamlit 中常用的自定义组件，保持一致的视觉风格。

组件清单：
  - render_header             页面标题栏（eyebrow + 标题 + 副标题）
  - render_hero               问答页欢迎区（Logo + 标题 + 能力标签）
  - render_status_card        侧边栏服务状态卡片
  - render_source_card        检索来源卡片
  - render_citations          引用标签
  - render_disclaimer         用药安全提醒
  - render_stat_card          统计数据卡片
  - render_latency_badge      耗时标签
  - render_typing_indicator   打字动画
  - render_empty_state        空状态提示
  - render_drug_badge         药品标签
  - render_feature_card       特性卡片（关于页）

设计规范（UI/UX Pro Max）：
  - 图标一律使用 SVG（webui/icons.py），不用 emoji
  - 可交互元素均有 hover / focus 状态
"""
import base64
import html
import re
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st

from .icons import ICONS

# 静态资源目录
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _svg_data_uri(path: Path) -> str:
    """读取 SVG 文件并转为 base64 data URI（用于 HTML <img>）"""
    return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode()


LOGO_DATA_URI = _svg_data_uri(ASSETS_DIR / "logo.svg")
AVATAR_AI_DATA_URI = _svg_data_uri(ASSETS_DIR / "avatar-ai.svg")
AVATAR_USER_DATA_URI = _svg_data_uri(ASSETS_DIR / "avatar-user.svg")


def render_user_message(text: str):
    """
    渲染用户消息（右对齐气泡）。

    使用纯 HTML 输出而非 st.chat_message，彻底避免依赖 Streamlit 内部
    DOM 结构（:has() 选择器在不同版本中不可靠），左右布局由 flex 保证。
    """
    safe = html.escape(str(text)).replace("\n", "<br>")
    st.markdown(
        f'<div class="msg-row msg-row-user">'
        f'<div class="msg-bubble">{safe}</div>'
        f'<img class="msg-avatar" src="{AVATAR_USER_DATA_URI}" alt="我" />'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_assistant_meta(
    latency_ms: int = 0,
    dialogue_turn: int = 0,
    original_query: str = "",
    resolved_query: str = "",
):
    """渲染助手消息底部元信息条（耗时 / 轮次 / 指代消解 / 安全提示）"""
    parts: List[str] = []

    if latency_ms:
        if latency_ms < 1000:
            level = "latency-fast"
        elif latency_ms < 3000:
            level = "latency-mid"
        else:
            level = "latency-slow"
        parts.append(f'<span class="latency-badge {level}">耗时 {latency_ms} ms</span>')

    if dialogue_turn:
        parts.append(f'<span class="meta-text">第 {dialogue_turn} 轮</span>')

    if resolved_query and original_query and resolved_query != original_query:
        parts.append(
            '<span class="meta-text">指代消解：'
            f'{html.escape(original_query)} → {html.escape(resolved_query)}</span>'
        )

    parts.append('<span class="meta-safety">内容源自《中国药典》，仅供参考，用药请遵医嘱</span>')

    st.markdown(
        f'<div class="msg-meta">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str = "", eyebrow: str = ""):
    """渲染页面标题栏"""
    if eyebrow:
        st.markdown(f'<div class="page-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<p class="page-subtitle">{subtitle}</p>',
            unsafe_allow_html=True,
        )


def render_hero(logo_data_uri: str = ""):
    """渲染问答页欢迎区（DeepSeek 式居中问候）"""
    # 不缩进：缩进的多行 HTML 会被 Markdown 当作代码块/独立段落处理，
    # 导致 <p> 脱离 .hero 容器、继承不到 text-align: center。
    # logo 与侧边栏品牌重复，首页不再重复展示。
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-title">有什么可以帮您？</div>'
        f'<p class="hero-subtitle">基于《中国药典》2020 年版一部 · 向量 + BM25 混合检索 · 回答标注药典出处</p>'
        f'<div class="capability-chips">'
        f'<span class="capability-chip">{ICONS["git-branch"]}混合检索</span>'
        f'<span class="capability-chip">{ICONS["bookmark"]}引用溯源</span>'
        f'<span class="capability-chip">{ICONS["shield"]}用药安全提醒</span>'
        f'<span class="capability-chip">{ICONS["history"]}多轮对话</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_status_card(health: Optional[Dict]):
    """渲染侧边栏服务状态卡片"""
    if health:
        model = health.get("model", "N/A")
        rows = (
            f'<div class="row"><span>模型</span><b>{model}</b></div>'
            f'<div class="row"><span>知识切片</span><b>{health.get("chunks_count", 0):,}</b></div>'
            f'<div class="row"><span>收录药品</span><b>{health.get("drug_count", 0):,}</b></div>'
        )
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-head"><span class="status-dot"></span>服务运行正常</div>
                <div class="status-rows">{rows}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-head" style="color: #C81E1E;">
                    <span class="status-dot" style="background-color: #C81E1E; animation: none;"></span>
                    服务未连接
                </div>
                <div class="status-rows">
                    <div class="row"><span>请先启动后端 API</span></div>
                    <div class="row"><span style="font-family: monospace; font-size: 0.78rem;">python scripts/run/run_api.py</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _clean_text(text) -> str:
    """剥离解析残留的 HTML 标签并转义，避免破坏页面结构"""
    text = str(text or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[a-zA-Z][^>]{0,120}>", "", text)
    return html.escape(text)


def render_source_card(source: Dict, index: int, max_score: float = None):
    """
    渲染检索来源卡片。

    Args:
        source: 来源数据（含 drug_name, section, content, score 等）
        index: 序号
        max_score: 本题最高分，用于把分数条归一到 0~100%
    """
    drug_name = _clean_text(source.get("drug_name", "未知"))
    section = _clean_text(source.get("section", "未知"))
    category = _clean_text(source.get("category", ""))

    # 关键：content 必须压成单行。
    # 多行文本会把 st.markdown 的 HTML 块截断，剩余闭合标签会被当成
    # 缩进代码块渲染成 "</div>" 之类的乱码。
    raw_content = str(source.get("content", ""))
    raw_content = re.sub(r"<br\s*/?>", "\n", raw_content, flags=re.IGNORECASE)
    raw_content = re.sub(r"</?[a-zA-Z][^>]{0,120}>", "", raw_content)
    if len(raw_content) > 300:
        raw_content = raw_content[:300] + "..."
    content = html.escape(raw_content).replace("\n", "<br>")

    score = source.get("score", 0)
    rerank_score = source.get("rerank_score")
    hit_sources = source.get("sources", [])
    source_scores = source.get("source_scores") or {}

    # 分数条按本题最高分归一化：RRF 分只有 0.01~0.03 量级，
    # 直接乘 100 会让所有卡片都停在 5%，看不出相对高低。
    base = max_score if max_score and max_score > 0 else score
    score_pct = max(5, min(100, int(score / base * 100))) if base else 5

    def _source_label(name: str) -> str:
        value = source_scores.get(name)
        if value is None:
            return _clean_text(name)
        # 向量是 0~1 余弦相似度，BM25 是无界分，精度分开给
        precision = 3 if name == "vector" else 2
        return f"{_clean_text(name)} {value:.{precision}f}"

    # 命中来源标签（带各路原始分）
    source_tags = " ".join(
        f'<span class="citation-badge">{_source_label(s)}</span>'
        for s in hit_sources
    ) if hit_sources else ""

    score_label = "重排分" if rerank_score is not None else "RRF 融合分"

    category_html = (
        f'<span class="section-name"> ({category})</span>' if category else ""
    )
    tags_html = source_tags

    # 整段 HTML 不缩进：缩进行在 Markdown 里会被识别为代码块
    st.markdown(
        f'<div class="source-card">'
        f'<div class="source-head">'
        f'<div class="source-title">'
        f'<span class="drug-name">{index}. {drug_name}</span>'
        f'<span class="section-name"> &gt; {section}</span>'
        f'{category_html}'
        f'</div>'
        f'<span class="citation-badge">{score_label}: {score:.4f}</span>'
        f'</div>'
        f'<div class="source-content">{content}</div>'
        f'<div class="score-bar"><div class="score-fill" style="width: {score_pct}%;"></div></div>'
        f'<div class="source-tags">{tags_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_citations(citations: List[str]):
    """
    渲染引用标签列表。

    用 flex 容器承载 badge：inline-block 的基线对齐会让父容器高度小于
    badge 实际高度，导致下一元素（来源折叠面板）压到标签上。
    """
    if not citations:
        return

    badges = "".join(
        f'<span class="citation-badge">{html.escape(str(c))}</span>'
        for c in citations
    )
    st.markdown(
        f'<div class="citation-row">'
        f'<span class="citation-label">引用来源：</span>{badges}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """渲染用药安全提醒"""
    st.markdown(
        f"""
        <div class="medical-disclaimer">
            {ICONS['shield']}
            <div>
                <strong>用药安全提醒</strong>：以上信息仅供参考，来源于《中国药典》记载。
                具体用药请遵医嘱，切勿自行用药。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(value: str, label: str, icon: str = ""):
    """
    渲染统计数据卡片。

    Args:
        value: 数值（字符串形式）
        label: 标签文字
        icon: ICONS 字典中的图标键名（如 "doc" / "leaf"）
    """
    icon_html = (
        f"<div class='stat-icon'>{ICONS[icon]}</div>" if icon in ICONS else ""
    )
    st.markdown(
        f"""
        <div class="stat-card">
            {icon_html}
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_latency_badge(latency_ms: int):
    """渲染耗时标签（按耗时分级配色，对比度均 ≥ 4.5:1）"""
    if latency_ms < 1000:
        css_class = "latency-fast"
    elif latency_ms < 3000:
        css_class = "latency-mid"
    else:
        css_class = "latency-slow"

    st.markdown(
        f'<span class="latency-badge {css_class}">耗时 {latency_ms}ms</span>',
        unsafe_allow_html=True,
    )


def render_typing_indicator():
    """渲染打字动画指示器"""
    st.markdown(
        """
        <div class="typing-indicator" role="status" aria-label="正在生成回答">
            <span></span>
            <span></span>
            <span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str, hint: str = ""):
    """渲染空状态提示"""
    hint_html = f"<p style='color: #64748B; font-size: 0.85rem; margin: 0;'>{hint}</p>" if hint else ""
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 3rem 1rem;
            color: #526074;
        ">
            <div style="color: #52B788; margin-bottom: 0.5rem;">{ICONS['search']}</div>
            <p style="font-size: 1.05rem; font-weight: 500; margin: 0 0 0.5rem 0;">{message}</p>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_drug_badge(drug_name: str):
    """渲染药品标签"""
    st.markdown(
        f"""
        <span style="
            display: inline-block;
            background-color: #E6F4EA;
            color: #1B4332;
            font-size: 0.85rem;
            font-weight: 500;
            padding: 4px 12px;
            border-radius: 6px;
            margin: 2px;
            border: 1px solid #52B788;
        ">{drug_name}</span>
        """,
        unsafe_allow_html=True,
    )


def render_consistency_warning(issues: List[str]):
    """渲染一致性校验警告"""
    if not issues:
        return

    items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues)
    st.markdown(
        f"""
        <div class="medical-disclaimer" style="background-color: #FFF3E0; border-left-color: #B45309;">
            {ICONS['alert']}
            <div>
                <strong style="color: #B45309;">一致性校验提醒</strong>
                <ul style="margin: 0.25rem 0 0 1rem; color: #1E293B;">{items}</ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(icon: str, title: str, desc: str):
    """渲染特性卡片（关于页）"""
    icon_html = ICONS.get(icon, "")
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-head">{icon_html}{title}</div>
            <p class="feature-desc">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
