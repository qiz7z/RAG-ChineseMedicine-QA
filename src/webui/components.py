# -*- coding: utf-8 -*-
"""
可复用 UI 组件
================
封装 Streamlit 中常用的自定义组件，保持一致的视觉风格。

组件清单：
  - render_header          页面标题栏
  - render_source_card     检索来源卡片
  - render_citations       引用标签
  - render_disclaimer      用药安全提醒
  - render_stat_card       统计数据卡片
  - render_latency_badge   耗时标签
  - render_typing_indicator 打字动画
  - render_empty_state     空状态提示
  - render_drug_badge      药品标签
"""
import streamlit as st
from typing import List, Dict, Optional


def render_header(title: str, subtitle: str = ""):
    """渲染页面标题栏"""
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f"<p style='color: #64748B; margin-top: -0.5rem; margin-bottom: 1rem;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )


def render_source_card(source: Dict, index: int):
    """
    渲染检索来源卡片。

    Args:
        source: 来源数据（含 drug_name, section, content, score 等）
        index: 序号
    """
    drug_name = source.get("drug_name", "未知")
    section = source.get("section", "未知")
    category = source.get("category", "")
    content = source.get("content", "")
    score = source.get("score", 0)
    rerank_score = source.get("rerank_score")
    hit_sources = source.get("sources", [])

    # 分数条宽度（百分比）
    score_pct = max(5, min(100, int(score * 100)))

    # 命中来源标签
    source_tags = " ".join(
        f"<span class='citation-badge'>{s}</span>" for s in hit_sources
    ) if hit_sources else ""

    rerank_tag = (
        f"<span class='citation-badge' style='background-color: #D8F3DC; color: #1B4332; border-color: #52B788;'>"
        f"重排: {rerank_score:.4f}</span>"
    ) if rerank_score is not None else ""

    st.markdown(
        f"""
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span class="drug-name">{index}. {drug_name}</span>
                    <span class="section-name"> &gt; {section}</span>
                    {f'<span class="section-name"> ({category})</span>' if category else ''}
                </div>
                <span class="citation-badge">相似度: {score:.4f}</span>
            </div>
            <div style="margin-top: 6px; font-size: 0.875rem; color: #1E293B; line-height: 1.5;">
                {content[:300]}{"..." if len(content) > 300 else ""}
            </div>
            <div class="score-bar">
                <div class="score-fill" style="width: {score_pct}%;"></div>
            </div>
            <div style="margin-top: 6px;">
                {source_tags}
                {rerank_tag}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citations(citations: List[str]):
    """渲染引用标签列表"""
    if not citations:
        return

    badges = "".join(
        f"<span class='citation-badge'>{c}</span>" for c in citations
    )
    st.markdown(
        f"""
        <div style="margin-top: 0.5rem;">
            <span style="font-size: 0.85rem; color: #64748B; font-weight: 500;">引用来源：</span>
            {badges}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """渲染用药安全提醒"""
    st.markdown(
        """
        <div class="medical-disclaimer">
            <strong>⚠ 用药安全提醒</strong>：以上信息仅供参考，来源于《中国药典》记载。
            具体用药请遵医嘱，切勿自行用药。
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
        icon: 前缀图标（文本形式）
    """
    icon_html = f"<span style='font-size: 1.25rem;'>{icon}</span>" if icon else ""
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
    """渲染耗时标签"""
    if latency_ms < 1000:
        color = "#2D6A4F"
        bg = "#D8F3DC"
    elif latency_ms < 3000:
        color = "#D4A373"
        bg = "#FAEDCD"
    else:
        color = "#E63946"
        bg = "#FDE8E8"

    st.markdown(
        f"""
        <span style="
            display: inline-block;
            background-color: {bg};
            color: {color};
            font-size: 0.8rem;
            font-weight: 500;
            padding: 2px 10px;
            border-radius: 12px;
            margin-top: 4px;
        ">耗时 {latency_ms}ms</span>
        """,
        unsafe_allow_html=True,
    )


def render_typing_indicator():
    """渲染打字动画指示器"""
    st.markdown(
        """
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str, hint: str = ""):
    """渲染空状态提示"""
    hint_html = f"<p style='color: #94A3B8; font-size: 0.85rem;'>{hint}</p>" if hint else ""
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 3rem 1rem;
            color: #64748B;
        ">
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">{message}</p>
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
            background-color: #D8F3DC;
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

    items = "".join(f"<li>{issue}</li>" for issue in issues)
    st.markdown(
        f"""
        <div style="
            background-color: #FFF3E0;
            border-left: 4px solid #F77F00;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-top: 0.5rem;
            font-size: 0.85rem;
        ">
            <strong style="color: #F77F00;">一致性校验提醒</strong>
            <ul style="margin: 0.25rem 0 0 1rem; color: #1E293B;">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
