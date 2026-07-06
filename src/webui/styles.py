# -*- coding: utf-8 -*-
"""
CSS 主题样式
=============
中医药智能问答系统的视觉主题。

设计理念：
  - 风格：Accessible & Ethical（WCAG AAA 无障碍标准）
  - 主色：深草药绿（#2D6A4F）— 代表传统中药
  - 辅色：暖琥珀金（#D4A373）— 传统质感
  - 背景：暖米白（#FAF8F1）— 柔和不刺眼
  - 字体：Noto Sans SC（中文无障碍字体）
  - 对比度：≥ 4.5:1（正文）、≥ 3:1（大标题）

遵循 UI/UX Pro Max 设计规范：
  - 不使用 emoji 作为结构性图标
  - 清晰的焦点环（focus ring 3-4px）
  - 触摸目标 ≥ 44px
  - 支持 prefers-reduced-motion
"""

# ============================================================
# 颜色常量（语义化色彩令牌）
# ============================================================

COLORS = {
    # 主色调 — 深草药绿
    "primary": "#2D6A4F",
    "primary_light": "#52B788",
    "primary_dark": "#1B4332",
    "primary_bg": "#D8F3DC",

    # 辅助色 — 暖琥珀金
    "accent": "#D4A373",
    "accent_light": "#FAEDCD",
    "accent_dark": "#B08968",

    # 功能色
    "error": "#E63946",
    "error_bg": "#FDE8E8",
    "warning": "#F77F00",
    "warning_bg": "#FFF3E0",
    "success": "#2D6A4F",
    "success_bg": "#D8F3DC",
    "info": "#3B82F6",
    "info_bg": "#DBEAFE",

    # 中性色
    "bg": "#FAF8F1",           # 暖米白背景
    "surface": "#FFFFFF",       # 卡片表面
    "surface_alt": "#F5F3ED",   # 次级表面
    "border": "#E0DDD5",        # 边框
    "text": "#1E293B",          # 正文文字
    "text_secondary": "#64748B", # 次要文字
    "text_muted": "#94A3B8",    # 弱化文字

    # 用户消息气泡
    "user_bubble": "#2D6A4F",
    "user_bubble_text": "#FFFFFF",
    # AI 消息气泡
    "ai_bubble": "#FFFFFF",
    "ai_bubble_text": "#1E293B",
    # 引用标签
    "citation_bg": "#FAEDCD",
    "citation_text": "#B08968",
}


# ============================================================
# 全局 CSS 样式
# ============================================================
# 使用 $placeholder$ 占位符 + string.Template 来避免 CSS 花括号冲突

_RAW_CSS = """
<style>
/* ===== 字体导入 ===== */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&display=swap');

/* ===== 全局重置 ===== */
.stApp {
    font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: $bg$;
    color: $text$;
    line-height: 1.6;
}

/* ===== 侧边栏 ===== */
section[data-testid="stSidebar"] {
    background-color: $surface$;
    border-right: 1px solid $border$;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: $primary_dark$;
}

/* ===== 标题 ===== */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Noto Sans SC', sans-serif;
    font-weight: 600;
    color: $text$;
}

h1 {
    font-size: 1.75rem;
    border-bottom: 2px solid $primary$;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ===== 按钮 ===== */
.stButton > button {
    font-family: 'Noto Sans SC', sans-serif;
    font-weight: 500;
    border-radius: 8px;
    border: 1px solid $primary$;
    background-color: $primary$;
    color: white;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
    min-height: 44px;
    cursor: pointer;
}

.stButton > button:hover {
    background-color: $primary_dark$;
    border-color: $primary_dark$;
    box-shadow: 0 2px 8px rgba(45, 106, 79, 0.25);
}

.stButton > button:focus {
    outline: 3px solid $primary_light$;
    outline-offset: 2px;
}

.stButton > button:active {
    transform: scale(0.98);
}

/* 次级按钮 */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background-color: $surface$;
    color: $primary$;
    border-color: $border$;
}

.stButton > button[kind="secondary"]:hover {
    background-color: $surface_alt$;
    border-color: $primary$;
    color: $primary_dark$;
}

/* ===== 输入框 ===== */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'Noto Sans SC', sans-serif;
    border-radius: 8px;
    border: 1px solid $border$;
    padding: 0.625rem 0.875rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    min-height: 44px;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: $primary$;
    box-shadow: 0 0 0 3px rgba(45, 106, 79, 0.15);
    outline: none;
}

/* ===== 消息气泡 ===== */
.chat-message {
    display: flex;
    margin-bottom: 1rem;
    animation: fadeInUp 0.3s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (prefers-reduced-motion: reduce) {
    .chat-message {
        animation: none;
    }
}

/* 用户消息 */
.chat-message-user {
    justify-content: flex-end;
}

.chat-message-user .bubble {
    background-color: $user_bubble$;
    color: $user_bubble_text$;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.25rem;
    max-width: 75%;
    word-wrap: break-word;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

/* AI 消息 */
.chat-message-ai .bubble {
    background-color: $ai_bubble$;
    color: $ai_bubble_text$;
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.25rem;
    max-width: 85%;
    word-wrap: break-word;
    border: 1px solid $border$;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

/* AI 头像 */
.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-right: 0.5rem;
}

.chat-avatar-ai {
    background-color: $primary_bg$;
    color: $primary_dark$;
    border: 2px solid $primary_light$;
}

.chat-avatar-user {
    background-color: $primary$;
    color: white;
    margin-left: 0.5rem;
    margin-right: 0;
}

/* ===== 引用标签 ===== */
.citation-badge {
    display: inline-block;
    background-color: $citation_bg$;
    color: $citation_text$;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 12px;
    margin: 2px 4px 2px 0;
    border: 1px solid $accent$;
}

/* ===== 来源卡片 ===== */
.source-card {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 10px;
    padding: 0.875rem 1rem;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.2s ease;
}

.source-card:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.source-card .drug-name {
    font-weight: 600;
    color: $primary_dark$;
    font-size: 0.95rem;
}

.source-card .section-name {
    color: $text_secondary$;
    font-size: 0.85rem;
}

.source-card .score-bar {
    height: 4px;
    background-color: $surface_alt$;
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
}

.source-card .score-fill {
    height: 100%;
    background-color: $primary_light$;
    border-radius: 2px;
    transition: width 0.3s ease;
}

/* ===== 信息卡片 ===== */
.info-card {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}

.info-card-accent {
    border-left: 4px solid $primary$;
}

/* ===== 安全提醒 ===== */
.medical-disclaimer {
    background-color: $warning_bg$;
    border-left: 4px solid $warning$;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: $text$;
}

.medical-disclaimer strong {
    color: $warning$;
}

/* ===== 统计卡片 ===== */
.stat-card {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: $primary$;
    line-height: 1.2;
}

.stat-card .stat-label {
    font-size: 0.8rem;
    color: $text_secondary$;
    margin-top: 0.25rem;
}

/* ===== 标签页 ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    min-height: 44px;
}

.stTabs [aria-selected="true"] {
    background-color: $primary_bg$;
    color: $primary_dark$;
    border-bottom: 3px solid $primary$;
}

/* ===== 加载动画 ===== */
.typing-indicator {
    display: inline-flex;
    gap: 4px;
    padding: 0.5rem 0;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: $primary_light$;
    animation: typingBounce 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typingBounce {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.5;
    }
    30% {
        transform: translateY(-8px);
        opacity: 1;
    }
}

@media (prefers-reduced-motion: reduce) {
    .typing-indicator span {
        animation: none;
        opacity: 0.7;
    }
}

/* ===== 分隔线 ===== */
hr {
    border: none;
    border-top: 1px solid $border$;
    margin: 1rem 0;
}

/* ===== 链接 ===== */
a {
    color: $primary$;
    text-decoration: none;
    transition: color 0.2s ease;
}

a:hover {
    color: $primary_dark$;
    text-decoration: underline;
}

/* ===== Streamlit 原生组件覆盖 ===== */
/* 消息容器 */
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border: 1px solid $border$;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 500;
    color: $primary_dark$;
    background-color: $surface_alt$;
    border-radius: 8px;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}

[data-testid="stMetricValue"] {
    color: $primary$;
}

/* 数据表格 */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid $border$;
}

.dataframe thead th {
    background-color: $primary_bg$;
    color: $primary_dark$;
    font-weight: 600;
}

.dataframe tbody tr:nth-child(even) {
    background-color: $surface_alt$;
}

/* 滚动条 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: $surface_alt$;
}

::-webkit-scrollbar-thumb {
    background: $border$;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: $text_muted$;
}
</style>
"""


def _build_css() -> str:
    """将 $placeholder$ 占位符替换为实际颜色值"""
    css = _RAW_CSS
    for key, value in COLORS.items():
        css = css.replace(f"${key}$", value)
    return css


GLOBAL_CSS = _build_css()
