# -*- coding: utf-8 -*-
"""
CSS 主题样式
=============
中医药智能问答系统的视觉主题。

设计理念（UI/UX Pro Max — Chat & Messaging 模式）：
  - 风格：Minimalism & Swiss + Micro-interactions
  - 气泡：发送者（用户，深绿实底）/ 接收者（AI，白底卡片）强对比
  - 主色：深草药绿（#2D6A4F）— 传统中药
  - 辅色：暖琥珀金（#B08968）— 传统质感
  - 背景：暖米白（#FAF8F1）
  - 字体：优先华文中宋（STZhongsong），回退宋体系（SimSun / Songti / Noto Serif）
  - 对比度：正文 ≥ 4.5:1（WCAG AA）
  - 交互：焦点环 3px、触摸目标 ≥ 44px、hover 200ms、prefers-reduced-motion
  - 图标：全部 SVG，不用 emoji
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

    # 辅助色 — 暖琥珀金（降饱和度保对比度）
    "accent": "#B08968",
    "accent_light": "#FAEDCD",
    "accent_dark": "#8A5A2B",

    # 功能色（文字色均经对比度校正 ≥ 4.5:1）
    "error": "#C81E1E",
    "error_bg": "#FDE8E8",
    "warning": "#B45309",
    "warning_bg": "#FFF3E0",
    "success": "#2D6A4F",
    "success_bg": "#D8F3DC",
    "info": "#1D4ED8",
    "info_bg": "#DBEAFE",

    # 中性色
    "bg": "#FAF8F1",           # 暖米白背景
    "surface": "#FFFFFF",       # 卡片表面
    "surface_alt": "#F5F3ED",   # 次级表面
    "border": "#E0DDD5",        # 边框
    "text": "#202B26",          # 正文文字（暖黑，与米白底更协调）
    "text_secondary": "#526074", # 次要文字（≥ 4.5:1 on bg）
    "text_muted": "#64748B",    # 弱化文字（仅白底卡片辅助信息）

    # 用户消息气泡（发送者）
    "user_bubble": "#2D6A4F",
    "user_bubble_text": "#FFFFFF",
    # AI 消息气泡（接收者）
    "ai_bubble": "#FFFFFF",
    "ai_bubble_text": "#1E293B",
    # 引用标签（深字浅底，对比度 ≈ 8:1）
    "citation_bg": "#E6F4EA",
    "citation_text": "#1B4332",
    "citation_border": "#52B788",
}


# ============================================================
# 全局 CSS 样式
# ============================================================
# 使用 $placeholder$ 占位符替换，避免 CSS 花括号冲突

_RAW_CSS = """
<style>
/* ===== Web 字体（思源宋体子集，保证客户端无中宋时也有宋体效果）===== */
@font-face {
    font-family: 'CN Serif';
    src: url('/app/static/fonts/cn-serif-regular.woff2') format('woff2');
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}
@font-face {
    font-family: 'CN Serif';
    src: url('/app/static/fonts/cn-serif-bold.woff2') format('woff2');
    font-weight: 700;
    font-style: normal;
    font-display: swap;
}

/* ===== 字体（华文中宋优先，回退 Web 宋体）===== */
.stApp {
    background-color: $bg$;
    color: $text$;
    line-height: 1.65;
    letter-spacing: 0.006em;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    font-feature-settings: 'kern' 1, 'liga' 1;
    font-optical-sizing: auto;
}

.stApp,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] td,
.stApp [data-testid="stMarkdownContainer"] th,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4,
.stApp [data-testid="stMarkdownContainer"] h5,
.stApp [data-testid="stMarkdownContainer"] h6 {
    font-family: 'STZhongsong', '华文中宋', 'CN Serif',
                 'Songti SC', 'SimSun',
                 'Noto Serif CJK SC', 'Source Han Serif SC',
                 -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 'PingFang SC', 'Microsoft YaHei', serif !important;
}

::selection {
    background-color: $primary_bg$;
    color: $primary_dark$;
}

/* ===== 标题层级 ===== */
h1, h2, h3, h4, h5, h6 {
    font-weight: 600;
    color: $text$;
    letter-spacing: 0.02em;
    line-height: 1.4;
}

h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: $primary_dark$;
    margin-bottom: 0.3rem;
}

h2 { font-size: 1.22rem; }
h3 { font-size: 1.06rem; }

/* Streamlit 默认 h1 为 2.75rem，会盖过上面的全局规则 */
[data-testid="stMarkdownContainer"] h1 {
    font-size: 1.5rem !important;
    font-weight: 700;
    color: $primary_dark$;
    letter-spacing: 0.01em;
    margin: 0 0 0.25rem;
}

[data-testid="stMarkdownContainer"] h2 {
    font-size: 1.25rem !important;
    font-weight: 600;
    color: $text$;
}

[data-testid="stMarkdownContainer"] h3 {
    font-size: 1.05rem !important;
    font-weight: 600;
    color: $text$;
}

.page-eyebrow {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: $primary$;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.page-subtitle {
    color: $text_secondary$;
    font-size: 0.9rem;
    margin-top: 0;
    margin-bottom: 1.25rem;
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

section[data-testid="stSidebar"] h4 {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: $text_muted$;
    margin: 0.25rem 0 0.5rem;
}

section[data-testid="stSidebar"] hr {
    margin: 0.85rem 0;
}

/* 侧边栏品牌区 */
.sidebar-brand {
    text-align: center;
    padding: 0.5rem 0 0.25rem;
}

.sidebar-logo {
    width: 54px;
    height: 54px;
}

.sidebar-title {
    margin-top: 0.55rem;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: $primary_dark$;
    line-height: 1.4;
}

.sidebar-sub {
    margin-top: 0.1rem;
    font-size: 0.76rem;
    color: $text_muted$;
    white-space: nowrap;
    line-height: 1.5;
}

/* 侧边栏导航（按钮式，容器 key=sidenav，按钮 key=nav_*）
   不使用 radio，也不再依赖任何 :has() / baseweb 内部结构 */
.st-key-sidenav {
    gap: 0.12rem !important;
}

[class*="st-key-nav_"] .stButton > button {
    justify-content: flex-start !important;
    width: 100%;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    color: $text_secondary$ !important;
    font-weight: 500;
    border-radius: 10px !important;
    padding: 0.5rem 0.8rem !important;
    min-height: 42px;
    box-shadow: none !important;
    transition: background-color 0.18s ease, color 0.18s ease,
                box-shadow 0.18s ease;
}

[class*="st-key-nav_"] .stButton > button > div {
    justify-content: flex-start !important;
    width: 100%;
}

[class*="st-key-nav_"] .stButton > button p {
    text-align: left;
    width: 100%;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}

[class*="st-key-nav_"] .stButton > button:hover {
    background-color: $surface_alt$ !important;
    color: $primary_dark$ !important;
    border-color: transparent !important;
}

[class*="st-key-nav_"][class*="_active"] .stButton > button {
    background-color: $primary_bg$ !important;
    color: $primary_dark$ !important;
    box-shadow: inset 3px 0 0 $primary$ !important;
}

[class*="st-key-nav_"][class*="_active"] .stButton > button p {
    font-weight: 600;
}

/* ===== 状态卡片（侧边栏）===== */
.status-card {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 12px;
    padding: 0.875rem 1rem;
}

.status-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    color: $primary_dark$;
    margin-bottom: 0.5rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: $primary$;
    flex-shrink: 0;
    animation: statusPulse 2s ease-in-out infinite;
}

@keyframes statusPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(45, 106, 79, 0.35); }
    50% { box-shadow: 0 0 0 5px rgba(45, 106, 79, 0); }
}

.status-rows {
    display: grid;
    gap: 0.3rem;
    font-size: 0.82rem;
}

.status-rows .row {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
}

.status-rows .row span { color: $text_secondary$; }
.status-rows .row b { color: $text$; font-weight: 600; }

/* ===== 按钮 ===== */
.stButton > button {
    font-family: inherit;
    font-weight: 500;
    border-radius: 8px;
    border: 1px solid $primary$;
    background-color: $primary$;
    color: #FFFFFF;
    padding: 0.5rem 1.5rem;
    transition: background-color 0.2s ease, border-color 0.2s ease,
                box-shadow 0.2s ease, transform 0.15s ease;
    min-height: 44px;
    cursor: pointer;
}

.stButton > button:hover {
    background-color: $primary_dark$;
    border-color: $primary_dark$;
    box-shadow: 0 2px 8px rgba(45, 106, 79, 0.25);
}

.stButton > button:active {
    transform: scale(0.98);
}

:focus-visible {
    outline: 3px solid $primary_light$;
    outline-offset: 2px;
}

.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background-color: $surface$;
    color: $primary_dark$;
    border-color: $border$;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    background-color: $primary_bg$;
    border-color: $primary$;
}

/* ===== 首页欢迎态：整体一屏内垂直居中 ===== */
.st-key-home {
    min-height: calc(100vh - 200px);
    max-width: 920px;
    margin: 0 auto;
    justify-content: center !important;
    gap: 0 !important;
}

/* 对话进行中立即隐藏首页容器：
   流式生成期间脚本未结束，旧帧的 hero / 示例问题会以 stale 状态残留，
   这里按 DOM 内容判断，不依赖后续脚本注入的样式 */
.stApp:has(.msg-row-user) .st-key-home,
.stApp:has(.st-key-amsg_live) .st-key-home,
.stApp:has(.st-key-amsg_h_0) .st-key-home {
    display: none !important;
}

/* 首页内联输入框：夹在 hero 与示例问题之间 */
.st-key-home [data-testid="stChatInput"] {
    width: 100%;
    margin: 1.15rem auto 0;
    padding: 0;
}

/* ===== 欢迎 Hero（DeepSeek 式居中画布）===== */
.hero {
    text-align: center !important;
    padding: 0 1rem 0.25rem;
}

.hero-title {
    font-size: 1.9rem;
    font-weight: 600;
    color: $text$;
    margin: 0 0 0.6rem;
    letter-spacing: 0.02em;
    line-height: 1.38;
    text-wrap: balance;
}

.hero .hero-subtitle {
    display: block;
    width: 100%;
    max-width: 680px;
    color: $text_secondary$;
    font-size: 0.9rem;
    margin: 0 auto !important;
    padding: 0;
    line-height: 1.75;
    letter-spacing: 0.01em;
    text-align: center !important;
    text-wrap: balance;
}

/* 能力项做成小卡片，与下方示例问题卡片同一视觉语言 */
.capability-chips {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 1.35rem;
}

.capability-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    min-width: 124px;
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 12px;
    padding: 9px 14px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: $text$;
    line-height: 1.3;
    box-shadow: 0 1px 2px rgba(30, 41, 59, 0.04);
    transition: border-color 0.18s ease, box-shadow 0.18s ease,
                transform 0.18s ease;
}

.capability-chip:hover {
    border-color: $primary_light$;
    box-shadow: 0 6px 16px -10px rgba(45, 106, 79, 0.35);
    transform: translateY(-1px);
}

.capability-chip svg {
    color: $primary$;
    flex-shrink: 0;
}

/* ===== 示例问题卡片（作用域：容器 key=example_grid → .st-key-example_grid）===== */
.section-label {
    max-width: 920px;
    margin: 1.35rem auto 0.7rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: $text_muted$;
    text-align: center;
}

.st-key-example_grid {
    max-width: 920px;
    margin: 1.25rem auto 0.5rem;
    gap: 0.5rem !important;
}

.st-key-example_grid .stButton > button {
    position: relative;
    justify-content: flex-start !important;
    align-items: center;
    text-align: left;
    width: 100%;
    height: 64px;
    background-color: $surface$;
    border: 1px solid $border$;
    color: $text$;
    border-radius: 12px;
    padding: 0.7rem 2.25rem 0.7rem 1rem;
    font-weight: 500;
    box-shadow: none;
    transition: border-color 0.2s ease, background-color 0.2s ease,
                transform 0.2s ease, box-shadow 0.2s ease;
}

.st-key-example_grid .stButton > button > div {
    justify-content: flex-start !important;
    width: 100%;
}

.st-key-example_grid .stButton > button p {
    text-align: left;
    width: 100%;
    font-size: 0.9rem;
    line-height: 1.5;
    letter-spacing: 0.01em;
    white-space: normal;
}

/* 右侧箭头（纯 CSS 几何图形，不依赖图标资源） */
.st-key-example_grid .stButton > button::after {
    content: "";
    position: absolute;
    top: 50%;
    right: 1.05rem;
    width: 7px;
    height: 7px;
    margin-top: -4px;
    border-top: 1.5px solid #A6B0BF;
    border-right: 1.5px solid #A6B0BF;
    transform: rotate(45deg);
    transition: border-color 0.2s ease, right 0.2s ease;
}

.st-key-example_grid .stButton > button:hover {
    border-color: $primary_light$;
    background-color: #F5FAF7;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(45, 106, 79, 0.1);
}

.st-key-example_grid .stButton > button:hover::after {
    border-color: $primary$;
    right: 0.9rem;
}

/* ===== 聊天消息：左右气泡 =====
   用户消息 → render_user_message() 输出纯 HTML（flex 右对齐）
   助手消息 → st.container(key="amsg_*") 左对齐卡片（::before 绘制头像）
   全程不依赖 st.chat_message 内部 DOM，也不使用 :has() */

.st-key-chat_thread {
    gap: 0.1rem !important;
}

/* 用户消息：右对齐深绿气泡 */
.msg-row-user {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 10px;
    width: 100%;
    max-width: 920px;
    margin: 0.15rem auto 1rem;
    animation: fadeInUp 0.25s ease-out;
}

.msg-row-user .msg-bubble {
    background: linear-gradient(135deg, #347A5C 0%, #2D6A4F 100%);
    color: #FFFFFF;
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1.05rem;
    font-size: 0.95rem;
    line-height: 1.72;
    letter-spacing: 0.01em;
    max-width: min(76%, 620px);
    word-break: break-word;
    box-shadow: 0 2px 10px -4px rgba(27, 67, 50, 0.45);
}

.msg-row-user .msg-avatar {
    width: 32px;
    height: 32px;
    flex: 0 0 32px;
    border-radius: 50%;
    border: 1px solid #DCE7DF;
    background-color: #FFFFFF;
    object-fit: cover;
}

/* 助手消息：左侧头像 + 白卡气泡 */
[class*="st-key-amsg_"] {
    position: relative;
    max-width: 920px;
    margin: 0 auto 1rem;
    padding: 0.85rem 1.15rem 1.35rem 3.15rem;
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 4px 16px 16px 16px;
    gap: 0.45rem !important;
    box-shadow: 0 1px 2px rgba(30, 41, 59, 0.04),
                0 10px 26px -20px rgba(30, 41, 59, 0.4);
    animation: fadeInUp 0.25s ease-out;
}

[class*="st-key-amsg_"]::before {
    content: "";
    position: absolute;
    top: 0.8rem;
    left: 0.85rem;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: $primary_bg$;
    background-image: url("data:image/svg+xml;base64,$avatar_ai$");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 24px 24px;
    border: 1px solid #D7E9DD;
}

/* ===== 气泡内 Markdown 排版 =====
   LLM 输出常带 #/## 标题，Streamlit 默认 h1=2.75rem 会把气泡撑成海报，
   这里统一压到正文尺度：标题只做层级区分，不做视觉主角。 */
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h1,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h2,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h3,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h4,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h5,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h6 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: $primary_dark$ !important;
    line-height: 1.55 !important;
    letter-spacing: 0 !important;
    margin: 0.9rem 0 0.4rem !important;
    padding: 0 !important;
    border: none !important;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h1 { font-size: 1.12rem !important; }
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h2 { font-size: 1.02rem !important; }

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h1:first-child,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h2:first-child,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] h3:first-child {
    margin-top: 0 !important;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] p {
    margin: 0 0 0.55rem;
    font-size: 0.94rem;
    line-height: 1.78;
    letter-spacing: 0.008em;
    color: $text$;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] p:last-child {
    margin-bottom: 0;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] ul,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] ol {
    margin: 0.25rem 0 0.6rem;
    padding-left: 1.15rem;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] li {
    margin: 0.18rem 0;
    padding-left: 0.1rem;
    font-size: 0.94rem;
    line-height: 1.72;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] li::marker {
    color: $primary_light$;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] strong {
    font-weight: 600;
    color: $primary_dark$;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] hr {
    margin: 0.9rem 0;
    border-top: 1px solid $border$;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] code {
    background-color: $surface_alt$;
    color: $primary_dark$;
    padding: 0.08rem 0.32rem;
    border-radius: 4px;
    font-size: 0.85em;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] pre {
    background-color: $surface_alt$;
    border: 1px solid $border$;
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    margin: 0.5rem 0;
    overflow-x: auto;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] pre code {
    background: transparent;
    padding: 0;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
    margin: 0.5rem 0 0.7rem;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] th {
    background-color: $primary_bg$;
    color: $primary_dark$;
    font-weight: 600;
    text-align: left;
}

[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] th,
[class*="st-key-amsg_"] [data-testid="stMarkdownContainer"] td {
    border: 1px solid $border$;
    padding: 0.35rem 0.55rem;
    line-height: 1.6;
}

/* 元信息条（margin 置 0，避免上外边距塌陷导致超出卡片背景） */
.msg-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.7rem;
    margin: 0;
    padding-top: 0.5rem;
    border-top: 1px dashed $border$;
}

.meta-text {
    font-size: 0.75rem;
    color: $text_muted$;
    line-height: 1.6;
}

.meta-safety {
    margin-left: auto;
    font-size: 0.75rem;
    color: $text_muted$;
    line-height: 1.6;
}

@media (max-width: 768px) {
    .meta-safety {
        margin-left: 0;
        flex-basis: 100%;
    }
}

[class*="st-key-amsg_"] blockquote {
    border-left: 3px solid $primary_light$;
    margin: 0.55rem 0;
    padding: 0.4rem 0.8rem;
    color: $text_secondary$;
    background-color: $surface_alt$;
    border-radius: 0 8px 8px 0;
}

[class*="st-key-amsg_"] blockquote p {
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.62;
    color: $text_secondary$;
}

[class*="st-key-amsg_"] blockquote p + p {
    margin-top: 0.35rem;
}

/* 流式输出光标 */
.streaming-cursor {
    color: $primary$;
    animation: cursorBlink 1s step-end infinite;
}

@keyframes cursorBlink {
    50% { opacity: 0; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
    [class*="st-key-amsg_"],
    .msg-row-user,
    .stat-card,
    .source-card,
    .status-dot {
        animation: none;
        transition: none;
    }
}

/* ===== 输入框 / 聊天输入 ===== */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    font-family: inherit;
    border-radius: 8px;
    border: 1px solid $border$;
    padding: 0.625rem 0.875rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    min-height: 44px;
    background-color: $surface$;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: $primary$;
    box-shadow: 0 0 0 3px rgba(45, 106, 79, 0.15);
}

.stSelectbox > div > div {
    border-radius: 8px;
    border: 1px solid $border$;
    min-height: 44px;
}

/* ===== 聊天输入框：一体化胶囊 =====
   外框（白底 + 细描边 + 柔和投影）承载 textarea 和发送按钮，
   聚焦时整体描边变绿并带光晕，避免“输入框 + 悬浮按钮”两张皮。 */
[data-testid="stChatInput"] {
    max-width: 920px;
    margin-left: auto;
    margin-right: auto;
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatInput"] > div {
    position: relative;
    background-color: $surface$ !important;
    border: 1px solid #DCD8CD !important;
    border-radius: 30px !important;
    padding: 9px 9px 9px 22px !important;
    box-shadow: 0 6px 22px -18px rgba(30, 41, 59, 0.45);
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: $primary$ !important;
    box-shadow: 0 0 0 3px rgba(45, 106, 79, 0.12),
                0 10px 26px -20px rgba(45, 106, 79, 0.5);
}

/* 输入区：透明、无边框，只负责文字 */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 6px 4px 6px 0 !important;
    font-family: inherit !important;
    font-size: 0.98rem !important;
    line-height: 1.55 !important;
    min-height: 28px !important;
    color: $text$ !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputTextArea"]::placeholder {
    color: #98A2B3 !important;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputTextArea"]:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* 发送按钮：胶囊内右侧的圆形主色按钮 */
[data-testid="stChatInputSubmitButton"] {
    display: flex !important;
    align-items: center;
    justify-content: center;
    width: 44px !important;
    height: 44px !important;
    min-height: 44px !important;
    padding: 0 !important;
    border: none !important;
    border-radius: 50% !important;
    background-color: $primary$ !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px -4px rgba(45, 106, 79, 0.55);
    transition: background-color 0.18s ease, transform 0.18s ease,
                box-shadow 0.18s ease;
}

[data-testid="stChatInputSubmitButton"]:hover {
    background-color: $primary_dark$ !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 14px -6px rgba(45, 106, 79, 0.6);
}

[data-testid="stChatInputSubmitButton"]:disabled {
    background-color: #E8E5DC !important;
    color: #A9B2C0 !important;
    box-shadow: none;
    transform: none;
}

[data-testid="stChatInputSubmitButton"] svg {
    fill: currentColor !important;
    color: inherit !important;
}

/* 底部输入区：容器本身透明，避免出现第二条背景带 */
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stBottomBlockContainer"] {
    max-width: 920px;
    padding-bottom: 1.1rem;
}

/* ===== 引用标签 ===== */
.citation-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-top: 0.75rem;
    margin-bottom: 0.1rem;
}

.citation-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: $text_secondary$;
    margin-right: 2px;
}

.citation-row .citation-badge {
    margin: 0;
}

.citation-badge {
    display: inline-flex;
    align-items: center;
    max-width: 100%;
    background-color: $citation_bg$;
    color: $citation_text$;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 12px;
    margin: 2px 4px 2px 0;
    border: 1px solid $citation_border$;
    line-height: 1.55;
    white-space: normal;
    word-break: break-word;
}

/* ===== 来源卡片 ===== */
.source-card {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 12px;
    padding: 0.875rem 1rem;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

.source-card:hover {
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.07);
    border-color: $primary_light$;
}

.source-card .source-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
}

.source-card .source-title {
    min-width: 0;
}

.source-card .source-content {
    margin-top: 6px;
    font-size: 0.86rem;
    color: $text$;
    line-height: 1.65;
    word-break: break-word;
}

.source-card .source-tags {
    margin-top: 6px;
}

.source-card .source-tags:empty {
    display: none;
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
    background-color: $primary$;
    border-radius: 2px;
    transition: width 0.3s ease;
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
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
}

.medical-disclaimer svg {
    flex-shrink: 0;
    margin-top: 2px;
    color: $warning$;
}

.medical-disclaimer strong {
    color: $warning$;
}

/* ===== 统计卡片 ===== */
.stat-card {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 12px;
    padding: 1.125rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}

.stat-card .stat-icon {
    color: $primary$;
    margin-bottom: 0.3rem;
}

.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: $primary_dark$;
    line-height: 1.2;
    font-variant-numeric: tabular-nums;
}

.stat-card .stat-label {
    font-size: 0.8rem;
    color: $text_secondary$;
    margin-top: 0.25rem;
}

/* ===== 特性卡片（关于页）===== */
.feature-card {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 12px;
    padding: 1rem 1.125rem;
    height: 100%;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.feature-card:hover {
    border-color: $primary_light$;
    box-shadow: 0 4px 14px rgba(45, 106, 79, 0.08);
}

.feature-card .feature-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 600;
    color: $primary_dark$;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}

.feature-card .feature-head svg { color: $primary$; flex-shrink: 0; }

.feature-card .feature-desc {
    font-size: 0.85rem;
    color: $text_secondary$;
    line-height: 1.6;
    margin: 0;
}

/* ===== 耗时标签 ===== */
.latency-badge {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 12px;
    margin-top: 4px;
    line-height: 1.6;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.01em;
}

.latency-fast { background-color: $success_bg$; color: $success$; }
.latency-mid  { background-color: $accent_light$; color: $accent_dark$; }
.latency-slow { background-color: $error_bg$; color: $error$; }

/* ===== 打字动画 ===== */
.typing-indicator {
    display: inline-flex;
    gap: 4px;
    padding: 0.5rem 0;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: $primary$;
    animation: typingBounce 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-8px); opacity: 1; }
}

@media (prefers-reduced-motion: reduce) {
    .typing-indicator span { animation: none; opacity: 0.7; }
}

/* ===== 分隔线 ===== */
hr {
    border: none;
    border-top: 1px solid $border$;
    margin: 1rem 0;
}

/* ===== 链接 ===== */
a {
    color: $primary_dark$;
    text-decoration: none;
    transition: color 0.2s ease;
}

a:hover {
    color: $primary$;
    text-decoration: underline;
}

/* ===== Expander（来源折叠：轻量卡片）===== */
[data-testid="stExpander"] {
    border: 1px solid $border$ !important;
    background-color: $surface_alt$ !important;
    border-radius: 10px !important;
    margin-top: 0.9rem;
    overflow: hidden;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] button {
    padding: 0.5rem 0.8rem !important;
    font-weight: 500;
    font-size: 0.85rem;
    color: $primary_dark$;
    background: transparent !important;
    border-radius: 0 !important;
    transition: background-color 0.18s ease;
}

[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] button:hover {
    background-color: $primary_bg$ !important;
}

[data-testid="stExpanderDetails"] {
    padding: 0 0.8rem 0.7rem !important;
    background: transparent !important;
}

[data-testid="stExpander"] .streamlit-expanderHeader {
    font-weight: 500;
    font-size: 0.85rem;
    color: $primary_dark$;
}

/* ===== Metrics ===== */
[data-testid="stMetric"] {
    background-color: $surface$;
    border: 1px solid $border$;
    border-radius: 12px;
    padding: 0.875rem 1rem;
}

[data-testid="stMetricValue"] {
    color: $primary_dark$;
}

[data-testid="stMetricLabel"] {
    color: $text_secondary$;
}

/* ===== 数据表格 ===== */
.dataframe {
    border-radius: 10px;
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

/* ===== 代码块 ===== */
.stCode { border-radius: 10px; }

code {
    font-family: 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace;
    font-size: 0.875em;
}

/* ===== 滚动条 ===== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: $surface_alt$; }
::-webkit-scrollbar-thumb { background: $border$; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: $text_muted$; }

/* ===== 页脚 / 顶部工具栏（只隐藏右侧动作区，保留侧边栏展开按钮）===== */
footer,
[data-testid="stDecoration"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
[data-testid="stToolbarActions"],
#MainMenu {
    display: none !important;
}

/* stToolbar 必须保留：侧边栏折叠时，stExpandSidebarButton 就在其中 */
[data-testid="stToolbar"],
[data-testid="stAppToolbar"] {
    display: flex !important;
    visibility: visible !important;
    background: transparent !important;
}

[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stExpandSidebarButton"] button {
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    min-height: 40px;
    padding: 0;
    border-radius: 10px;
    background-color: $surface$;
    border: 1px solid $border$;
    color: $primary_dark$;
    box-shadow: 0 2px 10px rgba(30, 41, 59, 0.12);
    transition: background-color 0.18s ease, border-color 0.18s ease;
}

[data-testid="stExpandSidebarButton"] button:hover {
    background-color: $primary_bg$;
    border-color: $primary_light$;
}

/* ===== 主内容区 ===== */
[data-testid="stMainBlockContainer"],
.block-container {
    max-width: 1140px;
    padding-top: 3.25rem !important;
    padding-bottom: 4rem !important;
}

/* ===== 滚动稳定性 =====
   流式输出时内容高度持续增长，浏览器的 scroll anchoring 会不断
   修正滚动位置，表现为“滚不动 / 一直跳”。关掉锚定后视口位置稳定。 */
html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    overflow-anchor: none;
}

[data-testid="stMain"] {
    scroll-padding-bottom: 4rem;
    scroll-behavior: auto;
    overscroll-behavior: contain;
}

/* 消息出现动画在 rerun 时会重播，滚动中造成位移跳动，聊天区关闭 */
[class*="st-key-amsg_"],
.msg-row-user {
    animation: none !important;
}

/* ===== 响应式 ===== */
@media (max-width: 768px) {
    .hero { padding: 1.5rem 0.5rem 0.5rem; }
    .hero-title { font-size: 1.45rem; }
    .stat-card .stat-value { font-size: 1.3rem; }
    .msg-row-user { max-width: 100%; }
    .msg-row-user .msg-bubble { max-width: 84%; font-size: 0.93rem; }
    [class*="st-key-amsg_"] { padding: 0.8rem 0.9rem 0.65rem 2.85rem; }
    [class*="st-key-amsg_"]::before {
        left: 0.7rem;
        width: 28px;
        height: 28px;
        background-size: 21px 21px;
    }
    .st-key-example_grid { max-width: 100%; }
    .st-key-example_grid .stButton > button {
        height: auto;
        min-height: 56px;
    }
    .section-label { margin-top: 1.5rem; }
    [data-testid="stMainBlockContainer"],
    .block-container {
        padding-top: 3rem !important;
    }
}
</style>
"""


def _asset_data_uri(filename: str) -> str:
    """读取 assets 下的 SVG 并转为 base64（供 CSS url() 使用）"""
    import base64
    from pathlib import Path

    path = Path(__file__).resolve().parent / "assets" / filename
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_css() -> str:
    """将 $placeholder$ 占位符替换为实际颜色值/静态资源"""
    css = _RAW_CSS
    for key, value in COLORS.items():
        css = css.replace(f"${key}$", value)
    # 助手头像（CSS 伪元素背景）
    css = css.replace("$avatar_ai$", _asset_data_uri("avatar-ai.svg"))
    return css


GLOBAL_CSS = _build_css()
