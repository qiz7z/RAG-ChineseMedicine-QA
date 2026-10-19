# 中华药典智能问答 — 设计系统 MASTER

> 由 UI/UX Pro Max skill 生成并落地（2026-10-18）。后续任何前端改动遵循本文件；
> 页面级例外放 `pages/<page>.md`。实现位置：`src/webui/{styles.py, components.py, app.py, assets/}`。

## 方向

- **风格**：Minimalism & Swiss（干净、留白、高对比、网格化）— 适配专业知识工具
- **主题**：草药绿（中药 thematic），暖米白背景
- **明暗**：Light（Streamlit 主题令牌见 `.streamlit/config.toml`）

## 色彩令牌（src/webui/styles.py COLORS）

| 角色 | 值 | 用途 |
|---|---|---|
| primary | `#2D6A4F` | 主按钮、强调、分数条 |
| primary_dark | `#1B4332` | 标题、深色文字 |
| primary_bg | `#D8F3DC` | 选中态、浅色底 |
| accent | `#B08968` | 装饰（琥珀金，降饱和度保对比度） |
| bg | `#FAF8F1` | 页面背景（暖米白） |
| surface | `#FFFFFF` | 卡片 |
| border | `#E0DDD5` | 边框 |
| text | `#1E293B` | 正文 |
| text_secondary | `#526074` | 次要文字（≥4.5:1 on bg） |
| citation | bg `#E6F4EA` / text `#1B4332` | 引用徽章（≈8:1） |
| error / warning | `#C81E1E` / `#B45309` | 功能色（均过 4.5:1） |

## 字体

系统中文字体栈，**禁止外链字体**（Google Fonts 国内不可达）：
`'Noto Sans SC', 'PingFang SC', 'HarmonyOS Sans SC', 'Microsoft YaHei', 'Source Han Sans SC', -apple-system, ...`
正文 16px / line-height 1.6；数字用 tabular-nums。

## 图标

一律 SVG：头像/Logo 在 `src/webui/assets/`（*.svg，Streamlit 原生支持）；
内联图标在 `components.py` 的 `ICONS` 字典（stroke=currentColor）。
**禁止 emoji 充当结构性图标**。

## 交互与无障碍（必须保持）

- 触摸目标 ≥ 44px（按钮/输入/导航项）
- `:focus-visible` 3px 焦点环，全局可见
- hover 过渡 200ms；`prefers-reduced-motion` 时禁用动画
- 对话区限宽 880px 居中（宽屏可读性）；窄屏 (<768px) 自适应收窄
- 对比度：正文 ≥ 4.5:1，大标题 ≥ 3:1

## 反模式（避免）

- emoji 当图标 / 外链字体 / 灰色配灰 /  raw hex 散落在组件里（用 COLORS 令牌）
- 单一 transition 时长通吃 / 无焦点态 / 横向滚动
