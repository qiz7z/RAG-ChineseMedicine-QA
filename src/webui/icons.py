# -*- coding: utf-8 -*-
"""
SVG 图标资源
============
两类图标：

1. ``ICONS``（components.py 使用）：内联 SVG 字符串，stroke=currentColor，
   可继承父元素文字颜色，直接插入 HTML。
2. ``NAV_ICONS``（styles.py 使用）：带固定颜色的 SVG，经 base64 编码后
   供 CSS ``content: url()`` 使用（侧边栏导航项、示例问题卡片）。

全部手绘几何图形，不用 emoji。
"""
import base64

# ============================================================
# 一、内联图标（currentColor，用于 HTML 插入）
# ============================================================

def _svg(paths: str, size: int = 16) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align: -3px;">{paths}</svg>'
    )


ICONS = {
    # 链接（检索来源）
    "link": _svg(
        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
        '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'
    ),
    # 盾牌（用药安全）
    "shield": _svg(
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
        '<path d="M12 8v4"/><path d="M12 16h.01"/>'
    ),
    # 文档（知识切片）
    "doc": _svg(
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h5"/>'
    ),
    # 叶片（收录药品）
    "leaf": _svg(
        '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/>'
        '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>'
    ),
    # 数据库（向量索引）
    "database": _svg(
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/>'
        '<path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>'
    ),
    # 芯片（LLM 模型）
    "chip": _svg(
        '<rect x="4" y="4" width="16" height="16" rx="2"/>'
        '<rect x="9" y="9" width="6" height="6"/>'
        '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>'
    ),
    # 警告三角（一致性校验）
    "alert": _svg(
        '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    # 搜索（空状态）
    "search": _svg(
        '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'
    ),
    # 对话（Hero/示例问题）
    "chat": _svg(
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
        '<path d="M8 9h8M8 13h5"/>'
    ),
    # 火花（Hero 主标识）
    "spark": _svg(
        '<path d="M12 3l1.9 5.8L20 10.7l-5.1 2.4L12 19l-2.9-5.9L4 10.7l6.1-1.9z"/>'
    ),
    # 混合检索（双路）
    "git-branch": _svg(
        '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>'
        '<path d="M6 9v0a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3"/>'
        '<path d="M18 15v0a3 3 0 0 1-3 3H9"/>'
    ),
    # 溯源（书签）
    "bookmark": _svg(
        '<path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>'
    ),
    # 多轮对话（历史）
    "history": _svg(
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>'
    ),
    # 对勾
    "check": _svg(
        '<path d="M20 6 9 17l-5-5"/>'
    ),
    # 图表（统计）
    "activity": _svg(
        '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
    ),
    # 信息
    "info": _svg(
        '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'
    ),
}


# ============================================================
# 二、CSS content:url() 图标（固定颜色 #2D6A4F）
# ============================================================

_NAV_PATHS = {
    "icon_nav_chat": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
        '<path d="M8 9h8M8 13h5"/>'
    ),
    "icon_nav_search": (
        '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'
    ),
    "icon_nav_leaf": (
        '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/>'
        '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>'
    ),
    "icon_nav_activity": (
        '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
    ),
    "icon_nav_info": (
        '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'
    ),
}


def _nav_svg(name: str) -> str:
    """生成带固定主色的导航 SVG 字符串"""
    paths = _NAV_PATHS[name]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
        'viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )


def nav_icon_data_uri(name: str) -> str:
    """导航图标 → base64 data URI（供 CSS content:url() 使用）"""
    return base64.b64encode(_nav_svg(name).encode("utf-8")).decode("ascii")
