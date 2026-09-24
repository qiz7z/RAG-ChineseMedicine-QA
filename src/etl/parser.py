# -*- coding: utf-8 -*-
"""
药典文档解析器
================
从 .docx 文件中提取结构化的药品数据。

文档层级结构:
    药典一部
    └── 药品条目 (Heading #3)
        ├── 拼音名 (Body text|3)
        ├── 拉丁名 (Body text|3)
        ├── 来源/概述 (正文)
        ├── 【章节1】 (正文，标记嵌在段首)
        ├── 【章节2】
        │   └── 表格
        ├── 饮片 (子节，Heading #4 或 Body text|4)
        │   ├── 【炮制】
        │   └── 【性味与归经】...
        └── 子剂型 (Heading #4)
            ├── 拼音名
            ├── 【处方】
            └── 【制法】...
"""
import re
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import docx
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph


# ============================================================
# 缺陷 16：Word 文本框（<w:txbxContent>）内容读取
# ============================================================
# 现象：文档里有 1,020 个文本框，承载 20,517 字，包含 86 处【性味与归经】、
#       90 处【功能与主治】、100 处【处方】，而 python-docx 的段落流读不到它们
#       → 大黄（「苦，寒。归脾、胃、大肠、肝、心包经」「泻下攻积…」）、
#         丁公藤、巴豆、天南星、白头翁、白果等 83 个药材的核心章节整段丢失。
# 处理：把文本框内的段落**按锚点位置搬回正文流**，并压平两类噪声：
#       ① 同一行被「裁切后重复」2~4 次（mc:Choice/Fallback × 裁切）；
#       ② 标记本身带 OCR 残留（`1功能与主治】` / `t性味与归经】`）。
# 这样后续解析逻辑（段落流 + 【章节】状态机）无需任何改动。

_BOX_MARKER_NAMES = (
    '性味与归经', '性味', '功能与主治', '功能主治', '用法与用量', '注意', '贮藏',
    '处方', '制法', '规格', '鉴别', '检查', '性状', '炮制', '浸出物', '含量测定',
    '特征图谱', '指纹图谱', '制剂', '功能主治',
)
_BOX_MARKER_RE = re.compile(
    r'^[^【〔]{0,3}[【〔]?(' + '|'.join(_BOX_MARKER_NAMES) + r')[】〕]?$')


def _collapse_box_repeats(text: str) -> str:
    """把文本框里「同一行被裁切后重复」压回一行。

    实例：'辛，温;有小毒。归肝、脾、胃经。辛，温;有小毒。归肝、脾、胃'
          → '辛，温;有小毒。归肝、脾、胃经。'

    做法：找**最短**的重复单元 p，使文本 = p 的若干完整重复 + p 的前缀（裁切尾巴）。
    ⚠️ 单元长度可以**超过文本一半**（"整份 + 裁切尾巴"是最常见的形态），
       所以上界取 `n-3` 而不是 `n//2`——第一版用 `n//2` 时找不到单元，
       实测 8 个样例里 6 个没压平。
    """
    orig = (text or '').strip()
    t = re.sub(r'\s+', '', orig)          # 仅供重复检测：空格不参与比对
    n = len(t)
    if n < 6:
        return orig
    for size in range(3, n - 2):
        unit = t[:size]
        k = n // size
        if k < 1 or t[:size * k] != unit * k:
            continue
        tail = t[size * k:]
        if not tail or unit.startswith(tail):      # 尾巴是单元的前缀 = 被裁切
            # ⚠️ 按**原文**截取，保留行内空格（拉丁学名 'Rheum palmatum L.' 不能被压成
            #    'RheumpalmatumL.'——第一版整行去空格，把内容行写坏了）
            return _cut_at_nonspace(orig, size)
    return orig


def _cut_at_nonspace(text: str, k: int) -> str:
    """截取 text 的前 k 个**非空白**字符（含它们之间的空白）"""
    cnt = 0
    for i, ch in enumerate(text):
        if not ch.isspace():
            cnt += 1
            if cnt == k:
                return text[:i + 1]
    return text


# 标记行常被 OCR 打碎（`1功能与主治】11功能与主治】功能与主治】`），
# 判据：把章节名全部抠掉后，剩余字符几乎只有噪声（数字/字母/括号），
# 且章节名占整行一半以上 → 整行就是一个被打碎的标记。
_BOX_JUNK_CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ|｜[]!【〔】〕. '


def _dominant_marker(text: str) -> str:
    """若整行基本就是某个章节名（可含 OCR 噪声），返回该章节名，否则返回空串"""
    # ⚠️ 评分要用「覆盖字符数」= 次数 × 名字长度，不能只比次数：
    #    't性味与归经】tt性味与归经】性味与归【' 里「性味」出现 3 次(6 字)、
    #    「性味与归经」2 次(12 字)——只比次数会选中「性味」并把整行判成内容行。
    best, best_score = '', 0
    for name in _BOX_MARKER_NAMES:
        score = text.count(name) * len(name)
        if score > best_score:
            best, best_score = name, score
    if not best:
        return ''
    if best_score < len(text) * 0.5:               # 章节名占比不足一半 → 是内容行
        return ''
    rest = text
    for name in sorted(set(_BOX_MARKER_NAMES), key=len, reverse=True):
        rest = rest.replace(name, '')
    # 标记被裁切时行里会留一段**章节名前缀**（「…】性味与归【」）——一并清掉。
    # 只删长度 ≥3 的前缀，且删完还要看剩余是否为空，故不会伤到内容词
    # （如内容行「性味甘平」删掉「性味」后仍剩「甘平」→ 不算标记）。
    # 清掉章节名的**任意子串**（长度 ≥2，长优先）：OCR 打碎后残留的不只是前缀——
    # 实测 't性味与归经】tt性味与归经】性味与归【' 删掉完整名后会剩「与归」（中间段）。
    # 安全性由"删完必须什么都不剩"保证：内容行如「性味甘平」删掉「性味」后仍剩「甘平」。
    frags = set()
    for name in set(_BOX_MARKER_NAMES):
        for i in range(len(name)):
            for cut in range(2, len(name) - i + 1):
                frags.add(name[i:i + cut])
    for frag in sorted(frags, key=len, reverse=True):
        rest = rest.replace(frag, '')
    return best if rest.strip(_BOX_JUNK_CHARS) == '' else ''


def _norm_box_line(text: str) -> str:
    """规范文本框里的一行：压平裁切重复 + 补回 OCR 残缺的【章节】标记"""
    t = _collapse_box_repeats(text)
    if not t:
        return ''
    m = _BOX_MARKER_RE.match(t)
    if m:
        return f'【{m.group(1)}】'
    dominant = _dominant_marker(t)
    if dominant:
        return f'【{dominant}】'
    return t


def _collect_box_lines(anchor) -> list:
    """按文档顺序取出锚点块内所有文本框的**逻辑行**（去重、近重复保长）"""
    lines = []
    for tx in anchor.findall('.//' + qn('w:txbxContent')):
        for para in tx.findall('.//' + qn('w:p')):
            text = _norm_box_line(''.join(para.itertext()))
            if not text:
                continue
            for i, prev in enumerate(lines):
                if text == prev:
                    break
                if text.startswith(prev) or prev.startswith(text):
                    # 同一行的两个裁切变体 → 保留更长的那个
                    if len(text) > len(prev):
                        lines[i] = text
                    break
            else:
                lines.append(text)
    return lines


# 章节标记被 OCR 打碎的形式：前缀是「t / 1 / ［ / [: / ￡」等对【的误认，
# 收尾可能是 】/］/ ]。实测正文里有 1,004 个这样的段落：
#     t鉴别】（1）叶表面观…        （328 处）
#     ［含量测定】总糖取本品粗粉…    （109 处）
#     1功能与主治】                （74 处，标记独占一段、内容在下一段）
#     [:规格】 每粒装0.425g        （46 处）
# 解析器的章节正则要求段落以【或〔开头，这些段落因此**不被识别为章节**，
# 内容被并进上一章节、或在条目开头处整段丢失
# （54 个药材缺【性味与归经】的主因，其中 14 个就写着 `t性味与归经】`）。
_JUNK_MARKER_RE = re.compile(
    r'^[^【〔】〕]{0,4}[【〔\[]?(' + '|'.join(_BOX_MARKER_NAMES) + r')[】〕\]]')


def normalize_ocr_section_markers(doc) -> int:
    """把正文段落开头被打碎的章节标记补回【】（缺陷 17）。返回处理数。

    ⚠️ 只取段落**自己的** w:t（直接子 run 下的），不能用 `.//w:t`：
    锚点段落里可能嵌着文本框，用后者会把文本框内容一并清空。
    """
    body = doc.element.body
    n = 0
    for para in body.iter(qn('w:p')):
        ts = [t for r in para.findall(qn('w:r')) for t in r.findall(qn('w:t'))]
        if not ts:
            continue
        text = ''.join(t.text or '' for t in ts).strip()
        if not text:
            continue
        m = _JUNK_MARKER_RE.match(text)
        if not m:
            continue
        name = m.group(1)
        if text.startswith(f'【{name}】'):       # 已规范，跳过（幂等）
            continue
        rest = text[m.end():]
        ts[0].text = f'【{name}】{rest}'
        for t in ts[1:]:
            t.text = ''
        n += 1
    return n


def linearize_textboxes(doc) -> list:
    """把文本框内的段落搬进正文流（缺陷 16）。返回**插入的段落元素列表**。

    ⚠️ 返回元素列表（而不是行数）是必须的：解析器判断"药名"依赖
    「本段是药名 且 下一段是拼音」这个前瞻（`_next_is_pinyin`）。
    我们插入的段落会落在药名段与拼音段之间，把前瞻挡掉，
    导致该药名不再被认作条目边界、整条并进上一个药（实测 茜草 整条消失、
    内容跑到 荆芥穗炭 名下）。所以调用方要用这个列表让前瞻跳过它们。
    """
    body = doc.element.body
    inserted = []
    for anchor in list(body.iterchildren()):
        if anchor.tag not in (qn('w:p'), qn('w:tbl')):
            continue
        lines = _collect_box_lines(anchor)
        if not lines:
            continue
        cursor = anchor
        for text in lines:
            new_p = anchor.makeelement(qn('w:p'), {})
            new_r = new_p.makeelement(qn('w:r'), {})
            new_t = new_p.makeelement(qn('w:t'), {})
            new_t.text = text
            new_r.append(new_t)
            new_p.append(new_r)
            cursor.addnext(new_p)
            cursor = new_p
            inserted.append(new_p)
    return inserted


def iter_block_items(parent):
    """
    按文档顺序遍历段落和表格（python-docx 默认不保证顺序）。
    这是官方文档推荐的方法，用于保持段落与表格的原始顺序。
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def table_to_markdown(table: Table) -> str:
    """将 docx 表格转换为 Markdown 格式文本。"""
    if not table.rows:
        return ""
    lines = []
    headers = [cell.text.strip().replace('\n', ' ') for cell in table.rows[0].cells]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table.rows[1:]:
        cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
        while len(cells) < len(headers):
            cells.append("")
        lines.append("| " + " | ".join(cells[:len(headers)]) + " |")
    return "\n".join(lines)


def contains_chinese(text: str) -> bool:
    """判断字符串是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_pinyin_only(text: str) -> bool:
    """判断是否为纯拼音/拉丁名（不含中文）"""
    text = text.strip()
    if not text:
        return False
    # 只含拉丁字母、空格、连字符
    return bool(re.match(r'^[A-Za-z\s\-]+$', text)) and not contains_chinese(text)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Section:
    """药典中的一个章节，如【性状】、【鉴别】等"""
    section_name: str
    content: str
    table_markdown: Optional[str] = None
    raw_paragraphs: list = field(default_factory=list)


@dataclass
class DrugEntry:
    """一个药品条目（或子剂型条目）"""
    drug_name: str
    pinyin_name: str = ""
    latin_name: str = ""
    parent_drug: str = ""
    is_sub_formulation: bool = False
    is_yinpian: bool = False
    category_hint: str = ""
    intro_text: str = ""
    sections: list = field(default_factory=list)
    para_start: int = 0
    para_end: int = 0


# ============================================================
# 解析器
# ============================================================

# 章节标记正则：匹配 【xxx】 或 〔xxx〕
SECTION_PATTERN = re.compile(r'^[【〔](.+?)[】〕]')

# "饮片"子节关键词
YINPIAN_KEYWORDS = {'饮片', '饮 片'}

# 缺陷 15：可从 -饮片 子条目**回填**给主条目的章节。
# 药典排版惯例：【性味与归经】【功能与主治】【用法与用量】【注意】【贮藏】
# 出现在条目末尾（【饮片】块之后），对药材与其饮片一并适用；
# 解析器按"最后一个条目边界"归属，这些章节因此被挂到 -饮片 子条目上，
# 主条目反而缺失（实测 2,183 个主条目里 1,855 个受影响，85%）。
INHERITABLE_SECTIONS = {'性味与归经', '功能与主治', '用法与用量', '注意', '贮藏'}

# 已知的大类标题（不应被识别为药品名）
SECTION_HEADER_KEYWORDS = {
    '药材和饮片', '成方制剂和单味制剂', '植物油脂和提取物',
    '一部', '二部', '三部',
    '附录', '索引',
}

# 三个大类标题 → category 取值。
# 药典一部按「药材和饮片 → 植物油脂和提取物 → 成方制剂和单味制剂」的顺序编排，
# 因此**按大类标题判定分类**比按章节特征启发式可靠得多（见 docs/08 第七节）。
# 注意：文档里「药材和饮片」标题未必以独立段落出现，故默认值取该分类。
MAJOR_CATEGORY_HEADINGS = {
    '药材和饮片': '药材和饮片',
    '植物油脂和提取物': '植物油脂和提取物',
    '成方制剂和单味制剂': '成方制剂和单味制剂',
}


def split_soft_lines(text: str) -> list:
    """按**软换行**把段落文本拆成若干逻辑行。

    Word 里同一段内可用 `<w:br/>` 换行，python-docx 将其渲染为 `'\\n'`。
    药典中有 6 个药品的「药名 / 拼音 / 拉丁名」三行**挤在同一个段落**里
    （八角茴香、自然铜、黄蜀葵花、黄精、甜瓜子、款冬花），例：

        '八角茴香\\nBajiaohuixiang\\nANISI STELLATI FRUCTUS'

    若拿整段文本做药名判断，`_is_likely_drug_name` 会因长度超标而否决
    （它只去空格、不去换行），于是：
      ① 该药品**不成条目**（6 个药品整条丢失）
      ② 其正文被追加进上一个药品的章节里（跨药品串块）
    因此凡涉及"药名判断"的地方都必须先按行拆开。见 docs/09。
    """
    return [ln.strip() for ln in (text or '').split('\n') if ln.strip()]

# 除 Heading #3/#4 外，可能包含药品名的段落样式
# 文档中药品名使用的样式非常多样，以下样式均已被验证包含药品名条目：
# - Body text|4: 甘草、茯苓、麻黄、柴胡、地黄、泽泻、附子、黄芪(OCR为黄茂)、黄芩(OCR为黄苔)等
# - Body text|5: 人参等（之前已支持）
# - Body text|6: 川木通等
# - Heading #1|1: 一捻金等（之前已支持）
# - Heading #2|1: 山药、山豆根、山茱萸等
# - Heading #5|1: 半夏、半边莲等（也常用于拼音名，但含中文时为药品名）
# - Normal: 部分药品名（之前已支持）
# - Body text|3: 七味铁屑丸、七味葡萄散、七味广枣丸、七珍丸、八正合剂、八珍益母丸 等
#   该样式**主要**用于拼音/拉丁名行，但含中文时就是药品名——与 Heading #5|1 同理。
#   （历史实现把它排除在外，导致这批成方制剂既不成条目、正文还被吞进上一个药品；
#    _is_likely_drug_name 要求含中文，因此纳入不会让拼音行被误判成药名。）
EXTRA_DRUG_NAME_STYLES = {
    'Body text|3', 'Body text|4', 'Body text|5', 'Body text|6',
    'Heading #1|1', 'Heading #2', 'Heading #5',
    'Normal',
}


class PharmacopoeiaParser:
    """药典文档结构化解析器"""

    def __init__(self, docx_path: str):
        self.docx_path = docx_path
        self.doc = docx.Document(docx_path)
        # 缺陷 17：先补回被打碎的章节标记（在遍历块之前，让章节正则能识别）
        self._markers_fixed = normalize_ocr_section_markers(self.doc)
        # 缺陷 16：再把文本框内容搬进正文流，然后遍历块
        _box_elems = linearize_textboxes(self.doc)
        self._textbox_lines_moved = len(_box_elems)
        # 这些是"只为承载内容"的合成段落：前瞻与边界判定必须跳过它们
        self._box_elem_ids = {id(e) for e in _box_elems}
        self.blocks = list(iter_block_items(self.doc))
        self._block_indices = []
        para_idx = 0
        table_idx = 0
        for b in self.blocks:
            if isinstance(b, Paragraph):
                self._block_indices.append(('para', para_idx, b))
                para_idx += 1
            else:
                self._block_indices.append(('table', table_idx, b))
                table_idx += 1
        self._last_main_drug_name = ""  # 追踪最近的主药品名
        # 追踪当前所处的**大类**（按文档顺序推进），用于判定 category。
        # _saw_category_heading=False 时说明文档没有大类标题（如合成测试文档），
        # 此时回退到 _infer_category 的启发式。
        self._current_category = "药材和饮片"
        self._saw_category_heading = False

    def _is_likely_drug_name(self, text: str) -> bool:
        """
        启发式判断文本是否像药品名（而非章节标题、正文片段等）。
        条件：
        1. 包含中文
        2. 去空格后 2~15 个字符
        3. 不是章节标记（不以【或〔开头）
        4. 不是"饮片"关键词
        5. 不是已知大类标题
        6. 不含数字
        7. 不含"质量标准"等非药品名关键词
        """
        text = text.strip()
        if not text:
            return False
        if not contains_chinese(text):
            return False
        compact = text.replace(' ', '').replace('\u3000', '')
        if len(compact) < 2 or len(compact) > 15:
            return False
        if SECTION_PATTERN.match(text):
            return False
        if text in YINPIAN_KEYWORDS:
            return False
        if text in SECTION_HEADER_KEYWORDS:
            return False
        if '质量标准' in text:
            return False
        if re.search(r'\d', text):
            return False
        return True

    def _next_is_pinyin(self, idx: int) -> bool:
        """
        检查 idx 之后最近的非空段落是否为拼音/拉丁名。
        药品名后通常紧跟拼音名（纯 ASCII 文本），这是判断药品名的关键上下文特征。
        """
        total = len(self._block_indices)
        seen = 0
        for j in range(idx + 1, total):
            btype, bidx, block = self._block_indices[j]
            if btype != 'para':
                return False
            if self._is_box_block(block):       # 文本框合成段落不参与前瞻
                continue
            text = block.text.strip()
            if not text:
                continue
            if is_pinyin_only(text):
                return True
            seen += 1
            if seen >= 2:                       # 只看最近的两个非空段落
                return False
        return False

    def _is_box_block(self, block) -> bool:
        """该块是否为「文本框内容线性化」插入的合成段落"""
        return id(getattr(block, '_element', None)) in getattr(self, '_box_elem_ids', ())

    def _is_entry_boundary(self, style_name: str, text: str, block_idx: int = -1) -> str:
        """
        判断当前段落是否为药品条目边界。
        
        Returns:
            'main'  - 主条目
            'sub'   - 子剂型 (Heading #4)
            'yinpian' - 饮片子节
            ''      - 非边界
        """
        text = text.strip()
        if not text:
            return ''

        # 段落内可能有软换行（w:br）：药名 / 拼音 / 拉丁名 挤在同一段里。
        # 一律按**首行**做药名判断；若本段自带拼音行，则等价于"后跟拼音名"。
        lines = split_soft_lines(text)
        head = lines[0] if lines else ''
        inline_pinyin = len(lines) >= 2 and is_pinyin_only(lines[1])

        def _has_pinyin() -> bool:
            return inline_pinyin or (block_idx >= 0 and self._next_is_pinyin(block_idx))

        # "饮片"关键词 —— 检查所有可能出现的样式
        if head in YINPIAN_KEYWORDS:
            if any(s in style_name for s in (
                'Heading #3', 'Heading #4', 'Body text|4', 'Body text|5'
            )):
                return 'yinpian'

        # 软换行段落：首行像药名、次行是拼音 → 「药名/拼音/拉丁名同段」的条目头。
        # 这类段落的样式在文档里并不统一（实测有 Body text|3、Body text|4、
        # Heading #3|1、Heading #5|1），而 Body text|3 在别处是"拼音样式"、
        # 不在 EXTRA_DRUG_NAME_STYLES 里，所以这里必须**脱离样式**单独判断。
        if len(lines) >= 2 and inline_pinyin and self._is_likely_drug_name(head):
            return 'main'

        # Heading #3: 主条目边界（原始逻辑）
        if 'Heading #3' in style_name:
            if is_pinyin_only(head):
                return ''
            if contains_chinese(head):
                return 'main'
            return ''

        # Heading #4: 子剂型 / 有时也是主条目
        if 'Heading #4' in style_name:
            if contains_chinese(head):
                # 如果下一段（或本段第二行）是拼音名 → 很可能是主条目（而非子剂型）
                if _has_pinyin():
                    return 'main'
                return 'sub'
            return ''

        # 额外样式：Body text|5, Heading #1|1, Normal
        # 文档中大量药品名使用这些样式（如"人 参"在 Body text|5，"一捻金"在 Heading #1|1）
        if block_idx >= 0:
            for style_pattern in EXTRA_DRUG_NAME_STYLES:
                if style_pattern in style_name:
                    if self._is_likely_drug_name(head) and _has_pinyin():
                        return 'main'
                    break

        return ''

    def parse(self) -> list:
        """
        解析整个文档，返回 DrugEntry 列表。
        """
        entries = []
        i = 0
        total = len(self._block_indices)

        while i < total:
            btype, bidx, block = self._block_indices[i]

            if btype == 'para':
                style_name = block.style.name if block.style else ''
                text = block.text.strip()

                if not text:
                    i += 1
                    continue

                # 大类标题（药材和饮片 / 植物油脂和提取物 / 成方制剂和单味制剂）：
                # 按文档顺序推进当前分类，标题本身不是药品条目，跳过。
                # 去空格后匹配，兼容「药 材和饮片」这类排版。
                _compact = text.replace(' ', '').replace('\u3000', '')
                if _compact in MAJOR_CATEGORY_HEADINGS:
                    self._current_category = MAJOR_CATEGORY_HEADINGS[_compact]
                    self._saw_category_heading = True
                    i += 1
                    continue

                # 合成段落只承载内容，永不作为条目边界（缺陷 16 防误判）
                boundary = '' if self._is_box_block(block) else \
                    self._is_entry_boundary(style_name, text, i)

                if boundary == 'main':
                    # 取首行：药名/拼音/拉丁名可能同处一段（软换行），
                    # 若用整段文本，饮片子条目会被命名成
                    # 「黄蜀葵花\nHuangshukuihua\nABELMOSCHICOROLLA-饮片」这种脏名字。
                    self._last_main_drug_name = split_soft_lines(text)[0]
                    entry, next_i = self._parse_drug_entry(i, is_sub=False)
                    if entry:
                        entries.append(entry)
                    i = next_i
                    continue

                elif boundary == 'sub':
                    entry, next_i = self._parse_drug_entry(i, is_sub=True)
                    if entry:
                        entry.parent_drug = self._last_main_drug_name
                        entries.append(entry)
                    i = next_i
                    continue

                elif boundary == 'yinpian':
                    entry, next_i = self._parse_drug_entry(i, is_sub=True, is_yinpian=True)
                    if entry:
                        # 饮片子节命名：父药品名-饮片
                        entry.drug_name = f"{self._last_main_drug_name}-饮片" if self._last_main_drug_name else "饮片"
                        entry.parent_drug = self._last_main_drug_name
                        entries.append(entry)
                    i = next_i
                    continue

            i += 1

        self._inherit_sections_from_yinpian(entries)
        return entries

    def _inherit_sections_from_yinpian(self, entries: list) -> int:
        """把 -饮片 子条目独有的核心章节回填给主条目（缺陷 15）。

        实例：天麻主条目只有 性状/鉴别/检查/含量测定/浸出物/特征图谱，
        而【性味与归经】【功能与主治】【用法与用量】【贮藏】全在 `天麻-饮片` 上。
        后果：查询「天麻的性味归经」解析出章节[性味与归经]，语料里却没有
        天麻主条目的承载切片，检索退化为 药品概要/含量测定，生成只好答
        "资料未收录"——生成侧 8 道覆盖率 0 的题大多源于此。

        规则：
        - 只回填主条目（非饮片、非子剂型）**完全缺失**的章节，不覆盖已有；
        - 饮片侧内容若是「同药材。」这类占位文本则跳过（⚠️ 不能用长度阈值：
          真实的性味与归经常只有「甘，平。归肝经。」8 个字，粗阈值会误杀）；
        - 拷贝的是 Section 的浅拷贝，两个条目各自持有，互不影响。
        """
        import copy as _copy
        yinpian = {}
        for e in entries:
            if e.is_yinpian and e.drug_name.endswith('-饮片'):
                yinpian[e.drug_name] = e

        fixed = 0
        for e in entries:
            if e.is_yinpian or e.is_sub_formulation or e.parent_drug:
                continue                                   # 只处理主条目
            child = yinpian.get(f"{e.drug_name}-饮片")
            if child is None:
                continue
            have = {s.section_name for s in e.sections}
            missing = INHERITABLE_SECTIONS - have
            if not missing:
                continue
            for s in list(child.sections):
                if s.section_name not in missing:
                    continue
                content = (s.content or '').strip()
                # ⚠️ 不能用 len>=15 这类粗阈值：药典真实的短章节很多——
                #    性味与归经常只有「甘，平。归肝经。」(8字)、用法与用量「3〜9g。」(5字)。
                #    真正要挡的只有两种：空文本、以及「同药材。」占位（拷了无信息量）。
                compact = re.sub(r"[\s，。、；；:：]", "", content)
                if len(compact) < 3 or compact == "同药材":
                    continue
                e.sections.append(_copy.copy(s))
                missing.discard(s.section_name)
                fixed += 1
        return fixed

    def _parse_drug_entry(self, start_idx: int, is_sub: bool, is_yinpian: bool = False) -> tuple:
        """
        解析一个药品条目（从药品名称到下一个药品名称之间）。
        """
        btype, bidx, name_block = self._block_indices[start_idx]
        name_lines = split_soft_lines(name_block.text)
        drug_name = name_lines[0] if name_lines else name_block.text.strip()

        entry = DrugEntry(
            drug_name=drug_name,
            is_sub_formulation=is_sub,
            is_yinpian=is_yinpian,
            para_start=bidx,
        )
        # 药名/拼音/拉丁名挤在同一个段落（软换行）时，后两行直接取自本段；
        # 否则这两行会走"非章节段落"分支被追加进正文，产生脏数据。
        if len(name_lines) >= 2:
            entry.pinyin_name = name_lines[1]
        if len(name_lines) >= 3:
            entry.latin_name = name_lines[2]

        i = start_idx + 1
        total = len(self._block_indices)
        current_section = None
        collected_intro = []

        while i < total:
            btype, bidx2, block = self._block_indices[i]

            if btype == 'para':
                style_name = block.style.name if block.style else ''
                text = block.text.strip()

                if not text:
                    i += 1
                    continue

                # 大类标题必须交回**外层**循环处理（用于切换当前分类）。
                # 注意：_is_entry_boundary 不会把它当边界（它在 SECTION_HEADER_KEYWORDS
                # 里被显式排除，避免被误认成药品名），若不在此显式 break，
                # 标题会被内层循环当作正文吞掉，外层永远看不到 → 分类切换失效。
                if text.replace(' ', '').replace('\u3000', '') in MAJOR_CATEGORY_HEADINGS:
                    break

                # 检查是否到达下一个条目边界
                boundary = self._is_entry_boundary(style_name, text, i)
                if boundary:  # main / sub / yinpian
                    break

                # 拼音名（紧跟药品名后的各种样式）
                if not current_section:
                    # Body text|3: 常见拼音/拉丁名样式
                    if 'Body text|3' in style_name:
                        if not entry.pinyin_name:
                            entry.pinyin_name = text
                        elif not entry.latin_name:
                            entry.latin_name = text
                        i += 1
                        continue
                    # Heading #5: 也常用于拼音名
                    if 'Heading #5' in style_name:
                        if not entry.pinyin_name:
                            entry.pinyin_name = text
                        i += 1
                        continue
                    # Body text|4: 有时用于拼音名
                    if 'Body text|4' in style_name and is_pinyin_only(text):
                        if not entry.pinyin_name:
                            entry.pinyin_name = text
                        elif not entry.latin_name:
                            entry.latin_name = text
                        i += 1
                        continue
                    # Heading #4: 有时用于拉丁名（如 STACHYURI MEDULLA）
                    if 'Heading #4' in style_name and is_pinyin_only(text):
                        if not entry.latin_name:
                            entry.latin_name = text
                        elif not entry.pinyin_name:
                            entry.pinyin_name = text
                        i += 1
                        continue

                # 检测章节标记
                section_match = SECTION_PATTERN.match(text)
                if section_match:
                    if current_section:
                        entry.sections.append(current_section)

                    section_name = section_match.group(1)
                    content_after = text[len(section_match.group(0)):].strip()

                    current_section = Section(
                        section_name=section_name,
                        content=content_after,
                        raw_paragraphs=[text],
                    )
                    i += 1
                    continue

                # 非章节段落
                if current_section is None:
                    collected_intro.append(text)
                else:
                    if current_section.content:
                        current_section.content += '\n' + text
                    else:
                        current_section.content = text
                    current_section.raw_paragraphs.append(text)

            elif btype == 'table':
                table_md = table_to_markdown(block)

                # 表格首格可能自带【章节】标记——成方制剂的【处方】就是这种形式
                # （全文共 556 张），首格形如「【处方】醋香附138g」。
                # 历史实现只在 current_section 已存在时把表格挂到"上一个章节"上，
                # 而【处方】表紧跟在拼音名之后、此时 current_section 仍为 None，
                # 于是整张处方表被静默丢弃：530 个成方制剂的处方内容不在知识库里，
                # 且因缺少【处方】信号被 _infer_category 误判为"药材和饮片"。
                # 详见 docs/08 与 tests/test_etl_parser.py
                head = block.rows[0].cells[0].text.strip() if block.rows else ''
                head_match = SECTION_PATTERN.match(head)
                if head_match:
                    if current_section:
                        entry.sections.append(current_section)
                    current_section = Section(
                        section_name=head_match.group(1).strip(),
                        content=table_md,
                        table_markdown=table_md,
                        raw_paragraphs=[head],
                    )
                elif current_section:
                    if current_section.table_markdown:
                        current_section.table_markdown += '\n\n' + table_md
                    else:
                        current_section.table_markdown = table_md

            i += 1

        if current_section:
            entry.sections.append(current_section)

        entry.intro_text = '\n'.join(collected_intro) if collected_intro else ''
        entry.para_end = bidx
        # 分类判定：优先用**大类标题**（文档按大类顺序编排，权威）。
        # 只有在文档没有任何大类标题时（如合成测试文档）才回退到启发式。
        entry.category_hint = (
            self._current_category if self._saw_category_heading
            else self._infer_category(entry)
        )

        return entry, i

    def _infer_category(self, entry: DrugEntry) -> str:
        """按章节特征**启发式**推断分类（仅作无大类标题时的回退）。

        ⚠️ 该启发式并不可靠：它以「有【处方】/【规格】→成方制剂、有【来源】/
        【性味与归经】→药材和饮片」判断，两头都不命中就落到兜底 `药材和饮片`。
        实测与按大类标题的判定不一致 103/2387，且多数是不一致方更错
        （如 人参总皂苷 / 肉桂油 / 松节油 属"植物油脂和提取物"却被判为"药材和饮片"）。
        正常解析路径请使用 `_current_category`，见 docs/08 第七节。
        """
        section_names = {s.section_name for s in entry.sections}

        if entry.is_yinpian:
            return "药材和饮片"
        if '处方' in section_names or '规格' in section_names:
            return "成方制剂和单味制剂"
        if '来源' in section_names or '性味与归经' in section_names:
            return "药材和饮片"
        if any(kw in entry.drug_name for kw in ['流浸膏', '浸膏', '提取物']):
            return "植物油脂和提取物"
        if entry.is_sub_formulation:
            return "成方制剂和单味制剂"
        return "药材和饮片"

    def to_json(self, entries: list, output_path: str):
        """将解析结果保存为JSON文件"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in entries]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(entries)} 个药品条目到 {output_path}")
        return data


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import sys
    from collections import Counter
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import RAW_DOCX_PATH, DRUGS_JSON_PATH

    print("=" * 60)
    print("药典文档解析器")
    print("=" * 60)

    parser = PharmacopoeiaParser(RAW_DOCX_PATH)
    entries = parser.parse()

    print(f"\n解析完成，共提取 {len(entries)} 个药品条目")

    main_entries = [e for e in entries if not e.is_sub_formulation]
    sub_entries = [e for e in entries if e.is_sub_formulation and not e.is_yinpian]
    yinpian_entries = [e for e in entries if e.is_yinpian]

    print(f"  主条目（药材/制剂）: {len(main_entries)}")
    print(f"  子剂型条目: {len(sub_entries)}")
    print(f"  饮片子节: {len(yinpian_entries)}")

    cat_counter = Counter(e.category_hint for e in entries)
    print(f"\n分类统计:")
    for cat, cnt in cat_counter.most_common():
        print(f"  {cat}: {cnt}")

    section_counter = Counter()
    total_sections = 0
    for e in entries:
        for s in e.sections:
            section_counter[s.section_name] += 1
            total_sections += 1
    print(f"\n章节总数: {total_sections}")
    print(f"章节类型数: {len(section_counter)}")
    print("前15个章节:")
    for name, cnt in section_counter.most_common(15):
        print(f"  【{name}】: {cnt}次")

    # 含表格的章节统计
    table_count = sum(1 for e in entries for s in e.sections if s.table_markdown)
    print(f"\n含表格的章节: {table_count}")

    parser.to_json(entries, DRUGS_JSON_PATH)

    # 样例展示
    print(f"\n{'='*60}")
    print("样例展示:")
    print(f"{'='*60}")

    # 药材样例
    for e in entries:
        if (not e.is_sub_formulation and e.category_hint == "药材和饮片" 
                and len(e.sections) >= 6 and e.intro_text):
            print(f"\n--- {e.drug_name} ({e.category_hint}) ---")
            print(f"  拼音: {e.pinyin_name}")
            print(f"  拉丁: {e.latin_name}")
            print(f"  概述: {e.intro_text[:100]}...")
            for s in e.sections[:8]:
                content_preview = s.content[:80] if s.content else "(空)"
                print(f"  【{s.section_name}】 {content_preview}...")
            break

    # 成方制剂样例
    for e in entries:
        if (e.category_hint == "成方制剂和单味制剂" and not e.is_sub_formulation
                and len(e.sections) >= 4):
            print(f"\n--- {e.drug_name} ({e.category_hint}) ---")
            for s in e.sections[:8]:
                content_preview = s.content[:80] if s.content else "(空)"
                has_table = " [含表格]" if s.table_markdown else ""
                print(f"  【{s.section_name}】{has_table} {content_preview}...")
            break

    # 饮片样例
    for e in entries:
        if e.is_yinpian and len(e.sections) >= 2:
            print(f"\n--- {e.drug_name} (饮片) ---")
            for s in e.sections[:5]:
                content_preview = s.content[:80] if s.content else "(空)"
                print(f"  【{s.section_name}】 {content_preview}...")
            break
