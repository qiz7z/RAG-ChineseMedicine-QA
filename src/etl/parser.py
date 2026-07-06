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
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph


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

# 已知的大类标题（不应被识别为药品名）
SECTION_HEADER_KEYWORDS = {
    '药材和饮片', '成方制剂和单味制剂', '植物油脂和提取物',
    '一部', '二部', '三部',
    '附录', '索引',
}

# 除 Heading #3/#4 外，可能包含药品名的段落样式
# 文档中药品名使用的样式非常多样，以下样式均已被验证包含药品名条目：
# - Body text|4: 甘草、茯苓、麻黄、柴胡、地黄、泽泻、附子、黄芪(OCR为黄茂)、黄芩(OCR为黄苔)等
# - Body text|5: 人参等（之前已支持）
# - Body text|6: 川木通等
# - Heading #1|1: 一捻金等（之前已支持）
# - Heading #2|1: 山药、山豆根、山茱萸等
# - Heading #5|1: 半夏、半边莲等（也常用于拼音名，但含中文时为药品名）
# - Normal: 部分药品名（之前已支持）
EXTRA_DRUG_NAME_STYLES = {
    'Body text|4', 'Body text|5', 'Body text|6',
    'Heading #1|1', 'Heading #2', 'Heading #5',
    'Normal',
}


class PharmacopoeiaParser:
    """药典文档结构化解析器"""

    def __init__(self, docx_path: str):
        self.docx_path = docx_path
        self.doc = docx.Document(docx_path)
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
        for j in range(idx + 1, min(total, idx + 3)):
            btype, bidx, block = self._block_indices[j]
            if btype != 'para':
                return False
            text = block.text.strip()
            if not text:
                continue
            return is_pinyin_only(text)
        return False

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

        # "饮片"关键词 —— 检查所有可能出现的样式
        if text in YINPIAN_KEYWORDS:
            if any(s in style_name for s in (
                'Heading #3', 'Heading #4', 'Body text|4', 'Body text|5'
            )):
                return 'yinpian'

        # Heading #3: 主条目边界（原始逻辑）
        if 'Heading #3' in style_name:
            if is_pinyin_only(text):
                return ''
            if contains_chinese(text):
                return 'main'
            return ''

        # Heading #4: 子剂型 / 有时也是主条目
        if 'Heading #4' in style_name:
            if contains_chinese(text):
                # 如果下一段是拼音名 → 很可能是主条目（而非子剂型）
                if block_idx >= 0 and self._next_is_pinyin(block_idx):
                    return 'main'
                return 'sub'
            return ''

        # 额外样式：Body text|5, Heading #1|1, Normal
        # 文档中大量药品名使用这些样式（如"人 参"在 Body text|5，"一捻金"在 Heading #1|1）
        if block_idx >= 0:
            for style_pattern in EXTRA_DRUG_NAME_STYLES:
                if style_pattern in style_name:
                    if self._is_likely_drug_name(text) and self._next_is_pinyin(block_idx):
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

                boundary = self._is_entry_boundary(style_name, text, i)

                if boundary == 'main':
                    self._last_main_drug_name = text
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

        return entries

    def _parse_drug_entry(self, start_idx: int, is_sub: bool, is_yinpian: bool = False) -> tuple:
        """
        解析一个药品条目（从药品名称到下一个药品名称之间）。
        """
        btype, bidx, name_block = self._block_indices[start_idx]
        drug_name = name_block.text.strip()

        entry = DrugEntry(
            drug_name=drug_name,
            is_sub_formulation=is_sub,
            is_yinpian=is_yinpian,
            para_start=bidx,
        )

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
                if current_section:
                    table_md = table_to_markdown(block)
                    if current_section.table_markdown:
                        current_section.table_markdown += '\n\n' + table_md
                    else:
                        current_section.table_markdown = table_md

            i += 1

        if current_section:
            entry.sections.append(current_section)

        entry.intro_text = '\n'.join(collected_intro) if collected_intro else ''
        entry.para_end = bidx
        entry.category_hint = self._infer_category(entry)

        return entry, i

    def _infer_category(self, entry: DrugEntry) -> str:
        """根据内容特征推断药品分类"""
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
