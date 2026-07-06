# -*- coding: utf-8 -*-
"""全面诊断：查看所有被用作药品名但样式不是 Heading #3 的段落"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import docx
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
from config import RAW_DOCX_PATH

def iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

doc = docx.Document(RAW_DOCX_PATH)
blocks = list(iter_block_items(doc))

# 统计所有段落样式
from collections import Counter
style_counter = Counter()
for block in blocks:
    if isinstance(block, Paragraph):
        style = block.style.name if block.style else '(none)'
        style_counter[style] += 1

print("=== 所有段落样式统计 ===")
for style, count in style_counter.most_common():
    print(f"  {style:30s}: {count}")

# 找出所有 Body text|5 样式且含中文的短文本（疑似药品名）
print("\n=== Body text|5 样式含中文的短文本（疑似药品名）===")
bt5_suspects = []
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        style = block.style.name if block.style else ''
        text = block.text.strip()
        if 'Body text|5' in style and contains_chinese(text) and len(text) <= 10:
            bt5_suspects.append((i, style, text))
            if len(bt5_suspects) <= 80:
                # 看看后面跟着什么
                next_styles = []
                for j in range(i+1, min(len(blocks), i+4)):
                    if isinstance(blocks[j], Paragraph):
                        s = blocks[j].style.name if blocks[j].style else ''
                        t = blocks[j].text.strip()[:50]
                        next_styles.append(f"  → [{j}] {s}: {t}")
                print(f"  [{i:5d}] {text}")
                for ns in next_styles:
                    print(f"    {ns}")

print(f"\n总共 {len(bt5_suspects)} 个 Body text|5 疑似药品名")

# 也看看 Body text|4 的
print("\n=== Body text|4 样式含中文的短文本（前30个）===")
bt4_count = 0
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        style = block.style.name if block.style else ''
        text = block.text.strip()
        if 'Body text|4' in style and contains_chinese(text) and len(text) <= 10:
            bt4_count += 1
            if bt4_count <= 30:
                for j in range(i, min(len(blocks), i+3)):
                    if isinstance(blocks[j], Paragraph):
                        s = blocks[j].style.name if blocks[j].style else ''
                        t = blocks[j].text.strip()[:50]
                        marker = " <<<" if j == i else ""
                        print(f"  [{j:5d}] {s:20s}: {t}{marker}")

print(f"\n总共 {bt4_count} 个 Body text|4 疑似药品名")

# 看看 Heading #4 的情况
print("\n=== Heading #4 样式（前30个）===")
h4_count = 0
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        style = block.style.name if block.style else ''
        text = block.text.strip()
        if 'Heading #4' in style and text:
            h4_count += 1
            if h4_count <= 30:
                print(f"  [{i:5d}] {style:20s}: {text[:60]}")

print(f"\n总共 {h4_count} 个 Heading #4 段落")
