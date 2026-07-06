# -*- coding: utf-8 -*-
"""搜索可能存在 OCR 错误的药材名"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import docx
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

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

doc = docx.Document('data/raw/2020年药典一部.docx')

# 搜索可能被 OCR 错误的药材
# 黄芪 -> 黄茂? 黄氏? 黄芭?
# 川芎 -> 川弯? 川夸? 川菖?
# 薏苡仁 -> 薏苡仁? 薏以仁?
# 黄芩 -> 黄苓? 黄琴?

search_terms = [
    # 黄芪变体
    '黄 茂', '黄茂', '黄 氏', '黄氏', '黄 芭', '黄芭', '黄 芪', '黄芪',
    # 川芎变体  
    '川 弯', '川弯', '川 夸', '川夸', '川 菖', '川菖', '川 芎', '川芎',
    # 薏苡仁变体
    '薏 苡 仁', '薏苡仁', '薏 以 仁', '薏以仁', '薏 苡', '薏苡',
    # 黄芩变体
    '黄 芩', '黄芩', '黄 苓', '黄苓', '黄 琴', '黄琴',
]

paragraphs = []
for i, block in enumerate(iter_block_items(doc)):
    if isinstance(block, Paragraph):
        text = block.text.strip()
        style = block.style.name if block.style else 'None'
        paragraphs.append((i, text, style))

print(f"文档总段落数: {len(paragraphs)}\n")

for term in search_terms:
    for idx, text, style in paragraphs:
        if term in text and len(text) < 30:
            print(f"搜索 '{term}': [{idx}] 样式='{style}' 文本='{text}'")

# 额外：搜索所有以"黄"开头的短段落（可能是药品名）
print("\n=== 以'黄'开头的短段落（<10字符）===")
for idx, text, style in paragraphs:
    compact = text.replace(' ', '').replace('\u3000', '')
    if compact.startswith('黄') and len(compact) <= 8 and compact != '黄':
        print(f"  [{idx}] 样式='{style}' 文本='{text}'")

print("\n=== 以'川'开头的短段落（<10字符）===")
for idx, text, style in paragraphs:
    compact = text.replace(' ', '').replace('\u3000', '')
    if compact.startswith('川') and len(compact) <= 8 and compact != '川':
        print(f"  [{idx}] 样式='{style}' 文本='{text}'")

print("\n=== 以'薏'开头的短段落（<10字符）===")
for idx, text, style in paragraphs:
    compact = text.replace(' ', '').replace('\u3000', '')
    if compact.startswith('薏') and len(compact) <= 8:
        print(f"  [{idx}] 样式='{style}' 文本='{text}'")

# 搜索 Body text|4 样式的短段落（药品名候选）
print("\n=== Body text|4 样式且长度<10的段落（前50个）===")
count = 0
for idx, text, style in paragraphs:
    compact = text.replace(' ', '').replace('\u3000', '')
    if 'Body text|4' in style and len(compact) <= 10 and compact:
        print(f"  [{idx}] 文本='{text}' compact='{compact}'")
        count += 1
        if count >= 50:
            break

# 搜索 Heading #2 样式的短段落
print("\n=== Heading #2 样式且长度<10的段落 ===")
for idx, text, style in paragraphs:
    compact = text.replace(' ', '').replace('\u3000', '')
    if 'Heading #2' in style and len(compact) <= 10 and compact:
        print(f"  [{idx}] 文本='{text}' compact='{compact}'")

# 搜索 Heading #5 样式的短段落
print("\n=== Heading #5 样式且长度<10的段落（前50个）===")
count = 0
for idx, text, style in paragraphs:
    compact = text.replace(' ', '').replace('\u3000', '')
    if 'Heading #5' in style and len(compact) <= 10 and compact:
        print(f"  [{idx}] 文本='{text}' compact='{compact}'")
        count += 1
        if count >= 50:
            break
