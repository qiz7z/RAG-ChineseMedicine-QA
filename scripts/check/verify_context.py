# -*- coding: utf-8 -*-
"""验证遗漏药品名的上下文 - 确认后跟拼音"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import docx
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph
import re

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
blocks = list(iter_block_items(doc))

# 检查特定段落索引的上下文
targets = [
    (3068, '甘草', 'Body text|4'),
    (9055, '茯苓', 'Body text|4'),
    (12065, '麻黄', 'Body text|4'),
    (4376, '半夏', 'Heading #5|1'),
    (10593, '柴胡', 'Body text|4'),
    (4590, '地黄', 'Body text|4'),
    (958, '山药', 'Heading #2|1'),
    (8619, '泽泻', 'Body text|4'),
    (7185, '附子', 'Body text|4'),
    (11418, '黄茂(黄芪)', 'Body text|4'),
    (11389, '黄苔(黄芩?)', 'Body text|4'),
    (1252, '川木通', 'Body text|6'),
]

for target_idx, name, expected_style in targets:
    print(f"\n=== [{target_idx}] {name} (期望样式: {expected_style}) ===")
    for j in range(max(0, target_idx-1), min(len(blocks), target_idx+5)):
        block = blocks[j]
        if isinstance(block, Paragraph):
            text = block.text.strip()
            style = block.style.name if block.style else 'None'
            is_pinyin = bool(re.match(r'^[A-Za-z\s\-]+$', text)) and bool(re.search(r'[\u4e00-\u9fff]', text)) == False
            marker = " <<<" if j == target_idx else ""
            print(f"  [{j}] 样式='{style}' 拼音={is_pinyin} | '{text[:60]}'{marker}")
        else:
            print(f"  [{j}] [TABLE]")

# 搜索 "薏苡仁" 的所有变体
print("\n\n=== 搜索薏苡仁变体 ===")
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        text = block.text.strip()
        compact = text.replace(' ', '').replace('\u3000', '')
        # 薏苡仁可能被OCR为薏以仁、薏苡人、薏苡仁等
        if ('薏' in compact and len(compact) < 15) or ('苡' in compact and len(compact) < 15):
            style = block.style.name if block.style else 'None'
            print(f"  [{i}] 样式='{style}' | '{text}'")

# 搜索"川芎"的所有变体 - 更广泛
print("\n=== 搜索川芎变体（在所有段落中搜索含'芎'的短文本）===")
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        text = block.text.strip()
        compact = text.replace(' ', '').replace('\u3000', '')
        if '芎' in compact and len(compact) < 15:
            style = block.style.name if block.style else 'None'
            print(f"  [{i}] 样式='{style}' | '{text}'")

# 搜索黄芩的所有变体
print("\n=== 搜索黄芩变体（搜索含'芩'的短文本）===")
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        text = block.text.strip()
        compact = text.replace(' ', '').replace('\u3000', '')
        if '芩' in compact and len(compact) < 15:
            style = block.style.name if block.style else 'None'
            print(f"  [{i}] 样式='{style}' | '{text}'")

# 搜索含'芪'的短文本
print("\n=== 搜索黄芪变体（搜索含'芪'的短文本）===")
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        text = block.text.strip()
        compact = text.replace(' ', '').replace('\u3000', '')
        if '芪' in compact and len(compact) < 15:
            style = block.style.name if block.style else 'None'
            print(f"  [{i}] 样式='{style}' | '{text}'")
