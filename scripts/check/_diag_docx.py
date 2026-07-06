# -*- coding: utf-8 -*-
"""诊断 Word 文档结构：查看前 200 个段落的样式和文本"""
import sys, io
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

doc = docx.Document(RAW_DOCX_PATH)
blocks = list(iter_block_items(doc))

print(f"Total blocks: {len(blocks)}")
print()

# 找到所有 Heading #3 段落（主药品条目边界）
print("=== 所有 Heading #3 段落（前 50 个）===")
heading3_count = 0
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        style = block.style.name if block.style else ''
        text = block.text.strip()
        if 'Heading #3' in style and text:
            heading3_count += 1
            if heading3_count <= 50:
                print(f"  block[{i:4d}] style={style:20s} text={text[:60]}")

print(f"\n总共 {heading3_count} 个 Heading #3 段落")

# 看看"人工牛黄"和"人参"附近的段落
print("\n=== '人工牛黄' 附近 ===")
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        text = block.text.strip()
        if '人工牛黄' in text and i < 200:
            # 打印前后5个段落
            for j in range(max(0, i-2), min(len(blocks), i+10)):
                if isinstance(blocks[j], Paragraph):
                    s = blocks[j].style.name if blocks[j].style else ''
                    t = blocks[j].text.strip()[:80]
                    marker = " <<<" if j == i else ""
                    print(f"  [{j:4d}] style={s:20s} text={t}{marker}")
            break

# 搜索"人参"作为 Heading #3
print("\n=== 搜索 '人参' 作为标题的段落 ===")
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        style = block.style.name if block.style else ''
        text = block.text.strip()
        if text == '人参' or text == '人 参':
            print(f"  [{i:4d}] style={style:20s} text={text}")
            # 打印前后段落
            for j in range(max(0, i-2), min(len(blocks), i+8)):
                if isinstance(blocks[j], Paragraph):
                    s = blocks[j].style.name if blocks[j].style else ''
                    t = blocks[j].text.strip()[:80]
                    marker = " <<<" if j == i else ""
                    print(f"    [{j:4d}] style={s:20s} text={t}{marker}")

# 搜索"黄芪"作为标题
print("\n=== 搜索 '黄芪' 作为标题的段落 ===")
for i, block in enumerate(blocks):
    if isinstance(block, Paragraph):
        style = block.style.name if block.style else ''
        text = block.text.strip()
        if '黄芪' in text and len(text) <= 5:
            print(f"  [{i:4d}] style={style:20s} text={text}")
            for j in range(max(0, i-2), min(len(blocks), i+8)):
                if isinstance(blocks[j], Paragraph):
                    s = blocks[j].style.name if blocks[j].style else ''
                    t = blocks[j].text.strip()[:80]
                    marker = " <<<" if j == i else ""
                    print(f"    [{j:4d}] style={s:20s} text={t}{marker}")
