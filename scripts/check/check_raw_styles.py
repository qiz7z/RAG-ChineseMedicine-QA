# -*- coding: utf-8 -*-
"""检查核心药材在原始 Word 文档中的样式和上下文"""
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

# 核心药材列表
core_herbs = ['黄芪', '甘草', '川芎', '茯苓', '麻黄', '半夏', '柴胡', '地黄', '山药', '泽泻', '薏苡仁', '附子', '黄芩']

# 收集所有段落
paragraphs = []
for i, block in enumerate(iter_block_items(doc)):
    if isinstance(block, Paragraph):
        text = block.text.strip()
        style = block.style.name if block.style else 'None'
        paragraphs.append((i, text, style, block))

print(f"文档总段落数: {len(paragraphs)}")
print()

# 对每个核心药材，在段落中搜索
for herb in core_herbs:
    print(f"=== 搜索 '{herb}' ===")
    found = False
    for idx, text, style, block in paragraphs:
        # 精确匹配段落文本
        compact = text.replace(' ', '').replace('\u3000', '')
        if compact == herb:
            # 找到了！打印上下文
            print(f"  [段落{idx}] 样式: '{style}' 文本: '{text}'")
            # 打印前后2段
            for j in range(max(0, idx-2), min(len(paragraphs), idx+3)):
                pidx, ptext, pstyle, _ = paragraphs[j]
                marker = " <<<" if pidx == idx else ""
                print(f"    [{pidx}] '{pstyle}' | {ptext[:60]}{marker}")
            found = True
            print()
    
    if not found:
        # 尝试模糊搜索 - 段落以该药材名开头
        for idx, text, style, block in paragraphs:
            compact = text.replace(' ', '').replace('\u3000', '')
            if compact.startswith(herb) and len(compact) <= len(herb) + 5:
                print(f"  [段落{idx}] 样式: '{style}' 文本: '{text}' (以{herb}开头)")
                for j in range(max(0, idx-2), min(len(paragraphs), idx+3)):
                    pidx, ptext, pstyle, _ = paragraphs[j]
                    marker = " <<<" if pidx == idx else ""
                    print(f"    [{pidx}] '{pstyle}' | {ptext[:60]}{marker}")
                found = True
                print()
    
    if not found:
        # 在所有段落文本中搜索包含
        for idx, text, style, block in paragraphs:
            if herb in text and len(text) < 30:
                print(f"  [段落{idx}] 样式: '{style}' 文本: '{text}' (包含{herb})")
        
        # 也搜索是否有 OCR 变体
        print(f"  未找到精确/开头匹配，尝试变体搜索...")
        # 黄芪 -> 黄茂?
        # 川芎 -> 川弯/川夸/川菖?
        # 茯苓 -> 茯苓?
        # 麻黄 -> 麻黄?
        # 柴胡 -> 柴胡?
        # 地黄 -> 地黄?
        # 山药 -> 山药?
        # 泽泻 -> 泽泻?
        # 薏苡仁 -> 薏苡仁?
        # 附子 -> 附子?
        # 黄芩 -> 黄芩?
        
        # 搜索包含部分
        for idx, text, style, block in paragraphs:
            if herb[-1] in text and len(text) < 20:
                compact = text.replace(' ', '').replace('\u3000', '')
                if herb[-2:] in compact:
                    print(f"  可能匹配 [段落{idx}] 样式: '{style}' 文本: '{text}'")
        
        print()
