# -*- coding: utf-8 -*-
"""
数据清洗器
================
对解析后的药品数据进行清洗：
1. OCR 错别字纠错（【検查】→【检查】等）
2. 全角/半角统一
3. 多余空白合并
4. 异常字符清理
"""
import re
import json
from pathlib import Path


# ============================================================
# OCR 错别字映射表
# ============================================================

# 章节标记级错误（从数据分析中发现）
SECTION_OCR_FIXES = {
    '【検查】': '【检查】',
    '【含量測定】': '【含量测定】',
    '〔礎藏〕': '〔贮藏〕',
    '〔疋藏〕': '〔贮藏〕',
    '〔规格（l）〕': '〔规格（1）〕',
    '【性味与帰経】': '【性味与归经】',
    '【性味與歸經】': '【性味与归经】',
}

# 章节名称纠错（解析后section_name已去除【】括号，需单独处理）
SECTION_NAME_NORMALIZE = {
    # 检查
    '検查': '检查', '检査': '检查',
    # 含量测定
    '含量測定': '含量测定', '含量则定': '含量测定',
    # 性味与归经
    '性昧与归经': '性味与归经', '性味與歸經': '性味与归经',
    # 贮藏（各种OCR变体）
    'St藏': '贮藏', '/藏': '贮藏', '巫藏': '贮藏', '旋藏': '贮藏',
    '疋藏': '贮藏', '礎藏': '贮藏', '触藏': '贮藏', '正藏': '贮藏',
    # 浸出物
    '浸岀物': '浸出物',
    # 功能主治
    '功能主治': '功能与主治',
    # 用法
    '用法': '用法与用量',
    # 注意
    '注意事项': '注意',
}

# 药品名 OCR 纠正映射
# 文档中部分字符因 OCR 识别错误导致药品名不正确
# 芪、芩、芎 三字在文档中完全不存在，分别被误识为茂/苓/苔/弯/夸/菖/萼
DRUG_NAME_OCR_FIXES = {
    '黄茂': '黄芪',       # 芪→茂
    '黄苓': '黄芩',       # 芩→苓
    '黄苔': '黄芩',       # 芩→苔
    '川弯': '川芎',       # 芎→弯
    '川夸': '川芎',       # 芎→夸
    '川菖': '川芎',       # 芎→菖
    '川萼': '川芎',       # 芎→萼
    '主茯苓': '土茯苓',   # 土→主
}

# 字符级错误（常见 OCR 误识别）
CHAR_OCR_FIXES = {
    '昔': '苷',     # 人参皂昔 → 人参皂苷（非常常见的OCR错误）
    '甘': '苷',     # 部分上下文中 甘 应为 苷（但需注意"甘"本身也是药味描述，只在特定上下文替换）
    '蔥': '蒽',     # 蒽醌类
    '月青': '肼',   # 
    '检査': '检查',
    '検查': '检查',
    '測定': '测定',
}

# 仅在特定上下文中替换的规则
# (正则模式, 替换文本)
CONTEXT_FIXES = [
    # "人参皂昔" → "人参皂苷"
    (r'(皂)昔', r'\1苷'),
    (r'(黄芩)昔', r'\1苷'),
    (r'(连翘)昔', r'\1苷'),
    (r'(芍药)昔', r'\1苷'),
    (r'(\w+苷)\s*昔', r'\1'),  # 防止重复
    # "照" 后面的常见错误
    (r'照髙效液相色谱法', '照高效液相色谱法'),
    (r'照高效液相色谱法（通则。512）', '照高效液相色谱法（通则0512）'),
    # 通则编号修正
    (r'通则。832', '通则0832'),
    (r'通则。502', '通则0502'),
    (r'通则。512', '通则0512'),
]

# 全角数字 → 半角
FULLWIDTH_DIGIT_MAP = str.maketrans(
    '０１２３４５６７８９',
    '0123456789'
)

# 全角字母 → 半角
FULLWIDTH_ALPHA_MAP = str.maketrans(
    'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
    'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
)


def clean_text(text: str) -> str:
    """
    清洗单段文本。
    
    清洗步骤:
    1. OCR 章节标记纠错
    2. 上下文敏感的字符纠错
    3. 全角→半角转换（数字和字母）
    4. 多余空白合并
    5. 异常字符清理
    """
    if not text:
        return text

    # 1. OCR 章节标记纠错
    for wrong, correct in SECTION_OCR_FIXES.items():
        text = text.replace(wrong, correct)

    # 2. 上下文敏感的字符纠错
    for pattern, replacement in CONTEXT_FIXES:
        text = re.sub(pattern, replacement, text)

    # 3. 全角→半角
    text = text.translate(FULLWIDTH_DIGIT_MAP)
    text = text.translate(FULLWIDTH_ALPHA_MAP)

    # 4. 多余空白合并
    # 合并连续空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除行首行尾空格
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # 合并连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. 异常字符清理
    # 去除一些 OCR 残留的特殊符号
    text = text.replace('■', '。')   # OCR 残留的句末标记
    text = text.replace('■', '。')
    # 修复 "0。832" → "0832" （通则编号中的多余句号）
    text = re.sub(r'通则0?\.?(\d{4})', r'通则\1', text)

    return text.strip()


def clean_entry(entry: dict) -> dict:
    """
    清洗一个药品条目（dict格式，来自parser输出的JSON）。
    
    Args:
        entry: 单个药品条目的字典
        
    Returns:
        清洗后的条目字典
    """
    # 清洗药品名 - 关键：去除空格以匹配查询中的"人参" vs "人 参"
    drug_name = entry.get('drug_name', '')
    # 去除普通空格和全角空格，但保留"-饮片"分隔符
    drug_name = drug_name.replace(' ', '').replace('\u3000', '')
    # 应用 OCR 药品名纠正（黄茂→黄芪等）
    for ocr_wrong, correct in DRUG_NAME_OCR_FIXES.items():
        if ocr_wrong in drug_name:
            drug_name = drug_name.replace(ocr_wrong, correct)
    entry['drug_name'] = clean_text(drug_name)
    
    # 清洗 parent_drug（子剂型的父级药品名）
    parent_drug = entry.get('parent_drug', '')
    if parent_drug:
        parent_drug = parent_drug.replace(' ', '').replace('\u3000', '')
        # 同样应用 OCR 纠正
        for ocr_wrong, correct in DRUG_NAME_OCR_FIXES.items():
            if ocr_wrong in parent_drug:
                parent_drug = parent_drug.replace(ocr_wrong, correct)
    entry['parent_drug'] = clean_text(parent_drug)
    
    entry['pinyin_name'] = clean_text(entry.get('pinyin_name', ''))
    entry['latin_name'] = clean_text(entry.get('latin_name', ''))
    entry['intro_text'] = clean_text(entry.get('intro_text', ''))

    # 清洗各章节内容
    cleaned_sections = []
    for section in entry.get('sections', []):
        # 先对章节名做规范化纠错
        raw_section_name = section.get('section_name', '')
        normalized_name = SECTION_NAME_NORMALIZE.get(raw_section_name, raw_section_name)
        cleaned_section = {
            'section_name': clean_text(normalized_name),
            'content': clean_text(section.get('content', '')),
            'table_markdown': clean_text(section.get('table_markdown', '')) if section.get('table_markdown') else None,
            'raw_paragraphs': [clean_text(p) for p in section.get('raw_paragraphs', [])],
        }
        # 跳过内容为空且无表格的章节
        if not cleaned_section['content'] and not cleaned_section['table_markdown']:
            continue
        cleaned_sections.append(cleaned_section)

    entry['sections'] = cleaned_sections
    return entry


def clean_entries(entries: list) -> list:
    """批量清洗药品条目"""
    cleaned = []
    skipped = 0
    for entry in entries:
        entry = clean_entry(entry)
        # 跳过完全没有内容的条目
        if not entry['sections'] and not entry['intro_text']:
            skipped += 1
            continue
        cleaned.append(entry)

    print(f"清洗完成: {len(entries)} → {len(cleaned)} 条目 (跳过空条目 {skipped} 个)")
    return cleaned


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DRUGS_JSON_PATH

    print("=" * 60)
    print("数据清洗器")
    print("=" * 60)

    # 读取解析后的数据
    with open(DRUGS_JSON_PATH, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    print(f"读取 {len(entries)} 个药品条目")

    # 清洗
    cleaned_entries = clean_entries(entries)

    # 保存（覆盖原文件）
    with open(DRUGS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned_entries, f, ensure_ascii=False, indent=2)

    print(f"已保存清洗后数据到 {DRUGS_JSON_PATH}")

    # 统计清洗效果
    print(f"\n清洗效果统计:")
    
    # 检查"皂昔"→"皂苷"替换
    saponin_fixes = 0
    for e in cleaned_entries:
        for s in e.get('sections', []):
            content = s.get('content', '')
            if '皂苷' in content:
                # 检查是否还有残留的"皂昔"
                pass
        # 这里只是展示，实际替换已在clean_text中完成
    
    # 检查章节标记纠错
    section_names = set()
    for e in cleaned_entries:
        for s in e.get('sections', []):
            section_names.add(s['section_name'])
    print(f"  章节类型: {sorted(section_names)}")
    
    # 检查是否还有已知的OCR错误
    remaining_errors = 0
    for e in cleaned_entries:
        all_text = json.dumps(e, ensure_ascii=False)
        for wrong in ['検查', '測定', '礎藏', '疋藏', '皂昔']:
            if wrong in all_text:
                remaining_errors += all_text.count(wrong)
    print(f"  残留已知OCR错误: {remaining_errors} 处")

    # 样例
    print(f"\n{'='*60}")
    print("清洗后样例:")
    print(f"{'='*60}")
    for e in cleaned_entries:
        if '人参' in e['drug_name'] and not e.get('is_yinpian'):
            print(f"\n--- {e['drug_name']} ---")
            for s in e.get('sections', [])[:5]:
                print(f"  【{s['section_name']}】 {s['content'][:100]}...")
            break
