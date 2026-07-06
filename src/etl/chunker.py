# -*- coding: utf-8 -*-
"""
药典智能切片器
================
将结构化的药品数据切分为适合 RAG 检索的 Chunk。

=== 切分策略：结构化语义切分（三级） ===

为什么不简单按字符数切？
  药典数据有天然的语义边界——【性状】【鉴别】【检查】等章节标记。
  按字符数切会导致：
  - "【性状】本品呈圆柱形..." 被从中间截断
  - "【含量测定】黄苓..." 和 "【含量测定】金银花..." 混在一起
  - 用户问"人参的性味归经"时，检索到的chunk可能包含含量测定的内容

切分策略：
  Level 1: 药品边界 → 每个药品/子剂型/饮片为独立单位（parser已完成）
  Level 2: 章节边界 → 按【性状】【鉴别】等章节标记分割（parser已完成）
  Level 3: 智能分组与二次切分：
    a) 短章节合并：性味与归经 + 功能与主治 + 用法与用量 + 注意 + 贮藏
       → 合并为一个"临床应用"chunk
    b) 长章节按成分切分：含量测定中按"黄苓/金银花/连翘"分别描述的
       → 按成分二次切分
    c) 超长章节滑动窗口：仍超过阈值的
       → 按自然句滑动窗口切分，保留重叠
    d) 中等章节保持完整：性状、鉴别等
       → 直接作为一个chunk

每个 Chunk 携带的元数据：
    drug_name: 药品名
    pinyin_name: 拼音名
    category: 分类
    section: 章节名
    is_yinpian: 是否饮片
    parent_drug: 父级药品
    chunk_type: chunk类型（overview/clinical/assay/detailed/table）
"""
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional


# ============================================================
# 配置
# ============================================================

# 最大chunk字符数（不含元数据）
MAX_CHUNK_CHARS = 1500

# 滑动窗口参数（用于超长章节的三级切分）
WINDOW_SIZE = 600
OVERLAP_CHARS = 100

# 短章节阈值：小于此值的章节被视为"短章节"，可合并
SHORT_SECTION_THRESHOLD = 300

# 最小chunk字符数，低于此值的detailed章节合并到clinical chunk
MIN_DETAILED_CHARS = 50

# 临床应用类章节（短章节，合并为一个chunk）
CLINICAL_SECTIONS = {
    '性味与归经', '性味', '归经',
    '功能与主治', '功能主治', '主治',
    '用法与用量', '用法',
    '注意', '注意事项', '禁忌',
    '贮藏',
    '规格',
}

# 概述类章节（来源、制法等，通常单独成chunk）
OVERVIEW_SECTIONS = {
    '来源', '制法', '处方', '炮制', '提取', '制法',
}

# 详述类章节（通常较长，需要检查是否需要二次切分）
DETAILED_SECTIONS = {
    '性状', '鉴别', '检查', '浸出物', '含量测定',
    '特征图谱', '指纹图谱', '酸溶性浸出物', '醇浸出物',
    '正丁醇浸出物', '正丁醇提取物', '乙酸乙酯浸出物',
    '挥发性醯浸出物', '挥发油', '总固体', '相对密度',
    '效价测定', '酸碱度',
}

# 含量测定中按成分切分的模式
# 匹配：成分名 + "照" + 检测方法
# 例如："黄苓照高效液相色谱法（通则0512）测定。"
ASSAY_COMPONENT_PATTERN = re.compile(
    r'^([\u4e00-\u9fa5]{1,6})\s*照\s*(?:高效液相色谱法|薄层色谱法|气相色谱法|紫外)',
    re.MULTILINE
)

# 句子切分正则（按中文句号、分号切分）
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[。；！？\n])')


# ============================================================
# Chunk 数据模型
# ============================================================

@dataclass
class Chunk:
    """一个检索用的文本块"""
    chunk_id: str               # 唯一ID
    content: str                # chunk正文内容
    drug_name: str              # 药品名
    pinyin_name: str = ""       # 拼音名
    latin_name: str = ""        # 拉丁名
    category: str = ""          # 分类（药材/提取物/成方制剂）
    section: str = ""           # 章节名（性状/鉴别/含量测定等）
    chunk_type: str = ""        # chunk类型
    is_yinpian: bool = False    # 是否饮片
    is_sub_formulation: bool = False  # 是否子剂型
    parent_drug: str = ""       # 父级药品名
    table_markdown: Optional[str] = None  # 关联表格
    char_count: int = 0         # 内容字符数


# ============================================================
# 切片器
# ============================================================

class PharmacopoeiaChunker:
    """药典智能切片器"""

    def __init__(self):
        self.chunks = []
        self._chunk_counter = 0

    def chunk_entries(self, entries: list) -> list:
        """
        对所有药品条目进行切片。
        
        Args:
            entries: DrugEntry dict 列表（来自 parser + cleaner 输出）
            
        Returns:
            Chunk dict 列表
        """
        self.chunks = []
        self._chunk_counter = 0

        for entry in entries:
            self._chunk_single_entry(entry)

        print(f"切片完成: {len(entries)} 个药品条目 → {len(self.chunks)} 个 Chunk")
        
        # 统计
        from collections import Counter
        type_counter = Counter(c.chunk_type for c in self.chunks)
        print(f"\nChunk 类型分布:")
        for t, cnt in type_counter.most_common():
            print(f"  {t}: {cnt}")

        char_lens = [c.char_count for c in self.chunks]
        if char_lens:
            import statistics
            print(f"\nChunk 字符数统计:")
            print(f"  最小: {min(char_lens)}")
            print(f"  最大: {max(char_lens)}")
            print(f"  中位数: {statistics.median(char_lens):.0f}")
            print(f"  平均: {statistics.mean(char_lens):.0f}")
            # 分桶
            buckets = {"<100": 0, "100-500": 0, "500-1000": 0, "1000-1500": 0, ">1500": 0}
            for l in char_lens:
                if l < 100: buckets["<100"] += 1
                elif l < 500: buckets["100-500"] += 1
                elif l < 1000: buckets["500-1000"] += 1
                elif l <= 1500: buckets["1000-1500"] += 1
                else: buckets[">1500"] += 1
            print(f"  分布:")
            for k, v in buckets.items():
                print(f"    {k}: {v} ({v/len(char_lens)*100:.1f}%)")

        return [asdict(c) for c in self.chunks]

    def _chunk_single_entry(self, entry: dict):
        """对单个药品条目进行切片"""
        drug_name = entry.get('drug_name', '')
        sections = entry.get('sections', [])
        intro_text = entry.get('intro_text', '')

        if not sections and not intro_text:
            return

        # 如果条目整体很短（< MAX_CHUNK_CHARS），作为一个完整chunk
        total_chars = len(intro_text) + sum(
            len(s.get('content', '')) for s in sections
        )
        if total_chars <= MAX_CHUNK_CHARS and len(sections) <= 8:
            self._make_whole_entry_chunk(entry, total_chars)
            return

        # 1. 生成药品概要chunk（药品名+拼音+拉丁名+分类+概述）
        self._make_summary_chunk(entry)

        # 2. 分组处理章节
        clinical_parts = []  # 收集短章节，稍后合并
        short_detailed_parts = []  # 收集过短的详述章节

        for section in sections:
            sec_name = section.get('section_name', '').strip()
            sec_content = section.get('content', '').strip()
            sec_table = section.get('table_markdown', '')

            if not sec_content and not sec_table:
                continue

            # 2a. 短章节 → 收集到 clinical_parts
            if sec_name in CLINICAL_SECTIONS and len(sec_content) < SHORT_SECTION_THRESHOLD:
                clinical_parts.append(section)
                continue

            # 2b. 过短的详述章节（如炮制只有一句话）→ 也合并到clinical
            if (sec_name in DETAILED_SECTIONS or sec_name in OVERVIEW_SECTIONS):
                full_len = len(sec_content) + (len(sec_table) if sec_table else 0)
                if full_len < MIN_DETAILED_CHARS and sec_name not in ('含量测定', '鉴别'):
                    clinical_parts.append(section)
                    continue
                # 正常详述章节 → 检查是否需要二次切分
                self._chunk_detailed_section(entry, section)
                continue

            # 2c. 其他未分类章节
            if sec_name not in CLINICAL_SECTIONS:
                self._chunk_detailed_section(entry, section)
                continue

            # 2d. 属于临床类但较长的章节
            clinical_parts.append(section)

        # 3. 合并短章节为"临床应用"chunk
        if clinical_parts:
            self._merge_clinical_sections(entry, clinical_parts)

    def _make_summary_chunk(self, entry: dict):
        """生成药品概要chunk，包含药品基本信息"""
        parts = []
        drug_name = entry.get('drug_name', '')
        pinyin = entry.get('pinyin_name', '')
        latin = entry.get('latin_name', '')
        category = entry.get('category_hint', '')
        intro = entry.get('intro_text', '').strip()
        parent = entry.get('parent_drug', '')

        if parent:
            parts.append(f"药品名称：{drug_name}（{parent}的子剂型）")
        else:
            parts.append(f"药品名称：{drug_name}")
        if pinyin:
            parts.append(f"拼音：{pinyin}")
        if latin:
            parts.append(f"拉丁名：{latin}")
        if category:
            parts.append(f"分类：{category}")
        if intro:
            parts.append(f"概述：{intro}")

        content = '\n'.join(parts)
        if len(content) > 10:
            self._add_chunk(
                content=content,
                entry=entry,
                section='药品概要',
                chunk_type='summary',
            )

    def _make_whole_entry_chunk(self, entry: dict, total_chars: int):
        """将整个药品条目作为一个chunk（适用于短条目）"""
        parts = []
        intro = entry.get('intro_text', '').strip()
        if intro:
            parts.append(intro)

        for section in entry.get('sections', []):
            sec_name = section.get('section_name', '')
            sec_content = section.get('content', '').strip()
            sec_table = section.get('table_markdown', '')

            if sec_content:
                parts.append(f"【{sec_name}】{sec_content}")
            if sec_table:
                parts.append(f"【{sec_name}-表格】\n{sec_table}")

        content = '\n'.join(parts)
        self._add_chunk(
            content=content,
            entry=entry,
            section='完整条目',
            chunk_type='whole_entry',
        )

    def _merge_clinical_sections(self, entry: dict, sections: list):
        """
        将多个短章节合并为一个"临床应用"chunk。
        
        例如:
          【性味与归经】甘、微苦，微温。归脾、肺、心经。
          【功能与主治】大补元气，复脉固脱...
          【用法与用量】3〜9g。
          【注意】不宜与藜芦同用。
          【贮藏】置阴凉干燥处，防潮。
        
        → 合并为一个chunk
        """
        parts = []
        for section in sections:
            sec_name = section.get('section_name', '').strip()
            sec_content = section.get('content', '').strip()
            if sec_content:
                parts.append(f"【{sec_name}】{sec_content}")

        if not parts:
            return

        content = '\n'.join(parts)

        # 如果合并后仍然超长，拆分
        if len(content) > MAX_CHUNK_CHARS:
            # 按2-3个章节一组切分
            current_parts = []
            current_len = 0
            for part in parts:
                if current_len + len(part) > MAX_CHUNK_CHARS and current_parts:
                    self._add_chunk(
                        content='\n'.join(current_parts),
                        entry=entry,
                        section='临床应用',
                        chunk_type='clinical',
                    )
                    current_parts = [part]
                    current_len = len(part)
                else:
                    current_parts.append(part)
                    current_len += len(part)
            if current_parts:
                self._add_chunk(
                    content='\n'.join(current_parts),
                    entry=entry,
                    section='临床应用',
                    chunk_type='clinical',
                )
        else:
            self._add_chunk(
                content=content,
                entry=entry,
                section='临床应用',
                chunk_type='clinical',
            )

    def _chunk_detailed_section(self, entry: dict, section: dict):
        """
        处理详述类章节（性状、鉴别、检查、含量测定等）。
        
        策略：
        1. 如果章节内容短（≤ MAX_CHUNK_CHARS），直接作为一个chunk
        2. 如果是【含量测定】且包含多个成分描述，按成分切分
        3. 如果仍然超长，按自然句滑动窗口切分
        """
        sec_name = section.get('section_name', '').strip()
        sec_content = section.get('content', '').strip()
        sec_table = section.get('table_markdown', '')

        # 将表格内容附加到章节内容末尾
        full_content = sec_content
        if sec_table:
            full_content = f"{sec_content}\n\n{sec_table}" if sec_content else sec_table

        if not full_content:
            return

        # 1. 短内容 → 直接成chunk
        if len(full_content) <= MAX_CHUNK_CHARS:
            self._add_chunk(
                content=full_content,
                entry=entry,
                section=sec_name,
                chunk_type='detailed',
                table_markdown=sec_table if sec_table else None,
            )
            return

        # 2. 含量测定 → 尝试按成分切分
        if sec_name == '含量测定':
            sub_chunks = self._split_by_component(full_content)
            if len(sub_chunks) > 1:
                for idx, sub_content in enumerate(sub_chunks):
                    self._add_chunk(
                        content=sub_content,
                        entry=entry,
                        section=f"{sec_name}-成分{idx+1}",
                        chunk_type='assay_component',
                    )
                return

        # 3. 检查类 → 尝试按检查项切分（仅当总长超过阈值时）
        if sec_name == '检查' and len(full_content) > MAX_CHUNK_CHARS:
            sub_chunks = self._split_check_items(full_content)
            if len(sub_chunks) > 1:
                for idx, sub_content in enumerate(sub_chunks):
                    if len(sub_content.strip()) < 20:
                        continue
                    self._add_chunk(
                        content=sub_content,
                        entry=entry,
                        section=f"{sec_name}-项{idx+1}",
                        chunk_type='check_item',
                    )
                return

        # 4. 超长内容 → 滑动窗口切分
        if len(full_content) > MAX_CHUNK_CHARS:
            self._sliding_window_chunk(full_content, entry, sec_name)
        else:
            self._add_chunk(
                content=full_content,
                entry=entry,
                section=sec_name,
                chunk_type='detailed',
                table_markdown=sec_table if sec_table else None,
            )

    def _split_by_component(self, content: str) -> list:
        """
        按成分切分含量测定内容。
        
        例如双黄连口服液的【含量测定】包含：
          黄苓照高效液相色谱法...（完整描述黄芩的测定方法）
          金银花照高效液相色谱法...（完整描述金银花的测定方法）
          连翘照高效液相色谱法...（完整描述连翘的测定方法）
        
        切分为3个独立chunk。
        """
        # 找到所有成分名的位置
        matches = list(ASSAY_COMPONENT_PATTERN.finditer(content))
        if len(matches) < 2:
            return [content]  # 无法切分，返回整体

        # 第一个匹配之前的内容（通常是总述）归到第一个chunk
        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            sub_content = content[start:end].strip()
            if sub_content:
                chunks.append(sub_content)

        # 如果第一个匹配之前有内容（如"照高效液相色谱法（通则0512）测定。"），加到第一个chunk前
        prefix = content[:matches[0].start()].strip()
        if prefix and chunks:
            chunks[0] = f"{prefix}\n{chunks[0]}"

        return chunks

    def _split_check_items(self, content: str) -> list:
        """
        按检查项切分【检查】内容。
        
        例如：
          水分 不得过12.0%（通则0832第二法）。
          总灰分 不得过10.0%（通则2302）。
          重金属及有害元素...
        
        切分为多个独立chunk。
        """
        # 检查项通常以"项目名 + 不得过/应/不得"开头
        # 或者以新行开始
        lines = content.split('\n')
        
        # 尝试按检查项名称切分
        # 检查项名称通常在行首，如"水分"、"总灰分"、"重金属"等
        item_pattern = re.compile(
            r'^(水分|总灰分|酸不溶性灰分|重金属及有害元素|砷盐|农药残留|'
            r'二氧化硫残留|黄曲霉毒素|干燥失重|炽灼残渣|膨胀度|'
            r'pH值|相对密度|折光率|乙醇量|甲醇量|'
            r'可见异物|不溶性微粒|装量差异|重量差异|崩解时限|'
            r'溶散时限|发泡量|硬度|脆碎度|溶出度|释放度|'
            r'含量均匀度|有关物质|残留溶剂|水分|干燥失重|'
            r'山银花|其他)'
        )

        chunks = []
        current_chunk = []

        for line in lines:
            if item_pattern.match(line.strip()) and current_chunk:
                # 新检查项开始，保存前一个chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        # 只有切分出2个以上且每个都不太短的才返回
        if len(chunks) >= 2:
            # 过滤掉太短的
            valid_chunks = [c for c in chunks if len(c.strip()) > 20]
            if len(valid_chunks) >= 2:
                return valid_chunks

        return [content]

    def _sliding_window_chunk(self, content: str, entry: dict, sec_name: str):
        """
        对超长内容使用滑动窗口切分。
        
        按自然句切分，窗口大小 WINDOW_SIZE，重叠 OVERLAP_CHARS。
        """
        # 按句子切分
        sentences = SENTENCE_SPLIT_PATTERN.split(content)
        sentences = [s for s in sentences if s.strip()]

        if len(sentences) <= 1:
            # 无法按句切分，强制按字符数切
            for i in range(0, len(content), MAX_CHUNK_CHARS):
                chunk_content = content[i:i + MAX_CHUNK_CHARS]
                self._add_chunk(
                    content=chunk_content,
                    entry=entry,
                    section=f"{sec_name}-段{(i // MAX_CHUNK_CHARS) + 1}",
                    chunk_type='window',
                )
            return

        # 滑动窗口组句
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) > MAX_CHUNK_CHARS and current_chunk:
                # 保存当前chunk
                self._add_chunk(
                    content=''.join(current_chunk),
                    entry=entry,
                    section=f"{sec_name}-段",
                    chunk_type='window',
                )
                # 保留重叠部分
                overlap_text = ''.join(current_chunk)
                if len(overlap_text) > OVERLAP_CHARS:
                    overlap_text = overlap_text[-OVERLAP_CHARS:]
                current_chunk = [overlap_text, sentence]
                current_len = len(overlap_text) + len(sentence)
            else:
                current_chunk.append(sentence)
                current_len += len(sentence)

        if current_chunk:
            self._add_chunk(
                content=''.join(current_chunk),
                entry=entry,
                section=f"{sec_name}-段",
                chunk_type='window',
            )

    def _add_chunk(self, content: str, entry: dict, section: str, 
                   chunk_type: str, table_markdown: str = None):
        """添加一个chunk"""
        self._chunk_counter += 1
        chunk = Chunk(
            chunk_id=f"chunk_{self._chunk_counter:06d}",
            content=content.strip(),
            drug_name=entry.get('drug_name', ''),
            pinyin_name=entry.get('pinyin_name', ''),
            latin_name=entry.get('latin_name', ''),
            category=entry.get('category_hint', ''),
            section=section,
            chunk_type=chunk_type,
            is_yinpian=entry.get('is_yinpian', False),
            is_sub_formulation=entry.get('is_sub_formulation', False),
            parent_drug=entry.get('parent_drug', ''),
            table_markdown=table_markdown,
            char_count=len(content.strip()),
        )
        self.chunks.append(chunk)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DRUGS_JSON_PATH, CHUNKS_JSON_PATH

    print("=" * 60)
    print("药典智能切片器")
    print("=" * 60)

    # 读取清洗后的数据
    with open(DRUGS_JSON_PATH, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    print(f"读取 {len(entries)} 个药品条目")

    # 切片
    chunker = PharmacopoeiaChunker()
    chunks = chunker.chunk_entries(entries)

    # 保存
    Path(CHUNKS_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {len(chunks)} 个 Chunk 到 {CHUNKS_JSON_PATH}")

    # 样例展示
    print(f"\n{'='*60}")
    print("切片样例展示:")
    print(f"{'='*60}")

    # 展示不同类型的chunk
    type_examples = {}
    for chunk in chunks:
        ct = chunk['chunk_type']
        if ct not in type_examples:
            type_examples[ct] = chunk
            if len(type_examples) >= 6:
                break

    for ct, chunk in type_examples.items():
        print(f"\n--- 类型: {ct} | 药品: {chunk['drug_name']} | 章节: {chunk['section']} ---")
        print(f"  字符数: {chunk['char_count']}")
        print(f"  内容预览: {chunk['content'][:200]}...")
        if chunk.get('table_markdown'):
            print(f"  [含表格]")
