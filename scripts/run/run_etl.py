# -*- coding: utf-8 -*-
"""
一键运行 ETL 流水线
====================
用法: python scripts/run/run_etl.py

流程: .docx → 解析(parser) → 清洗(cleaner) → 切片(chunker) → chunks.json
"""
import sys
import time
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from config import RAW_DOCX_PATH, DRUGS_JSON_PATH, CHUNKS_JSON_PATH
from etl.parser import PharmacopoeiaParser
from etl.cleaner import clean_entries
from etl.chunker import PharmacopoeiaChunker


def main():
    print("=" * 60)
    print("  药典 RAG 系统 - ETL 流水线")
    print("=" * 60)

    # 检查输入文件
    if not Path(RAW_DOCX_PATH).exists():
        print(f"错误: 找不到输入文件 {RAW_DOCX_PATH}")
        sys.exit(1)

    total_start = time.time()

    # Step 1: 解析
    print("\n[1/3] 文档解析...")
    t0 = time.time()
    parser = PharmacopoeiaParser(RAW_DOCX_PATH)
    raw_entries = parser.parse()
    # 转为dict格式
    from dataclasses import asdict
    entries = [asdict(e) for e in raw_entries]
    print(f"  耗时: {time.time() - t0:.1f}s")
    print(f"  提取 {len(entries)} 个药品条目")

    # Step 2: 清洗
    print("\n[2/3] 数据清洗...")
    t0 = time.time()
    entries = clean_entries(entries)
    # 保存清洗后数据
    import json
    with open(DRUGS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # Step 3: 切片
    print("\n[3/3] 智能切片...")
    t0 = time.time()
    chunker = PharmacopoeiaChunker()
    chunks = chunker.chunk_entries(entries)
    with open(CHUNKS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # 汇总
    print(f"\n{'='*60}")
    print(f"  流水线完成!")
    print(f"  总耗时: {time.time() - total_start:.1f}s")
    print(f"  药品条目: {len(entries)}")
    print(f"  切片数量: {len(chunks)}")
    print(f"  平均每药品: {len(chunks)/len(entries):.1f} 个chunk")
    print(f"  输出文件:")
    print(f"    结构化数据: {DRUGS_JSON_PATH}")
    print(f"    切片数据:   {CHUNKS_JSON_PATH}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
