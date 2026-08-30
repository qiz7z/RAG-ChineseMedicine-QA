# -*- coding: utf-8 -*-
"""
LangChain 标准版索引构建
=========================
构建两份索引（与主项目索引完全隔离）：
  1. FAISS 向量索引（langchain_community FAISS，MAX_INNER_PRODUCT + 归一化向量）
  2. BM25Retriever（langchain_community，jieba 分词，pickle 持久化）
  3. drug_names.json（药品名清单，供查询理解使用）

FAISS 已存在时默认复用（跳过编码），--rebuild 强制重建。

用法:
  python langchain_app/build_index.py [--rebuild] [--limit N] [--test]
"""
import sys
import json
import pickle
import time
import argparse
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from config import (
    CHUNKS_JSON_PATH,
    LC_INDEX_DIR,
    LC_BM25_PATH,
    LC_DRUG_NAMES_PATH,
    BM25_K,
)


def build(rebuild: bool = False, limit: int = None, run_test: bool = False):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_community.retrievers import BM25Retriever
    from embeddings import build_embeddings
    from retrievers import load_documents, tokenize

    print("=" * 60)
    print("  LangChain 标准版索引构建")
    print(f"  FAISS 目录: {LC_INDEX_DIR}")
    print("=" * 60)

    documents = load_documents()
    if limit:
        documents = documents[:limit]
    print(f"\n加载 Document: {len(documents)} 条")

    # ---------- 1. FAISS 向量索引 ----------
    if LC_INDEX_DIR.exists() and not rebuild:
        print(f"\n[跳过] FAISS 索引已存在: {LC_INDEX_DIR}（--rebuild 可强制重建）")
    else:
        t0 = time.time()
        print("\n构建 FAISS 向量索引（首次需 GPU 编码，约 4-5 分钟）...")
        vs = FAISS.from_documents(
            documents,
            build_embeddings(),
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            ids=[d.metadata["chunk_id"] for d in documents],
        )
        LC_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(LC_INDEX_DIR))
        print(f"FAISS 构建完成: {vs.index.ntotal} 条 ({time.time() - t0:.0f}s)")

    # ---------- 2. BM25Retriever ----------
    t0 = time.time()
    print("\n构建 BM25Retriever（jieba 分词）...")
    bm25 = BM25Retriever.from_documents(documents, preprocess_func=tokenize)
    bm25.k = BM25_K
    with open(LC_BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 构建完成: {len(bm25.docs)} 条 ({time.time() - t0:.0f}s) -> {LC_BM25_PATH}")

    # ---------- 3. 药品名清单 ----------
    drug_names = sorted({d.metadata["drug_name"] for d in documents if d.metadata.get("drug_name")})
    LC_DRUG_NAMES_PATH.write_text(
        json.dumps(drug_names, ensure_ascii=False), encoding="utf-8"
    )
    print(f"药品名清单: {len(drug_names)} 个 -> {LC_DRUG_NAMES_PATH}")

    # ---------- 4. 冒烟测试 ----------
    if run_test:
        print("\n" + "=" * 60)
        print("检索冒烟测试")
        print("=" * 60)
        from retrievers import build_hybrid_retriever

        retriever = build_hybrid_retriever()
        for q in ["人参的性味归经是什么？", "哪些药材有补气功效？", "黄芪的用法用量"]:
            docs = retriever.invoke(q)
            print(f"\n查询: {q} (返回 {len(docs)} 条)")
            for d in docs[:3]:
                m = d.metadata
                print(f"  [{m.get('drug_name')}/{m.get('section')}] {d.page_content[:50]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 LangChain 标准版索引")
    parser.add_argument("--rebuild", action="store_true", help="强制重建 FAISS")
    parser.add_argument("--limit", type=int, default=None, help="只构建前 N 条（调试用）")
    parser.add_argument("--test", action="store_true", help="构建后运行冒烟测试")
    args = parser.parse_args()
    build(rebuild=args.rebuild, limit=args.limit, run_test=args.test)
