# -*- coding: utf-8 -*-
"""
LangChain 版向量索引构建
=========================
用 LangChain 标准组件（HuggingFaceEmbeddings + langchain FAISS）从
data/processed/chunks.json 重建向量索引，与手撕版索引（data/vectorstore/chroma/）
完全隔离，互不影响。

编码规则与手撕版 embedder.py 对齐（保证对比实验公平）：
  - 文档编码不加指令前缀，查询编码加 BGE_QUERY_INSTRUCTION
  - 归一化向量 + 内积（MAX_INNER_PRODUCT），等价于余弦相似度

用法:
  python langchain_app/build_index_lc.py [--limit N] [--test]
"""
import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNKS_JSON_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    BGE_QUERY_INSTRUCTION,
)

# LangChain 版索引目录（与手撕版 data/vectorstore/chroma/ 隔离）
LC_INDEX_DIR = str(PROJECT_ROOT / "data" / "vectorstore" / "langchain_faiss")

# chunks.json 中除 content 外写入 Document.metadata 的字段
METADATA_FIELDS = [
    "chunk_id", "drug_name", "pinyin_name", "latin_name", "category",
    "section", "chunk_type", "is_yinpian", "is_sub_formulation",
    "parent_drug", "char_count",
]


def load_documents(limit=None):
    """读取 chunks.json 并转为 LangChain Document 列表"""
    from langchain_core.documents import Document

    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if limit:
        chunks = chunks[:limit]

    documents = []
    for c in chunks:
        content = (c.get("content") or "").strip()
        if not content:
            continue
        metadata = {k: c.get(k) for k in METADATA_FIELDS}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_embeddings():
    """构建与手撕版 embedder.py 编码规则一致的 BGEEmbeddings"""
    from embeddings import build_embeddings as _build
    return _build()


def build(limit=None, run_test=False):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    print("=" * 60)
    print("  LangChain 版向量索引构建")
    print(f"  索引目录: {LC_INDEX_DIR}")
    print("=" * 60)

    t0 = time.time()
    documents = load_documents(limit)
    print(f"\n加载 Document: {len(documents)} 条 ({time.time() - t0:.1f}s)")
    if not documents:
        print("没有可用的 Document，退出")
        return

    t0 = time.time()
    print("\n加载 Embedding 模型...")
    embeddings = build_embeddings()
    print(f"Embedding 模型加载完成 ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print(f"\n开始编码并入库（{len(documents)} 条）...")
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        ids=[d.metadata["chunk_id"] for d in documents],
    )
    print(f"索引构建完成 ({time.time() - t0:.1f}s)，共 {vectorstore.index.ntotal} 条向量")

    Path(LC_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(LC_INDEX_DIR)
    print(f"索引已保存: {LC_INDEX_DIR}")

    if run_test:
        print("\n" + "=" * 60)
        print("检索冒烟测试")
        print("=" * 60)
        db = FAISS.load_local(
            LC_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        for q in ["人参的性味归经是什么？", "黄芪的功能主治", "双黄连口服液的含量测定"]:
            docs = db.similarity_search(q, k=3)
            print(f"\n查询: {q}")
            for d in docs:
                m = d.metadata
                print(f"  [{m.get('drug_name')}/{m.get('section')}] {d.page_content[:60]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 LangChain 版 FAISS 向量索引")
    parser.add_argument("--limit", type=int, default=None, help="只构建前 N 条（调试用）")
    parser.add_argument("--test", action="store_true", help="构建后运行检索冒烟测试")
    args = parser.parse_args()
    build(limit=args.limit, run_test=args.test)
