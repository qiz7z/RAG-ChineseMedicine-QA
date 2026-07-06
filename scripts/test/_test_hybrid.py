# -*- coding: utf-8 -*-
"""混合检索测试（索引已构建好，只需加载测试）"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from config import VECTOR_TOP_K, BM25_TOP_K, VECTORSTORE_DIR, BM25_INDEX_PATH, SQLITE_DB_PATH
from indexing.embedder import Embedder
from indexing.vector_index import FaissVectorIndex
from indexing.keyword_index import BM25Index
from indexing.fusion import rrf_fusion
from indexing.metadata_store import MetadataStore

print("=" * 60)
print("混合检索测试（向量 + BM25 + RRF 融合）")
print("=" * 60)

# 加载索引
print("\n加载索引...")
embedder = Embedder()
vector_index = FaissVectorIndex()
bm25_index = BM25Index()
bm25_index.load(BM25_INDEX_PATH)
meta_store = MetadataStore(SQLITE_DB_PATH)
print(f"  Chroma: {vector_index.count()} 条")
print(f"  BM25: {bm25_index.count()} 条")
print(f"  SQLite: {meta_store.count()} 条")

test_queries = [
    "人参的性味归经是什么？",
    "双黄连口服液的含量测定方法",
    "哪些药材含有挥发油成分？",
    "高效液相色谱法的操作步骤",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"查询: '{query}'")
    print("-" * 60)

    # 向量检索
    query_emb = embedder.encode_query(query)
    vector_results = vector_index.query(query_emb, top_k=VECTOR_TOP_K)
    print(f"  向量检索: {len(vector_results)} 条")

    # BM25 检索
    bm25_results = bm25_index.query(query, top_k=BM25_TOP_K)
    print(f"  BM25检索: {len(bm25_results)} 条")

    # RRF 融合
    fused = rrf_fusion([vector_results, bm25_results], top_n=5)
    print(f"  融合后 Top-5:")

    # 用 SQLite 补全元数据
    top_ids = [r['id'] for r in fused]
    meta_map = {m['chunk_id']: m for m in meta_store.get_by_ids(top_ids)}

    for i, r in enumerate(fused, 1):
        meta = meta_map.get(r['id'], {})
        drug = meta.get('drug_name', '?')
        section = meta.get('section', '?')
        score = r.get('rrf_score', 0)
        content_preview = r.get('content', '')[:80].replace('\n', ' ')
        print(f"    {i}. [rrf={score:.6f}] {drug} > {section}")
        print(f"       {content_preview}...")

meta_store.close()
print(f"\n{'='*60}")
print("✅ 混合检索测试完成!")
