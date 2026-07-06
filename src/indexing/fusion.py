# -*- coding: utf-8 -*-
"""
RRF 多路检索融合
==================
将向量检索和 BM25 检索的结果通过 RRF (Reciprocal Rank Fusion) 算法融合。

RRF 算法原理：
  对每个文档，计算它在所有检索路径中排名的倒数之和：
    score(d) = Σ 1 / (k + rank_i(d))
  其中 k 是平滑参数（默认 60），rank_i(d) 是文档 d 在第 i 路检索中的排名。

RRF 优势：
  - 不依赖各路检索的原始分数（向量相似度 vs BM25 分数量纲不同）
  - 只看排名，天然适合融合不同检索器的结果
  - 简单高效，无需训练
"""
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RRF_K


def rrf_fusion(
    rank_lists: List[List[Dict[str, Any]]],
    k: int = None,
    top_n: int = 30,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion 融合多路检索结果。

    Args:
        rank_lists: 多路检索结果列表，每一路是按相关度排序的结果列表
                    每个结果是一个 dict，必须包含 "id" 字段
        k: RRF 平滑参数（默认 60）
        top_n: 融合后返回的 Top-N 数量

    Returns:
        融合后的结果列表，按 RRF 分数降序排列
    """
    k = k or RRF_K
    scores = defaultdict(float)
    doc_info = {}  # 保留文档信息（content, metadata 等）

    for rank_list in rank_lists:
        for rank, doc in enumerate(rank_list, 1):
            doc_id = doc["id"]
            scores[doc_id] += 1.0 / (k + rank)
            # 保留文档信息（后出现的覆盖先出现的，保留更高排名的版本）
            if doc_id not in doc_info:
                doc_info[doc_id] = doc
            else:
                # 合并来源信息
                if "sources" not in doc_info[doc_id]:
                    doc_info[doc_id]["sources"] = [doc_info[doc_id].get("source", "")]
                doc_info[doc_id]["sources"].append(doc.get("source", ""))

    # 按分数排序
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # 组装结果
    results = []
    for doc_id in sorted_ids[:top_n]:
        doc = doc_info[doc_id].copy()
        doc["id"] = doc_id
        doc["rrf_score"] = scores[doc_id]
        results.append(doc)

    return results


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("RRF 融合算法测试")
    print("=" * 60)

    # 模拟两路检索结果
    vector_results = [
        {"id": "chunk_001", "content": "人参性味归经", "score": 0.95, "source": "vector"},
        {"id": "chunk_002", "content": "人参功能主治", "score": 0.88, "source": "vector"},
        {"id": "chunk_003", "content": "人参含量测定", "score": 0.82, "source": "vector"},
        {"id": "chunk_004", "content": "黄芪性味归经", "score": 0.75, "source": "vector"},
        {"id": "chunk_005", "content": "西洋参功能主治", "score": 0.70, "source": "vector"},
    ]

    bm25_results = [
        {"id": "chunk_003", "content": "人参含量测定", "score": 15.2, "source": "bm25"},
        {"id": "chunk_001", "content": "人参性味归经", "score": 12.8, "source": "bm25"},
        {"id": "chunk_006", "content": "人参用法用量", "score": 10.5, "source": "bm25"},
        {"id": "chunk_007", "content": "人参检查", "score": 9.3, "source": "bm25"},
        {"id": "chunk_002", "content": "人参功能主治", "score": 8.1, "source": "bm25"},
    ]

    print("\n向量检索结果:")
    for i, r in enumerate(vector_results, 1):
        print(f"  Rank {i}: {r['id']} (score={r['score']:.3f}) {r['content']}")

    print("\nBM25 检索结果:")
    for i, r in enumerate(bm25_results, 1):
        print(f"  Rank {i}: {r['id']} (score={r['score']:.3f}) {r['content']}")

    # RRF 融合
    fused = rrf_fusion([vector_results, bm25_results], k=60, top_n=10)

    print("\nRRF 融合后结果:")
    for i, r in enumerate(fused, 1):
        print(f"  Rank {i}: {r['id']} (rrf_score={r['rrf_score']:.6f}) {r['content']}")
