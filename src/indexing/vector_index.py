# -*- coding: utf-8 -*-
"""
FAISS 向量索引管理
===================
将 Chunk 向量存入 FAISS 索引，支持：
  - 批量写入向量 + 文档 + 元数据
  - 语义检索（余弦相似度 / 内积）
  - 元数据过滤（内存过滤）

为什么从 ChromaDB 切换到 FAISS？
  - ChromaDB 0.5.x 在 Windows 上 HNSW 索引无法持久化（已知 bug）
  - ChromaDB 1.x 的 Rust 后端在 Windows 上同样有 HNSW 加载问题
  - FAISS 是 Facebook 出品的成熟向量检索库，Windows 兼容性完美
  - FAISS + pickle 元数据存储方案更简单可靠

存储方案：
  - 向量数据 → FAISS 索引文件（.index）
  - 文档内容 + 元数据 → pickle 文件（.pkl）
  - 两者的行号一一对应
"""
import sys
import os
import pickle
import gc
import platform
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR, CHROMA_COLLECTION_NAME, CHROMA_DISTANCE_METRIC, EMBEDDING_DIM


def _get_short_path(path: str) -> str:
    """
    获取 Windows 短路径名（8.3 格式），解决 FAISS C++ 库无法处理中文路径的问题。

    如果文件不存在，先获取父目录的短路径，再拼接文件名。
    在非 Windows 系统上直接返回原路径。
    """
    if platform.system() != 'Windows':
        return path

    import ctypes
    buf = ctypes.create_unicode_buffer(260)
    GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW

    # 尝试直接获取短路径（文件已存在时）
    result = GetShortPathName(path, buf, 260)
    if result and result < 260 and buf.value != path:
        return buf.value

    # 文件不存在时，获取父目录的短路径
    parent = os.path.dirname(path)
    filename = os.path.basename(path)
    result = GetShortPathName(parent, buf, 260)
    if result and result < 260:
        return os.path.join(buf.value, filename)

    return path

# FAISS 索引文件路径
FAISS_INDEX_PATH = os.path.join(CHROMA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(CHROMA_DIR, "faiss_meta.pkl")


class FaissVectorIndex:
    """
    FAISS 向量索引。

    使用 IndexFlatIP（内积索引）+ L2 归一化向量实现余弦相似度检索。
    对于 11k 规模的数据，Flat 索引暴力扫描即可，无需 HNSW 近似索引。
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        """
        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称（兼容旧接口，实际不使用）
        """
        import faiss

        self.persist_dir = persist_dir or CHROMA_DIR
        self.dimension = EMBEDDING_DIM

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "faiss_meta.pkl")

        # 尝试加载已有索引（使用短路径避免中文路径问题）
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            short_index_path = _get_short_path(self.index_path)
            self.index = faiss.read_index(short_index_path)
            with open(self.meta_path, 'rb') as f:
                self.ids = pickle.load(f)
                self.documents = pickle.load(f)
                self.metadatas = pickle.load(f)
            print(f"  FAISS 索引已加载: {self.index.ntotal} 条向量")
        else:
            # 创建新索引（内积索引，配合归一化向量 = 余弦相似度）
            self.index = faiss.IndexFlatIP(self.dimension)
            self.ids: List[str] = []
            self.documents: List[str] = []
            self.metadatas: List[Dict] = []

    # ----------------------------------------------------------
    # 持久化 & 关闭
    # ----------------------------------------------------------

    def close(self):
        """持久化索引到磁盘"""
        self._save()
        gc.collect()

    def _save(self):
        """保存索引和元数据到磁盘"""
        import faiss

        # 确保目录存在
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # 使用 Windows 短路径名，避免 FAISS C++ 库无法处理中文路径的问题
        short_index_path = _get_short_path(self.index_path)
        faiss.write_index(self.index, short_index_path)

        # 元数据用 Python pickle 保存（支持中文路径）
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.ids, f)
            pickle.dump(self.documents, f)
            pickle.dump(self.metadatas, f)

    # ----------------------------------------------------------
    # 写入
    # ----------------------------------------------------------

    def add_documents(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 500,
    ):
        """
        批量添加文档到向量索引。

        Args:
            ids: 文档唯一 ID 列表
            embeddings: 向量矩阵 (N, dim)，必须已归一化
            documents: 文档文本列表
            metadatas: 元数据列表
            batch_size: 每批写入数量（FAISS 支持批量写入，此参数仅用于日志）
        """
        total = len(ids)
        embeddings_array = np.array(embeddings, dtype=np.float32) if not isinstance(embeddings, np.ndarray) else embeddings.astype(np.float32)

        # 确保是连续内存
        if not embeddings_array.flags['C_CONTIGUOUS']:
            embeddings_array = np.ascontiguousarray(embeddings_array)

        # 添加到 FAISS 索引
        self.index.add(embeddings_array)

        # 添加到元数据列表
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

        # 立即持久化
        self._save()

        print(f"  写入向量: {total}/{total}")

    def clear(self):
        """清空索引"""
        import faiss

        self.index = faiss.IndexFlatIP(self.dimension)
        self.ids = []
        self.documents = []
        self.metadatas = []

        # 删除旧文件
        for path in [self.index_path, self.meta_path]:
            if os.path.exists(path):
                os.remove(path)

        print(f"已清空 FAISS 索引")

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 15,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量语义检索。

        Args:
            query_embedding: 查询向量 (dim,)，必须已归一化
            top_k: 返回结果数
            where: 元数据过滤条件（如 {"drug_name": "人参"}）

        Returns:
            结果列表，每个元素包含 id, content, metadata, score
        """
        if self.index.ntotal == 0:
            return []

        query_array = np.array([query_embedding], dtype=np.float32) if isinstance(query_embedding, np.ndarray) else np.array([query_embedding], dtype=np.float32)
        if not query_array.flags['C_CONTIGUOUS']:
            query_array = np.ascontiguousarray(query_array)

        # 如果有过滤条件，先过滤再检索
        if where:
            # 找到满足条件的索引
            valid_indices = []
            for i, meta in enumerate(self.metadatas):
                match = True
                for key, value in where.items():
                    if key == "$in":
                        continue
                    if isinstance(value, dict) and "$in" in value:
                        if meta.get(key) not in value["$in"]:
                            match = False
                            break
                    elif meta.get(key) != value:
                        match = False
                        break
                if match:
                    valid_indices.append(i)

            if not valid_indices:
                return []

            # 在满足条件的子集中检索
            valid_array = np.array(valid_indices, dtype=np.int64)
            sub_vectors = np.array([self.index.reconstruct(i) for i in valid_indices], dtype=np.float32)

            # 使用临时索引检索
            import faiss
            temp_index = faiss.IndexFlatIP(self.dimension)
            temp_index.add(sub_vectors)

            search_k = min(top_k, len(valid_indices))
            scores, local_indices = temp_index.search(query_array, search_k)

            formatted = []
            for i in range(len(local_indices[0])):
                idx = local_indices[0][i]
                if idx < 0:
                    continue
                original_idx = valid_indices[idx]
                formatted.append({
                    "id": self.ids[original_idx],
                    "content": self.documents[original_idx],
                    "metadata": self.metadatas[original_idx],
                    "score": float(scores[0][i]),
                    "source": "vector",
                })
            return formatted

        # 无过滤条件，直接全库检索
        search_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_array, search_k)

        formatted = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0:
                continue
            formatted.append({
                "id": self.ids[idx],
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(scores[0][i]),
                "source": "vector",
            })

        return formatted

    # ----------------------------------------------------------
    # 状态查询
    # ----------------------------------------------------------

    def count(self) -> int:
        """返回索引中的文档数"""
        return self.index.ntotal

    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """按 ID 获取文档"""
        try:
            idx = self.ids.index(chunk_id)
            return {
                "id": self.ids[idx],
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
            }
        except ValueError:
            return None


# ============================================================
# 兼容性别名：保持与旧代码兼容
# ============================================================
ChromaVectorIndex = FaissVectorIndex


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("FAISS 向量索引测试")
    print("=" * 60)

    index = FaissVectorIndex()
    print(f"当前索引文档数: {index.count()}")
