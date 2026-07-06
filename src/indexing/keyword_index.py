# -*- coding: utf-8 -*-
"""
BM25 关键词索引
================
使用 jieba 分词 + rank-bm25 构建全文关键词索引。

BM25 优势（与向量检索互补）：
  - 精确匹配专业术语（药品名、化学成分名、通则编号）
  - 对数值/编号类查询效果好（如 "通则0512"、"0.10%"）
  - 无需 GPU，纯 CPU 计算

药典领域需要自定义词典：
  - 药品名不应被切分（如"双黄连口服液"不应切成"双/黄连/口服液"）
  - 化学成分名应保持完整（如"人参皂苷Rg1"）
  - 检测方法名应保持完整（如"高效液相色谱法"）
"""
import sys
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BM25_INDEX_PATH


# ============================================================
# jieba 分词器（带自定义词典）
# ============================================================

# 药典领域自定义词汇
CUSTOM_WORDS = [
    # 检测方法
    "高效液相色谱法", "薄层色谱法", "气相色谱法", "紫外分光光度法",
    "高效液相", "液相色谱", "薄层色谱", "气相色谱",
    # 药典通则
    "通则0512", "通则0502", "通则0832", "通则2302", "通则2201",
    # 常见化学成分（不切碎）
    "人参皂苷", "黄芪甲苷", "黄芩苷", "绿原酸", "连翘苷",
    "芍药苷", "丹参酮", "小檗碱", "马钱子碱", "士的宁",
    "挥发油", "总黄酮", "总皂苷",
    # 药典章节标记
    "性味与归经", "功能与主治", "用法与用量", "含量测定",
    "特征图谱", "指纹图谱", "浸出物",
    # 常见药材名（多字词）
    "双黄连", "口服液", "六味地黄丸", "牛黄解毒片",
    # 其他专业术语
    "对照品", "对照药材", "供试品", "测试品",
    "甲醇", "乙腈", "乙酸乙酯", "三氯甲烷", "正丁醇",
]


class JiebaTokenizer:
    """jieba 分词器，内置药典领域自定义词典"""

    _initialized = False

    @classmethod
    def _init_jieba(cls):
        """初始化 jieba（只执行一次）"""
        if cls._initialized:
            return
        import jieba
        for word in CUSTOM_WORDS:
            jieba.add_word(word, freq=10000)
        # 关闭 jieba 日志
        jieba.setLogLevel(20)
        cls._initialized = True

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        分词，返回 token 列表。

        策略：
          1. jieba 分词
          2. 过滤空白和纯标点
          3. 保留英文+数字混合 token（如 Rg1, C27H82O18）
        """
        cls._init_jieba()
        import jieba

        tokens = jieba.cut(text, cut_all=False)
        # 过滤：去空白、去纯标点
        result = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            # 保留有意义的 token（含中文、字母或数字）
            if re.search(r'[\u4e00-\u9fffA-Za-z0-9]', tok):
                result.append(tok)
        return result

    @classmethod
    def add_custom_words(cls, words: List[str]):
        """动态添加自定义词汇"""
        cls._init_jieba()
        import jieba
        for word in words:
            if word and len(word) >= 2:
                jieba.add_word(word, freq=10000)


# ============================================================
# BM25 索引
# ============================================================

class BM25Index:
    """BM25 关键词索引"""

    def __init__(self):
        self.bm25 = None
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

    # ----------------------------------------------------------
    # 构建
    # ----------------------------------------------------------

    def build(self, ids: List[str], texts: List[str], custom_words: List[str] = None):
        """
        构建 BM25 索引。

        Args:
            ids: 文档 ID 列表
            texts: 文档文本列表
            custom_words: 额外的自定义词典（如从药品数据中提取的药品名）
        """
        from rank_bm25 import BM25Okapi

        # 添加自定义词汇
        if custom_words:
            JiebaTokenizer.add_custom_words(custom_words)

        print(f"正在分词（{len(texts)} 条文档）...")
        self.doc_ids = ids
        self.documents = texts
        self.tokenized_corpus = [JiebaTokenizer.tokenize(text) for text in texts]

        print("正在构建 BM25 索引...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 索引构建完成: {len(self.doc_ids)} 条文档")

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def query(self, query_text: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        BM25 关键词检索。

        Args:
            query_text: 查询文本
            top_k: 返回结果数

        Returns:
            结果列表，每个元素包含 id, content, score
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 build() 或 load()")

        tokenized_query = JiebaTokenizer.tokenize(query_text)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # 取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:  # 过滤零分结果
                continue
            results.append({
                "id": self.doc_ids[idx],
                "content": self.documents[idx],
                "score": float(scores[idx]),
                "source": "bm25",
            })

        return results

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------

    def save(self, path: str = None):
        """保存索引到磁盘"""
        path = path or BM25_INDEX_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "doc_ids": self.doc_ids,
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"BM25 索引已保存到 {path}")

    def load(self, path: str = None):
        """从磁盘加载索引"""
        from rank_bm25 import BM25Okapi

        path = path or BM25_INDEX_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.doc_ids = data["doc_ids"]
        self.documents = data["documents"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 索引已加载: {len(self.doc_ids)} 条文档")

    # ----------------------------------------------------------
    # 状态查询
    # ----------------------------------------------------------

    def count(self) -> int:
        """返回索引中的文档数"""
        return len(self.doc_ids)


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("BM25 关键词索引测试")
    print("=" * 60)

    # 测试分词
    test_texts = [
        "人参皂苷Rg1的含量测定方法",
        "照高效液相色谱法（通则0512）测定",
        "双黄连口服液中金银花的含量测定",
    ]
    print("\n分词测试:")
    for text in test_texts:
        tokens = JiebaTokenizer.tokenize(text)
        print(f"  {text}")
        print(f"    → {' / '.join(tokens)}")

    # 测试 BM25
    print("\nBM25 检索测试:")
    index = BM25Index()
    ids = ["d1", "d2", "d3", "d4"]
    texts = [
        "人参的性味与归经：甘、微苦，微温。归脾、肺、心经。",
        "黄芪的功能与主治：补气固表，利尿托毒，排脓，敛疮生肌。",
        "人参的含量测定：照高效液相色谱法（通则0512）测定人参皂苷Rg1的含量。",
        "黄连的功能与主治：清热燥湿，泻火解毒。",
    ]
    index.build(ids, texts)

    query = "人参皂苷Rg1含量测定"
    results = index.query(query, top_k=3)
    print(f"\n查询: '{query}'")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}")
