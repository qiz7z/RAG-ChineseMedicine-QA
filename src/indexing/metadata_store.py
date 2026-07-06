# -*- coding: utf-8 -*-
"""
SQLite 元数据存储
==================
将 Chunk 元数据存入 SQLite，支持结构化过滤检索。

用途：
  1. 精确过滤：按药品名、分类、章节等条件筛选 chunk
  2. ID 查询：根据 chunk_id 获取完整元数据
  3. 统计分析：按维度统计 chunk 分布

为什么需要 SQLite？
  - Chroma 的 where 过滤功能有限（不支持 LIKE、OR 等复杂查询）
  - BM25 只能做全文检索，无法按元数据过滤
  - SQLite 支持完整的 SQL 查询，灵活性最高
"""
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SQLITE_DB_PATH


class MetadataStore:
    """SQLite 元数据存储"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or SQLITE_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # 结果以字典形式访问
        self._init_db()

    def _init_db(self):
        """初始化数据库表和索引"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        TEXT PRIMARY KEY,
                content         TEXT,
                drug_name       TEXT,
                pinyin_name     TEXT,
                latin_name      TEXT,
                category        TEXT,
                section         TEXT,
                chunk_type      TEXT,
                is_yinpian      INTEGER,
                is_sub_formulation INTEGER,
                parent_drug     TEXT,
                table_markdown  TEXT,
                char_count      INTEGER
            )
        ''')

        # 为常用过滤字段创建索引
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_drug_name ON chunks(drug_name)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON chunks(category)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_section ON chunks(section)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_parent_drug ON chunks(parent_drug)')
        self.conn.commit()

    # ----------------------------------------------------------
    # 写入
    # ----------------------------------------------------------

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """批量写入 chunk 元数据"""
        records = []
        for chunk in chunks:
            records.append((
                chunk.get('chunk_id', ''),
                chunk.get('content', ''),
                chunk.get('drug_name', ''),
                chunk.get('pinyin_name', ''),
                chunk.get('latin_name', ''),
                chunk.get('category', ''),
                chunk.get('section', ''),
                chunk.get('chunk_type', ''),
                int(chunk.get('is_yinpian', False)),
                int(chunk.get('is_sub_formulation', False)),
                chunk.get('parent_drug', ''),
                chunk.get('table_markdown'),
                chunk.get('char_count', 0),
            ))

        self.conn.executemany('''
            INSERT OR REPLACE INTO chunks
            (chunk_id, content, drug_name, pinyin_name, latin_name, category,
             section, chunk_type, is_yinpian, is_sub_formulation, parent_drug,
             table_markdown, char_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records)
        self.conn.commit()
        print(f"已写入 {len(records)} 条元数据到 SQLite")

    def clear(self):
        """清空表"""
        self.conn.execute('DELETE FROM chunks')
        self.conn.commit()

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """按 chunk_id 获取元数据"""
        row = self.conn.execute(
            'SELECT * FROM chunks WHERE chunk_id = ?', (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """按多个 chunk_id 批量获取元数据"""
        if not chunk_ids:
            return []
        placeholders = ','.join(['?'] * len(chunk_ids))
        rows = self.conn.execute(
            f'SELECT * FROM chunks WHERE chunk_id IN ({placeholders})', chunk_ids
        ).fetchall()
        return [dict(row) for row in rows]

    def filter(
        self,
        drug_name: str = None,
        category: str = None,
        section: str = None,
        chunk_type: str = None,
        is_yinpian: bool = None,
        parent_drug: str = None,
        limit: int = None,
    ) -> List[Dict]:
        """
        按元数据条件过滤 chunk。

        Args:
            drug_name: 药品名（精确匹配）
            category: 分类
            section: 章节名
            chunk_type: chunk 类型
            is_yinpian: 是否饮片
            parent_drug: 父级药品名
            limit: 返回数量限制

        Returns:
            匹配的 chunk 元数据列表
        """
        conditions = []
        params = []

        if drug_name is not None:
            conditions.append('drug_name = ?')
            params.append(drug_name)
        if category is not None:
            conditions.append('category = ?')
            params.append(category)
        if section is not None:
            conditions.append('section = ?')
            params.append(section)
        if chunk_type is not None:
            conditions.append('chunk_type = ?')
            params.append(chunk_type)
        if is_yinpian is not None:
            conditions.append('is_yinpian = ?')
            params.append(int(is_yinpian))
        if parent_drug is not None:
            conditions.append('parent_drug = ?')
            params.append(parent_drug)

        where_clause = ' AND '.join(conditions) if conditions else '1=1'
        sql = f'SELECT * FROM chunks WHERE {where_clause}'
        if limit:
            sql += f' LIMIT {limit}'

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_all_drug_names(self) -> List[str]:
        """获取所有不重复的药品名"""
        rows = self.conn.execute('SELECT DISTINCT drug_name FROM chunks ORDER BY drug_name').fetchall()
        return [row['drug_name'] for row in rows]

    def count(self) -> int:
        """返回总记录数"""
        return self.conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]

    def stats(self) -> Dict[str, Any]:
        """返回统计信息"""
        total = self.count()
        by_category = {
            row['category']: row['cnt']
            for row in self.conn.execute(
                'SELECT category, COUNT(*) as cnt FROM chunks GROUP BY category'
            ).fetchall()
        }
        by_chunk_type = {
            row['chunk_type']: row['cnt']
            for row in self.conn.execute(
                'SELECT chunk_type, COUNT(*) as cnt FROM chunks GROUP BY chunk_type'
            ).fetchall()
        }
        drug_count = self.conn.execute(
            'SELECT COUNT(DISTINCT drug_name) FROM chunks'
        ).fetchone()[0]
        return {
            'total_chunks': total,
            'total_drugs': drug_count,
            'by_category': by_category,
            'by_chunk_type': by_chunk_type,
        }

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("SQLite 元数据存储测试")
    print("=" * 60)

    store = MetadataStore()
    print(f"总记录数: {store.count()}")

    stats = store.stats()
    print(f"\n统计信息:")
    print(f"  药品数: {stats['total_drugs']}")
    print(f"  分类分布: {stats['by_category']}")
    print(f"  Chunk类型分布: {stats['by_chunk_type']}")

    # 测试过滤
    print(f"\n按药品名过滤（人参）:")
    results = store.filter(drug_name="人参", limit=5)
    for r in results:
        print(f"  {r['chunk_id']} | {r['section']} | {r['chunk_type']} | {r['char_count']}字符")

    store.close()
