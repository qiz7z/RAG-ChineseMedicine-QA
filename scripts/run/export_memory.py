# -*- coding: utf-8 -*-
"""
导出会话记忆为 JSON 文件（只读备份）
====================================

把 metadata.db 里的 sessions / dialogue_turns 导成每个会话一个 JSON 文件，
便于肉眼检查摘要质量、做迁移或离线分析。

数据库仍是唯一真相，导出物可随时删掉重导。

用法:
    python scripts/run/export_memory.py                  # 导出所有活跃会话
    python scripts/run/export_memory.py --all            # 含 expired / deleted
    python scripts/run/export_memory.py --session sess_xxx
    python scripts/run/export_memory.py --out /tmp/mem   # 自定义输出目录
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from config import SQLITE_DB_PATH  # noqa: E402


def _load_session(conn, row) -> dict:
    sid = row['session_id']
    turns = conn.execute(
        'SELECT role, content, resolved_query, created_at '
        'FROM dialogue_turns WHERE session_id = ? ORDER BY id',
        (sid,),
    ).fetchall()

    summary = None
    if row['summary_json']:
        try:
            summary = json.loads(row['summary_json'])
        except (ValueError, TypeError):
            summary = {'_parse_error': row['summary_json'][:200]}

    return {
        'session_id': sid,
        'user_id': row['user_id'],
        'status': row['status'],
        'created_at': row['created_at'],
        'last_active': row['last_active'],
        'summary': summary,
        'turn_count': len(turns) // 2,
        'turns': [dict(t) for t in turns],
        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def export(out_dir: Path, session_id: str = None, include_all: bool = False) -> int:
    db_path = Path(SQLITE_DB_PATH)
    if not db_path.exists():
        print(f'数据库不存在: {db_path}')
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    sql = 'SELECT * FROM sessions'
    params = []
    where = []
    if session_id:
        where.append('session_id = ?')
        params.append(session_id)
    if not include_all:
        where.append("status = 'active'")
    if where:
        sql += ' WHERE ' + ' AND '.join(where)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print('没有可导出的会话')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        data = _load_session(conn, row)
        path = out_dir / f"{data['session_id']}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        has_summary = '有摘要' if data['summary'] else '无摘要'
        print(f"  {data['session_id']}  {data['turn_count']} 轮  {has_summary}  ->  {path.name}")
        written += 1

    conn.close()
    print(f'\n已导出 {written} 个会话到 {out_dir}')
    return written


def main():
    ap = argparse.ArgumentParser(description='导出会话记忆为 JSON')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'memory'), help='输出目录')
    ap.add_argument('--session', default=None, help='只导出指定 session_id')
    ap.add_argument('--all', action='store_true', help='包含 expired / deleted 状态')
    args = ap.parse_args()

    export(Path(args.out), session_id=args.session, include_all=args.all)


if __name__ == '__main__':
    main()
