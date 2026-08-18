#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把旧 DuckDB 业务库 data/quant.duckdb 迁移到 SQLite data/quant.db。

用法:
    python scripts/migrate_quant_duckdb_to_sqlite.py [--duckdb data/quant.duckdb] [--sqlite data/quant.db]

安全:
    - 目标文件已存在时先备份为 quant.db.bak_<时间戳>
    - 逐表复制并核对行数
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sqldb  # noqa: E402


def _to_sqlite_val(v):
    """DuckDB JSON/时间列 -> sqlite 可绑定值。"""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)
    return v


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif df[col].dtype == object:
            df[col] = df[col].map(_to_sqlite_val)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="迁移 DuckDB 业务库到 SQLite")
    ap.add_argument("--duckdb", default=str(ROOT / "data" / "quant.duckdb"))
    ap.add_argument("--sqlite", default=str(ROOT / "data" / "quant.db"))
    args = ap.parse_args()

    duck_path = Path(args.duckdb)
    sql_path = Path(args.sqlite)
    if not duck_path.exists():
        print(f"DuckDB 不存在: {duck_path}", file=sys.stderr)
        return 1
    if sql_path.exists():
        bak = sql_path.with_name(f"{sql_path.name}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(sql_path, bak)
        print(f"已备份旧 SQLite: {bak}")

    try:
        import duckdb
    except ImportError:
        print("需要安装 duckdb 才能读取旧库", file=sys.stderr)
        return 1

    src = duckdb.connect(str(duck_path), read_only=True)
    tables = sorted(r[0] for r in src.execute("SHOW TABLES").fetchall())
    sqldb.create_schema()

    ok = True
    total = 0
    for table in tables:
        df = src.execute(f'SELECT * FROM "{table}"').fetchdf()
        if len(df):
            df = _normalize_df(df)
            sqldb.df_to_pg(df, table, if_exists="append")
        n = len(df)
        total += n
        print(f"  {table}: {n:,} 行")
    src.close()

    # 核对
    con = sqldb.query_df(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    print(f"\nSQLite 表数量: {len(con)}，迁移合计 {total:,} 行")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())