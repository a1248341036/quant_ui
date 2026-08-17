#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DuckDB 查询层演示/自检：python scripts/query_db.py [--sql '...']"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db


DEFAULT_SQL = """
SELECT code, count(*) AS rows, min(date) AS start, max(date) AS end,
       round(avg(close), 2) AS avg_close
FROM panel
GROUP BY code
ORDER BY code
LIMIT 5
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="DuckDB 查询层自检")
    parser.add_argument("--sql", default=DEFAULT_SQL, help="要执行的 SQL（默认统计示例）")
    parser.add_argument("--tables", action="store_true", help="只列出可用视图")
    args = parser.parse_args()

    if args.tables:
        print("\n".join(db.tables()))
        return 0

    df = db.query(args.sql)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
