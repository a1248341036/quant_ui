#!/usr/bin/env python3
"""Build one canonical yearly daily-bar file from quant_dataset snapshots.

The source dataset keeps a full historical snapshot and separate incremental
packages. This script selects one calendar year, overlays increments by
``(ts_code, trade_date)``, and writes the existing yearly layout:

    <dataset-root>/<year>/<year>/day/stock_daily.parquet

It never reads or copies intraday files. Run without ``--apply`` first to
inspect the exact coverage and conflicts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb


DEFAULT_ROOT = Path(r"D:\Quant\quant_dataset")
DEFAULT_FULL = DEFAULT_ROOT / "行情数据更新至2026.8.14" / "stock_daily.parquet"
DEFAULT_INCREMENT = DEFAULT_ROOT / "行情数据（增量数据8.10-8.14）" / "stock_daily.parquet"


def _quoted(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description="整理 quant_dataset 某年份日频数据")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL, help="全历史日频快照")
    parser.add_argument("--increment", type=Path, action="append", default=[DEFAULT_INCREMENT])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="原子写入规范年度目录")
    args = parser.parse_args()

    full = args.full.resolve()
    increments = [path.resolve() for path in args.increment if path.exists()]
    output = args.output or args.root / str(args.year) / str(args.year) / "day" / "stock_daily.parquet"
    output = output.resolve()
    if not full.is_file():
        raise SystemExit(f"全历史日频快照不存在: {full}")

    start = f"{args.year}-01-01"
    end = f"{args.year + 1}-01-01"
    con = duckdb.connect()
    try:
        full_source = f"SELECT *, 0 AS _priority FROM read_parquet({_quoted(full)})"
        sources = [full_source]
        for priority, path in enumerate(increments, start=1):
            sources.append(f"SELECT *, {priority} AS _priority FROM read_parquet({_quoted(path)})")
        # Increment packages were exported with a different column order.
        # Name-based alignment prevents a positional union from silently
        # placing values such as is_st into an unrelated field.
        union_sql = " UNION ALL BY NAME ".join(sources)
        base_sql = (
            "SELECT * EXCLUDE (_priority), "
            "row_number() OVER (PARTITION BY ts_code, trade_date ORDER BY _priority DESC) AS _row_rank "
            f"FROM ({union_sql}) "
            f"WHERE trade_date >= DATE '{start}' AND trade_date < DATE '{end}'"
        )
        stats = con.execute(
            "SELECT count(*) AS source_rows, count(DISTINCT ts_code || CAST(trade_date AS VARCHAR)) AS unique_keys, "
            "min(trade_date) AS first_date, max(trade_date) AS last_date, count(DISTINCT ts_code) AS symbols "
            f"FROM ({union_sql}) WHERE trade_date >= DATE '{start}' AND trade_date < DATE '{end}'"
        ).fetchone()
        result_stats = con.execute(
            f"SELECT count(*) AS rows, min(trade_date) AS first_date, max(trade_date) AS last_date, "
            f"count(DISTINCT ts_code) AS symbols FROM ({base_sql}) WHERE _row_rank = 1"
        ).fetchone()
        conflicts = con.execute(
            f"SELECT count(*) FROM ({base_sql}) WHERE _row_rank > 1"
        ).fetchone()[0]
        print(f"year={args.year}")
        print(f"full={full}")
        print(f"increments={len(increments)}")
        print(f"input rows={stats[0]:,} unique_keys={stats[1]:,} duplicate_rows={stats[0] - stats[1]:,}")
        print(f"input coverage={stats[2]} .. {stats[3]} symbols={stats[4]:,}")
        print(f"result rows={result_stats[0]:,} coverage={result_stats[1]} .. {result_stats[2]} symbols={result_stats[3]:,}")
        print(f"output={output}")
        if not args.apply:
            print("dry-run only; pass --apply to write")
            return 0

        if not result_stats[0]:
            raise SystemExit(f"{args.year} 没有可写入的日频数据")
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        con.execute(
            f"COPY (SELECT * EXCLUDE (_row_rank) FROM ({base_sql}) WHERE _row_rank = 1 "
            "ORDER BY trade_date, ts_code) "
            f"TO {_quoted(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        os.replace(tmp, output)
        print(f"written={output} bytes={output.stat().st_size:,}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
