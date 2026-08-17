#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式导出 PostgreSQL 行情/财务表到 Parquet（内存安全版）。

用法:
    python scripts/export_pg_to_parquet.py [--out-dir data/pg_parquet] [--tables stock_daily,share_float] [--batch 200000]

设计约束:
    - 3.6G 内存小机器：server-side cursor + 分批 fetch，绝不整表载入。
    - 显式 pyarrow schema（从 information_schema 推断），避免全 NULL 列导致的 schema 漂移。
    - zstd 压缩；每张表一个文件，流式写入。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import pg  # noqa: E402

DSN = pg.PG_DSN if hasattr(pg, "PG_DSN") and pg.PG_DSN else os.getenv("PG_DSN")

# 导出白名单：行情/财务/基础信息；业务表(回测/模拟盘/账本)留在 PG
DEFAULT_TABLES = [
    "stock_daily",
    "share_float",
    "fina_indicator",
    "income",
    "balancesheet",
    "cashflow",
    "dividend",
    "stk_surv",
    "forecast",
    "express",
    "namechange",
    "stock_basic",
    "trade_cal",
    "report_rc",
    "index_weight",
]

PG_TO_ARROW = {
    "smallint": pa.int64(),
    "integer": pa.int64(),
    "bigint": pa.int64(),
    "double precision": pa.float64(),
    "real": pa.float64(),
    "numeric": pa.float64(),
    "character varying": pa.string(),
    "text": pa.string(),
    "character": pa.string(),
    "name": pa.string(),
    "date": pa.date32(),
    "timestamp without time zone": pa.timestamp("us"),
    "timestamp with time zone": pa.timestamp("us", tz="UTC"),
    "boolean": pa.bool_(),
}


def build_schema(conn, table: str) -> pa.Schema:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        cols = cur.fetchall()
    fields = []
    for name, dtype in cols:
        arrow = PG_TO_ARROW.get(dtype)
        if arrow is None:
            raise RuntimeError(f"{table}.{name}: unsupported pg type {dtype}")
        fields.append(pa.field(name, arrow))
    return pa.schema(fields)


def _rows_to_table(rows: list[tuple], schema: pa.Schema) -> pa.Table:
    """列级构造 Arrow Table，避免 from_pylist 的 dict 开销（3.6G 机器内存关键）。"""
    arrays = []
    for values, field in zip(zip(*rows), schema):
        vals = [
            float(v) if isinstance(v, Decimal) else v
            for v in values
        ]
        try:
            arrays.append(pa.array(vals, type=field.type))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            # 类型推断兜底（极少出现，如全 NULL 与后续类型不一致）
            arrays.append(pa.array(vals).cast(field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def export_table(conn, table: str, out_path: Path, batch_size: int) -> tuple[int, float, int]:
    schema = build_schema(conn, table)
    sql = f'SELECT {", ".join(schema.names)} FROM "public"."{table}"'
    t0 = time.time()
    rows = 0
    writer = None
    with conn.cursor(name=f"export_{table}") as cur:
        cur.itersize = batch_size
        cur.execute(sql)
        while True:
            batch = cur.fetchmany(batch_size)
            if not batch:
                break
            rows += len(batch)
            table_arrow = _rows_to_table(batch, schema)
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(out_path, schema, compression="zstd")
            writer.write_table(table_arrow)
            if rows % (batch_size * 5) < batch_size:
                print(f"  {table}: {rows:,} rows, {time.time()-t0:.0f}s", flush=True)
    if writer is not None:
        writer.close()
    elapsed = time.time() - t0
    size = out_path.stat().st_size if out_path.exists() else 0
    return rows, elapsed, size


def export_last_adj(conn, out_path: Path) -> tuple[int, str]:
    """导出每只股票最新复权因子快照（前复权锚点）。

    只取最后一个交易日的 adj_factor（约 5000 行），内存占用极小。
    core.data 读取时用它做 qfq 锚点，避免同一天在不同查询区间显示不同价格。
    """
    with conn.cursor() as cur:
        cur.execute("SELECT max(trade_date) FROM stock_daily")
        row = cur.fetchone()
        max_date = row[0] if row else None
        if max_date is None:
            return 0, ""
        cur.execute(
            "SELECT ts_code, trade_date, adj_factor FROM stock_daily "
            "WHERE trade_date = %s AND adj_factor IS NOT NULL",
            (max_date,),
        )
        rows = cur.fetchall()
    schema = pa.schema([
        pa.field("ts_code", pa.string()),
        pa.field("ref_date", pa.date32()),
        pa.field("last_adj", pa.float64()),
    ])
    table = pa.Table.from_arrays(
        [[r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]],
        schema=schema,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    return len(rows), str(max_date)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "pg_parquet"))
    ap.add_argument("--tables", default=",".join(DEFAULT_TABLES))
    # 3.6G 机器：20 万行 x 158 列(财务宽表)峰值约 2.2G，5 万行约 850M
    ap.add_argument("--batch", type=int, default=50000)
    args = ap.parse_args()

    if not DSN:
        print("PG_DSN 未配置", file=sys.stderr)
        sys.exit(1)

    if args.tables.strip().lower() == "all":
        tables = DEFAULT_TABLES
    else:
        tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir={out_dir}  tables={tables}  batch={args.batch}", flush=True)

    total_t0 = time.time()
    with psycopg.connect(DSN) as conn:
        for table in tables:
            out_path = out_dir / f"{table}.parquet"
            try:
                rows, elapsed, size = export_table(conn, table, out_path, args.batch)
                print(
                    f"[ok] {table}: rows={rows:,} time={elapsed:.1f}s "
                    f"size={size/1024/1024:.1f}MB -> {out_path}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[FAIL] {table}: {exc}", flush=True)
        # stock_daily 导出成功后补一份复权因子快照，供读取层做稳定 qfq 锚点
        if "stock_daily" in tables:
            try:
                n, ref_date = export_last_adj(conn, out_dir / "stock_daily_last_adj.parquet")
                print(f"[ok] stock_daily_last_adj: rows={n} ref_date={ref_date}", flush=True)
            except Exception as exc:
                print(f"[FAIL] stock_daily_last_adj: {exc}", flush=True)
    print(f"total: {time.time()-total_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
