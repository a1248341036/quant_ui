#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 PG 原始价 + 复权因子重建股票 panel.parquet（稳定前复权，不再随腾讯 qfq 漂移）。

用法:
    python scripts/rebuild_stock_panel_from_pg.py [--start 2020-01-01] [--batch 200]

数据来源:
    - 行情: data/pg_parquet/stock_daily.parquet（每日 export_pg_to_parquet.py 生成）
    - 前复权锚点: data/pg_parquet/stock_daily_last_adj.parquet（导出时同步生成）
    - 股票池: data/universe.csv（沪深300+中证500+中证1000）

口径:
    与 core.data._finalize_stock_df 一致：qfq = 原始价 * adj_factor / last_adj，
    turn20/am20 为每只股票滚动 20 日均值（min_periods=15）。
    panel 文件保留原 schema（date/open/close/turnover/amount/code/turn20/am20/volume）。

内存:
    按股票分批处理，每批几百只股票的历史，3.6G 机器安全。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data import _finalize_stock_df  # noqa: E402
from core.store import PANEL_FILE, UNIVERSE_FILE  # noqa: E402

PANEL_SCHEMA = pa.schema([
    pa.field("date", pa.timestamp("ns")),
    pa.field("open", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("turnover", pa.float32()),
    pa.field("amount", pa.float64()),
    pa.field("code", pa.string()),
    pa.field("turn20", pa.float32()),
    pa.field("am20", pa.float32()),
    pa.field("volume", pa.float32()),
])

KEEP_COLS = ["date", "open", "close", "turnover", "amount", "code",
             "turn20", "am20", "volume"]
BUFFER_DAYS = 40


def load_universe_codes() -> list[str]:
    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"股票池不存在: {UNIVERSE_FILE}")
    df = pd.read_csv(UNIVERSE_FILE, dtype={"code": str})
    codes = sorted({str(c).zfill(6) for c in df["code"].astype(str)})
    if not codes:
        raise SystemExit(f"股票池为空: {UNIVERSE_FILE}")
    return codes


def load_ts_code_map() -> dict[str, str]:
    """6 位代码 -> 完整 ts_code（读导出快照 stock_basic.parquet）。"""
    path = ROOT / "data" / "pg_parquet" / "stock_basic.parquet"
    if not path.exists():
        raise SystemExit(f"股票注册表不存在: {path}（先跑 export_pg_to_parquet.py）")
    df = pd.read_parquet(path, columns=["ts_code", "symbol"])
    return {str(r["symbol"]).zfill(6): str(r["ts_code"]) for _, r in df.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser(description="从 PG 原始价+复权因子重建股票 panel.parquet")
    ap.add_argument("--start", default="2015-01-01", help="面板起始日期（默认 2015-01-01，覆盖旧 panel 全部区间）")
    ap.add_argument("--batch", type=int, default=200, help="每批股票数（内存安全）")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    calc_start = (start - pd.Timedelta(days=BUFFER_DAYS)).date().isoformat()
    codes = load_universe_codes()
    ts_map = load_ts_code_map()
    ts_by_code = {c: ts_map.get(c) for c in codes}
    missing = [c for c, t in ts_by_code.items() if t is None]
    if missing:
        print(f"警告: {len(missing)} 只股票不在 stock_basic 中（跳过）: {missing[:10]}", flush=True)
    ts_codes = sorted(t for t in ts_by_code.values() if t)
    stock_path = ROOT / "data" / "pg_parquet" / "stock_daily.parquet"
    if not stock_path.exists():
        raise SystemExit(f"行情快照不存在: {stock_path}（先跑 export_pg_to_parquet.py）")

    con = duckdb.connect()
    max_date = con.execute(
        "SELECT max(trade_date) FROM read_parquet(?)", [str(stock_path)]
    ).fetchone()[0]
    print(f"股票池 {len(codes)} 只（可匹配 {len(ts_codes)} 只），行情 {start.date()} ~ {max_date}（起点前移 {BUFFER_DAYS} 天做滚动因子）", flush=True)

    t0 = time.time()
    rows_total = 0
    writer = None
    out_path = PANEL_FILE
    tmp_path = PANEL_FILE.with_name(f".{PANEL_FILE.name}.rebuild.tmp")
    batches = [ts_codes[i:i + args.batch] for i in range(0, len(ts_codes), args.batch)]
    for bi, batch in enumerate(batches, 1):
        marks = ", ".join(["?"] * len(batch))
        sql = f"""
            SELECT ts_code, trade_date AS date, open, high, low, close,
                   vol AS volume, amount, turnover_rate AS turnover, adj_factor
            FROM read_parquet(?)
            WHERE ts_code IN ({marks}) AND trade_date >= CAST(? AS DATE)
            ORDER BY ts_code, trade_date
        """
        df = con.execute(sql, [str(stock_path)] + batch + [calc_start]).fetchdf()
        if df.empty:
            print(f"  batch {bi}/{len(batches)}: 空", flush=True)
            continue
        out = _finalize_stock_df(df, adj="qfq")
        # 停牌/缺行情日只有 adj_factor 没有价格，与 PG 路径 close IS NOT NULL 口径一致
        out = out.dropna(subset=["open", "close"])
        out = out[out["date"] >= start]
        if out.empty:
            print(f"  batch {bi}/{len(batches)}: 空（过滤后）", flush=True)
            continue
        out = out[KEEP_COLS].copy()
        out["code"] = out["code"].astype(str)
        table = pa.Table.from_pandas(out, schema=PANEL_SCHEMA, preserve_index=False)
        if writer is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(tmp_path), PANEL_SCHEMA, compression="zstd")
        writer.write_table(table)
        rows_total += len(out)
        print(f"  batch {bi}/{len(batches)}: +{len(out)} 行（累计 {rows_total}）", flush=True)

    if writer is None:
        raise SystemExit("无数据写入，panel 未重建")
    writer.close()
    os.replace(tmp_path, out_path)  # 原子替换，避免写入中断损坏正在使用的面板
    print(f"完成: rows={rows_total} codes={len(codes)} time={time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
