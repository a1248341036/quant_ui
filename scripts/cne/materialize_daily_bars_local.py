# -*- coding: utf-8 -*-
"""Materialize curated daily_bars from the local tushare-wide archive.

The repair path re-fetched every trading day from the tushare API even though
the identical data (33 identical columns) sits in the external wide parquet.
This seeds the remaining days directly from local files into the same staging
layout the backfill used, then the normal compact merges them into curated.

Reads the coverage already staged (2017-06-27..2025-01-23) and fills the rest
of 2016-01-01..today from data/quant_dataset/<year>/<year>/day/stock_daily.parquet.
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, r"D:\Quant\quant_ui\CNEquity\src")

from cnequity.config import load_config  # noqa: E402
from cnequity.storage import StagingWriter  # noqa: E402

CONFIG_PATH = r"D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml"
WIDE_GLOB = r"D:\Quant\quant_ui\data\quant_dataset\*\*\day\stock_daily.parquet"
STAGING_ROOT = Path(r"D:\Quant\quant_ui\CNEquity\data\quant_dataset\_cnequity\staging")

# Schema contract observed from the backfill's own staging files.
EXPECTED_COLS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount", "up_limit", "down_limit",
    "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb",
    "ps", "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share",
    "free_share", "total_mv", "circ_mv", "adj_factor", "suspend_timing",
    "suspend_type", "is_st", "listed_days",
]


def main() -> int:
    config = load_config(CONFIG_PATH)
    run_id = str(uuid.uuid4())
    writer = StagingWriter(config.staging_root)

    wide = pl.scan_parquet(
        [str(f) for f in sorted(Path(r"D:\Quant\quant_ui\data\quant_dataset").glob(r"*\*\day\stock_daily.parquet"))]
    ).select(EXPECTED_COLS).with_columns(pl.col("trade_date").cast(pl.Date))

    # Days already staged (both the API walk and any prior local run):
    # parse dates from ALL staging parquet filenames for daily_bars.
    staged: set[date] = set()
    for f in STAGING_ROOT.glob("daily_bars/run_id=*/**/*.parquet"):
        try:
            staged.add(date.fromisoformat(f.stem.replace("part-tushare-wide-", "").replace("batch-local-", "")))
        except ValueError:
            pass
    print(f"already staged days: {len(staged)} ({min(staged)} .. {max(staged)})" if staged else "no staged days")

    all_days = (
        wide.select("trade_date").unique().collect().get_column("trade_date").sort().to_list()
    )
    todo = [d for d in all_days if d not in staged]
    print(f"days in wide archive: {len(all_days)}, to materialize: {len(todo)}")

    written = 0
    BATCH = 60  # rows per batch file size is fine; one file per day like the API walk
    for d in todo:
        day_df = (
            wide.filter(pl.col("trade_date") == d).collect()
        )
        if day_df.is_empty():
            continue
        writer.write_batch("daily_bars", run_id, f"batch-local-{d.isoformat()}", day_df)
        written += 1
        if written % 200 == 0:
            print(f"  {written}/{len(todo)} days staged ({d})")
    print(f"done: {written} days -> staging daily_bars run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
