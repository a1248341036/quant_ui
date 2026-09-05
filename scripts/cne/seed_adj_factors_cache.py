# -*- coding: utf-8 -*-
"""Seed the adj_factors per-symbol cache from the surviving tushare wide parquet.

The 2026-09-03 lake wipe destroyed meta/adj_factors_cache, forcing the derive
to re-fetch 5832 symbols from Sina at ~1s each. The tushare wide daily parquet
(data/quant_dataset/<year>/<year>/day/stock_daily.parquet) carries an adj_factor
column -- the same hfq cumulative factor -- so the cache can be rebuilt locally
in minutes. Rows are compressed to the step-function representation Sina emits
(one row per factor-level change) to match the cache's native semantics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, r"D:\Quant\quant_ui\CNEquity\src")

from cnequity.config import load_config  # noqa: E402
from cnequity.derive.adj_factors import _cache_path  # noqa: E402
from cnequity.storage.atomic import write_parquet_atomic  # noqa: E402

CONFIG_PATH = r"D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml"
WIDE_GLOB = r"D:\Quant\quant_ui\data\quant_dataset\*\*\day\stock_daily.parquet"


def main() -> int:
    config = load_config(CONFIG_PATH)
    files = sorted(Path(r"D:\Quant\quant_ui\data\quant_dataset").glob(r"*\*\day\stock_daily.parquet"))
    if not files:
        print("no wide parquet found", file=sys.stderr)
        return 1
    print(f"scanning {len(files)} year files: {files[0].parent.parent.parent} .. {files[-1].parent.parent.parent}")

    lf = pl.scan_parquet([str(f) for f in files])
    df = (
        lf.select(
            pl.col("ts_code").alias("symbol"),
            pl.col("trade_date").cast(pl.Date).alias("trade_date"),
            pl.col("adj_factor").alias("factor"),
        )
        .drop_nulls(subset=["factor"])
        .filter(pl.col("factor") > 0)
        .collect(streaming=True)
    )
    print(f"rows with adj_factor: {df.height:,}  symbols: {df['symbol'].n_unique():,}")

    # Compress the daily series into the step function the cache natively holds:
    # keep a row only where the factor level changes (first row per symbol kept).
    df = df.sort(["symbol", "trade_date"])
    step = df.filter(
        (pl.col("factor") != pl.col("factor").shift(1).over("symbol"))
        | (pl.col("trade_date") == pl.col("trade_date").first().over("symbol"))
    )
    print(f"step rows (factor-level changes): {step.height:,}")

    written = 0
    for (sym,), group in step.group_by("symbol", maintain_order=True):
        factors = group.select("trade_date", "factor").sort("trade_date")
        path = _cache_path(config, sym, "hfq")
        write_parquet_atomic(path, factors, compression="zstd")
        written += 1
        if written % 500 == 0:
            print(f"  {written} cache files written")
    print(f"done: {written} cache files -> {config.meta_root / 'adj_factors_cache'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
