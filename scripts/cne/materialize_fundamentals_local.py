# -*- coding: utf-8 -*-
"""Local materialization of fundamentals + trading calendar from pg_parquet.

Rebuilds without API traffic (2026-09 lake-wipe recovery):
- income / fina_indicator / express / forecast <- data/pg_parquet/<name>.parquet
- trading_calendar full 2009..2026 rebuild:
    2015-2026 is_open flags <- pg_parquet/trade_cal.parquet
    2009-2014 session days  <- wide-parquet bar evidence (all 1456 bar days are
        weekdays; the 109 bar-less weekdays are genuine holidays)

Staging layout matches the API walk, so `cne compact --run-id <id>` merges into
curated exactly like a fetched run.
"""
from __future__ import annotations

import datetime as dt
import sys
import uuid
from pathlib import Path

import polars as pl

sys.path.insert(0, r"D:\Quant\quant_ui\CNEquity\src")

from cnequity.config import load_config  # noqa: E402
from cnequity.domain.schemas import (  # noqa: E402
    DATASET_SCHEMAS,
    PRIMARY_KEYS,
    data_version_for,
    with_provenance,
)
from cnequity.storage import StagingWriter  # noqa: E402

CONFIG_PATH = r"D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml"
PG = Path(r"D:\Quant\quant_ui\data\pg_parquet")
WIDE_GLOB = r"D:\Quant\quant_ui\data\quant_dataset\*\*\day\stock_daily.parquet"

FINANCIALS = ["income", "fina_indicator", "express", "forecast"]
CALENDAR_START = dt.date(2009, 1, 1)
CALENDAR_END = dt.date(2026, 12, 31)


def materialize_financial(name: str, writer: StagingWriter, run_id: str) -> None:
    schema = DATASET_SCHEMAS[name]
    pk = PRIMARY_KEYS[name]
    df = pl.read_parquet(str(PG / f"{name}.parquet"))
    if "ts_code" in df.columns:
        df = df.rename({"ts_code": "symbol"})
    exprs = [
        pl.col(c).cast(dtype, strict=False) for c, dtype in schema.items() if c in df.columns
    ]
    df = df.select(exprs)
    before = df.height
    # Latest announcement wins: corrections re-announce the same period.
    sort_keys = [*pk, *[c for c in ("ann_date", "f_ann_date") if c in df.columns]]
    df = df.sort(sort_keys).unique(subset=pk, keep="last")
    df = with_provenance(df, source="pg_parquet", data_version=data_version_for(name))
    writer.write_batch(name, run_id, "batch-local-pg", df)
    print(f"{name}: {before} rows -> {df.height} after PK dedupe ({pk})", flush=True)


def materialize_calendar(writer: StagingWriter, run_id: str) -> None:
    wide_dates = (
        pl.scan_parquet(WIDE_GLOB)
        .select(pl.col("trade_date").unique())
        .collect()
        .get_column("trade_date")
        .to_list()
    )
    wide_set = {(d.date() if isinstance(d, dt.datetime) else d) for d in wide_dates}

    rows: list[tuple[dt.date, bool, str]] = []
    trade_cal = pl.read_parquet(str(PG / "trade_cal.parquet"))
    for _exch, cal_date, is_open, _pre in trade_cal.iter_rows():
        if CALENDAR_START <= cal_date <= CALENDAR_END:
            rows.append((cal_date, bool(is_open), "pg_trade_cal"))

    day = CALENDAR_START
    while day <= dt.date(2014, 12, 31):
        rows.append((day, day.weekday() < 5 and day in wide_set, "bar_evidence"))
        day += dt.timedelta(days=1)

    df = pl.DataFrame(
        rows,
        schema={"trade_date": pl.Date, "is_trading": pl.Boolean, "source": pl.Utf8},
        orient="row",
    )
    df = df.unique(subset=["trade_date"], keep="first").sort("trade_date")
    df = with_provenance(
        df, source="pg_trade_cal", data_version=data_version_for("trading_calendar")
    )
    trading = df.filter(pl.col("is_trading")).height
    writer.write_batch("trading_calendar", run_id, "batch-local-cal", df)
    print(
        f"trading_calendar: {df.height} calendar days ({trading} trading, "
        f"{df.height - trading} closed), "
        f"{df.get_column('trade_date').min()}..{df.get_column('trade_date').max()}",
        flush=True,
    )


def main() -> int:
    config = load_config(CONFIG_PATH)
    run_id = str(uuid.uuid4())
    writer = StagingWriter(config.staging_root)
    print(f"run_id: {run_id}", flush=True)
    for name in FINANCIALS:
        materialize_financial(name, writer, run_id)
    materialize_calendar(writer, run_id)
    print(f"done. next: cne compact --run-id {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
