#!/usr/bin/env python3
"""增量同步股票日线到 CNEquity 年度档案。"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.store import QUANT_DATASET_DIR  # noqa: E402
from core.tushare_client import trade_dates  # noqa: E402


def latest_date(root: Path) -> date | None:
    files = [
        str(p)
        for p in root.glob("*/**/day/stock_daily.parquet")
        if p.parent.name == "day"
    ]
    if not files:
        return None
    import duckdb

    con = duckdb.connect()
    try:
        value = con.execute(
            "SELECT max(trade_date) FROM read_parquet(?)", [files]
        ).fetchone()[0]
    finally:
        con.close()
    return pd.Timestamp(value).date() if value is not None else None


def cne_config(root: Path):
    # The fetcher only needs these fields; avoid config-relative-path surprises.
    return SimpleNamespace(
        external_tushare_wide_root=root,
        external_tushare_wide_token=os.getenv("TUSHARE_TOKEN", "").strip(),
        external_tushare_wide_url=os.getenv("TUSHARE_URL", "").strip(),
        external_tushare_wide_interval=float(os.getenv("TUSHARE_SLEEP", "0.3")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="同步股票日线到 CNE 年度档案")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--since", default=None, help="覆盖自动计算的起始日期")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    end = pd.Timestamp(args.end).date()
    latest = latest_date(QUANT_DATASET_DIR)
    start = pd.Timestamp(args.since).date() if args.since else (
        latest + timedelta(days=1) if latest else end - timedelta(days=10)
    )
    if start > end:
        print(f"CNE 日线已是最新: {latest}")
        return 0

    dates = trade_dates(start.isoformat(), end.isoformat())
    print(f"CNE 日线缺口: {start} ~ {end}, {len(dates)} 个交易日")
    if args.dry_run or not dates:
        return 0

    from cnequity.external.tushare_fetch import fetch_and_merge_one_date

    cfg = cne_config(QUANT_DATASET_DIR)
    failed = []
    for raw in dates:
        d = pd.Timestamp(raw).date()
        result = fetch_and_merge_one_date(cfg, d)
        if result.get("status") != "success":
            failed.append((d.isoformat(), result.get("error", "unknown")))
        else:
            print(f"{d}: {result.get('rows_read', 0)} rows")
    if failed:
        for d, error in failed:
            print(f"{d} failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
