#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只刷新 ETF（池子 + 日线面板），不碰股票和基金。

用法:
    python scripts/refresh_etf.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fetcher  # noqa: E402
from core.store import ETF_FILE, ETF_PANEL_FILE, save_csv  # noqa: E402


def main() -> int:
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = "2020-01-01"

    print("拉取 ETF 池...", flush=True)
    etf = fetcher.fetch_etf_universe()
    save_csv(etf, ETF_FILE)
    codes = sorted(etf["code"].tolist())
    print(f"ETF 池: {len(codes)} 只", flush=True)

    etf_existing = None
    if ETF_PANEL_FILE.exists():
        etf_existing = pd.read_parquet(ETF_PANEL_FILE)
        etf_existing["code"] = etf_existing["code"].astype(str).str.zfill(6)
        etf_existing = fetcher._compact_panel(etf_existing)
        print(f"已有日线: {len(etf_existing)} 行, {etf_existing['code'].nunique()} 只", flush=True)

    print(f"开始拉取日线 {start} ~ {end}, 2 线程并发...", flush=True)
    t0 = time.time()

    etf_panel = fetcher.fetch_etf_daily_bars(
        codes,
        start=start,
        end=end,
        existing=etf_existing,
        max_workers=2,
    )

    if len(etf_panel):
        etf_panel["price_basis"] = "qfq"
        ETF_PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        etf_panel.to_parquet(ETF_PANEL_FILE, index=False)

    elapsed = time.time() - t0
    n_codes = etf_panel["code"].nunique() if len(etf_panel) else 0
    print(f"\n完成: {n_codes} 只, {len(etf_panel)} 行, 耗时 {elapsed:.0f}s ({elapsed/60:.1f} 分钟)", flush=True)
    if len(etf_panel):
        print(f"数据范围: {etf_panel['date'].min().date()} ~ {etf_panel['date'].max().date()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
