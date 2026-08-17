#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只刷新 ETF 与场外基金（池子 + 日线/净值），不重拉股票 panel。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fetcher  # noqa: E402
from core.store import (ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE,  # noqa: E402
                        save_csv)


def main() -> int:
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = "2020-01-01"

    print("start ETF", flush=True)
    etf = fetcher.fetch_etf_universe()
    save_csv(etf, ETF_FILE)
    etf_existing = None
    if ETF_PANEL_FILE.exists():
        etf_existing = pd.read_parquet(ETF_PANEL_FILE)
        etf_existing["code"] = etf_existing["code"].astype(str).str.zfill(6)
        etf_existing = fetcher._compact_panel(etf_existing)
    etf_panel = fetcher.fetch_etf_daily_bars(
        sorted(etf["code"]),
        start=start,
        end=end,
        existing=etf_existing,
        max_workers=2,
    )
    if len(etf_panel):
        ETF_PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        etf_panel.to_parquet(ETF_PANEL_FILE, index=False)
    print(f"ETF ok: codes={etf_panel['code'].nunique()} rows={len(etf_panel)} "
          f"end={etf_panel['date'].max().date()}", flush=True)

    print("start fund", flush=True)
    fund = fetcher.fetch_fund_universe()
    save_csv(fund, FUND_FILE)
    fund_existing = None
    if FUND_NAV_FILE.exists():
        fund_existing = pd.read_parquet(FUND_NAV_FILE)
        fund_existing["code"] = fund_existing["code"].astype(str).str.zfill(6)
    fund_panel = fetcher.fetch_fund_navs(
        sorted(fund["code"]),
        start=start,
        end=end,
        existing=fund_existing,
        max_workers=2,
    )
    if len(fund_panel):
        FUND_NAV_FILE.parent.mkdir(parents=True, exist_ok=True)
        fund_panel.to_parquet(FUND_NAV_FILE, index=False)
    print(f"FUND ok: codes={fund_panel['code'].nunique()} rows={len(fund_panel)} "
          f"end={fund_panel['date'].max().date()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
