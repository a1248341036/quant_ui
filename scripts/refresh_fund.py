#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只刷新场外基金（池子 + 净值），不碰股票和 ETF。

用法:
    python scripts/refresh_fund.py
    python scripts/refresh_fund.py --workers 8   # 加大并发
"""
from __future__ import annotations

import sys
import time
import argparse
import subprocess
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fetcher  # noqa: E402
from core.store import FUND_FILE, FUND_NAV_FILE, save_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新场外基金净值")
    parser.add_argument("--workers", type=int, default=6,
                        help="并发线程数（默认 6）")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="净值起始日期（默认 2015-01-01）")
    parser.add_argument("--no-fees", action="store_true", help="不更新基金费率")
    args = parser.parse_args()

    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = args.start

    print("拉取全量基金池...", flush=True)
    fund = fetcher.fetch_fund_universe()
    save_csv(fund, FUND_FILE)
    codes = sorted(fund["code"].tolist())
    print(f"基金池: {len(codes)} 只", flush=True)

    fund_existing = None
    if FUND_NAV_FILE.exists():
        fund_existing = pd.read_parquet(FUND_NAV_FILE)
        fund_existing["code"] = fund_existing["code"].astype(str).str.zfill(6)
        print(f"已有净值: {len(fund_existing)} 行, {fund_existing['code'].nunique()} 只", flush=True)

    print(f"开始拉取净值 {start} ~ {end}, {args.workers} 线程并发...", flush=True)
    t0 = time.time()

    def progress(done, total, code):
        if done % 500 == 0 or done == total:
            elapsed = time.time() - t0
            pct = done / total * 100
            eta = elapsed / done * (total - done) if done else 0
            print(f"  [{done}/{total}] {pct:.0f}%  {elapsed:.0f}s  ETA {eta:.0f}s  last={code}", flush=True)

    fund_panel = fetcher.fetch_fund_navs(
        codes,
        start=start,
        end=end,
        existing=fund_existing,
        max_workers=args.workers,
        progress=progress,
    )

    if len(fund_panel):
        FUND_NAV_FILE.parent.mkdir(parents=True, exist_ok=True)
        fund_panel.to_parquet(FUND_NAV_FILE, index=False)

    elapsed = time.time() - t0
    n_codes = fund_panel["code"].nunique() if len(fund_panel) else 0
    print(f"\n完成: {n_codes} 只, {len(fund_panel)} 行, 耗时 {elapsed:.0f}s ({elapsed/60:.1f} 分钟)", flush=True)
    if len(fund_panel):
        print(f"数据范围: {fund_panel['date'].min().date()} ~ {fund_panel['date'].max().date()}", flush=True)
    if not args.no_fees:
        fee_script = Path(__file__).with_name("refresh_fund_fees.py")
        subprocess.run([sys.executable, str(fee_script), "--workers", str(args.workers)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
