#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日收盘后增量刷新行情数据（systemd timer 调用）。

默认在 panel 更新后把日线跟进到 PostgreSQL/TimescaleDB（stock_daily），
PG 未配置或失败时不影响 panel 更新；可用 --no-sync-pg 关闭。
"""
from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.fetcher import update_data


def _sync_pg(end: str) -> None:
    try:
        from core.pg import configured
        if not configured():
            print("PG_DSN 未配置，跳过 PostgreSQL 日线同步", flush=True)
            return
        since = (pd.Timestamp(end) - pd.Timedelta(days=10)).strftime("%Y%m%d")
        sync_script = str(Path(__file__).resolve().parent / "sync_postgres.py")
        r1 = subprocess.run([sys.executable, sync_script, "--daily-from-panel"],
                            capture_output=True, text=True, timeout=1800)
        print(r1.stdout.strip(), flush=True)
        if r1.returncode != 0:
            print(r1.stderr.strip(), file=sys.stderr, flush=True)
            return
        r2 = subprocess.run([sys.executable, sync_script, "--daily-since", since],
                            capture_output=True, text=True, timeout=3600)
        print(r2.stdout.strip(), flush=True)
        if r2.returncode != 0:
            print(r2.stderr.strip(), file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"PostgreSQL 日线同步失败（不影响 panel 更新）: {exc}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新行情面板")
    parser.add_argument("--workers", type=int, default=3,
                        help="抓取日线并发线程数（内存紧张时调低，如 3-4）")
    parser.add_argument("--no-sync-pg", action="store_true",
                        help="panel 刷新后不同步 PostgreSQL")
    parser.add_argument("--skip-stock-panel", action="store_true",
                        help="跳过腾讯股票日线刷新（股票行情由 Tushare PG 承担），"
                             "仅保留 ETF/基金/指数")
    args = parser.parse_args()
    print("start refresh", flush=True)
    try:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        result = update_data(mode="incremental", end=end, max_workers=args.workers,
                             include_stocks=not args.skip_stock_panel)
        print(f"ok: {result}", flush=True)
        if not args.no_sync_pg:
            _sync_pg(end)
        try:
            from core.data import load_fund_nav
            from core.fund_engine import build_fund_panel
            from core.store import FUND_PANEL_FILE, save_fund_panel
            nav = load_fund_nav()
            if len(nav):
                fp = build_fund_panel(nav)
                save_fund_panel(fp)
                print(f"fund_panel: rows={len(fp)} codes={fp['code'].nunique()}", flush=True)
        except Exception as exc:
            print(f"fund_panel 生成失败: {exc}", file=sys.stderr, flush=True)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
