#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日收盘后增量刷新行情数据（systemd timer 调用）。

默认在 panel 更新后把日线跟进到 PostgreSQL/TimescaleDB（stock_daily），
PG 未配置或失败时不影响 panel 更新；可用 --no-sync-pg 关闭。
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.updater import refresh_all


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新行情面板")
    parser.add_argument("--workers", type=int, default=3,
                        help="抓取日线并发线程数（内存紧张时调低，如 3-4）")
    parser.add_argument("--no-sync-pg", action="store_true",
                        help="panel 刷新后不同步 PostgreSQL")
    parser.add_argument("--no-export-parquet", action="store_true",
                        help="PG 同步后不导出 Parquet（默认导出 stock_daily）")
    parser.add_argument("--export-tables",
                        default="stock_daily,stock_basic,fina_indicator,income",
                        help="要导出的表（逗号分隔，默认行情/财务核心表；all=全部白名单）")
    parser.add_argument("--skip-stock-panel", action="store_true",
                        help="跳过腾讯股票日线刷新（股票行情由 Tushare PG 承担），"
                             "仅保留 ETF/基金/指数")
    args = parser.parse_args()
    print("start refresh", flush=True)
    try:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        result = refresh_all(mode="incremental", end=end, max_workers=args.workers,
                             include_stocks=not args.skip_stock_panel,
                             sync_pg=not args.no_sync_pg,
                             export_parquet_tables=None if args.no_export_parquet
                             else ("all" if args.export_tables == "all"
                                   else args.export_tables))
        print(f"ok: {result}", flush=True)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
