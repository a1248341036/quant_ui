#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日收盘后增量刷新行情数据（systemd timer 调用）。

股票日线由 CNEquity fetcher 增量写入 data/quant_dataset 年度档案，再由
rebuild_stock_panel_from_pg.py 重建 panel；可用 --no-sync-pg 关闭日线同步。
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.qqnotify import send_qq_text  # noqa: E402
from core.run_log import job  # noqa: E402
from core.store import (  # noqa: E402
    ETF_PANEL_FILE,
    FUND_PANEL_FILE,
    PANEL_FILE,
)
from core.updater import refresh_all


def _max_date_parquet(path: Path) -> str:
    """轻量读取 parquet 的 date 列最大值（DuckDB，不整表进 pandas）。"""
    try:
        import duckdb
        con = duckdb.connect()
        try:
            row = con.execute(
                "SELECT strftime(max(date), '%Y-%m-%d') FROM read_parquet(?)",
                [str(path)],
            ).fetchone()
            return row[0] if row and row[0] else ""
        finally:
            con.close()
    except Exception as exc:
        print(f"[refresh] 读取 {path.name} 最新日期失败: {exc}",
              file=sys.stderr, flush=True)
        return ""


def _parquet_max_trade_date() -> str:
    """查询 CNE 年度日线档案最新交易日。"""
    try:
        import duckdb
        from core.store import QUANT_DATASET_DIR

        files = [str(p) for p in QUANT_DATASET_DIR.glob("*/**/day/stock_daily.parquet")]
        if not files:
            return ""
        con = duckdb.connect()
        try:
            row = con.execute(
                "SELECT strftime(max(trade_date), '%Y-%m-%d') FROM read_parquet(?)",
                [files],
            ).fetchone()
            return row[0] if row and row[0] else ""
        finally:
            con.close()
    except Exception as exc:
        print(f"[refresh] 查询 CNE stock_daily 最新日期失败: {exc}",
              file=sys.stderr, flush=True)
        return ""


def _refresh_summary(started: datetime) -> str:
    now = datetime.now()
    elapsed = str(now - started).split(".")[0]
    return "\n".join([
        "【quant_ui】数据刷新完成 ✅",
        f"时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（耗时 {elapsed}）",
        f"股票 panel 最新：{_max_date_parquet(PANEL_FILE) or 'N/A'}",
        f"ETF panel 最新：{_max_date_parquet(ETF_PANEL_FILE) or 'N/A'}",
        f"基金 panel 最新：{_max_date_parquet(FUND_PANEL_FILE) or 'N/A'}",
        f"Tushare 日线最新：{_parquet_max_trade_date() or 'N/A'}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新行情面板")
    parser.add_argument("--workers", type=int, default=3,
                        help="抓取日线并发线程数（内存紧张时调低，如 3-4）")
    parser.add_argument("--no-sync-pg", action="store_true",
                        help="panel 刷新后不同步 Tushare 日线到 Parquet")
    parser.add_argument("--no-rebuild-panel", action="store_true",
                        help="不重建股票 panel.parquet（默认从 Tushare parquet 重建）")
    parser.add_argument("--skip-stock-panel", action="store_true",
                        help="跳过腾讯股票日线刷新（股票行情由 Tushare parquet 承担），"
                             "仅保留 ETF/基金/指数")
    parser.add_argument("--no-fund-fees", action="store_true",
                        help="不补齐基金费率数据")
    args = parser.parse_args()
    started = datetime.now()
    send_qq_text(
        "【quant_ui】数据刷新开始 \U0001f504\n"
        f"时间：{started.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "任务：ETF/基金/指数 + Tushare 日线直写 Parquet + Panel 重建"
    )
    print("start refresh", flush=True)
    run_meta = {
        "mode": "incremental",
        "workers": args.workers,
        "skip_stock_panel": args.skip_stock_panel,
        "sync_tushare": not args.no_sync_pg,
        "rebuild_panel": not args.no_rebuild_panel,
        "sync_fund_fees": not args.no_fund_fees,
    }
    with job("quant_ui:refresh_data", metadata=run_meta):
        try:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            result = refresh_all(mode="incremental", end=end, max_workers=args.workers,
                                 include_stocks=not args.skip_stock_panel,
                                 sync_tushare=not args.no_sync_pg,
                                 rebuild_panel=not args.no_rebuild_panel,
                                 sync_fund_fees=not args.no_fund_fees,
                                 )
            print(f"ok: {result}", flush=True)
            send_qq_text(_refresh_summary(started))
            return 0
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr, flush=True)
            send_qq_text(
                "【quant_ui】数据刷新失败 \u274c\n"
                f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"错误：{exc}"
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
