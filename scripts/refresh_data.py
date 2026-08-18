#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日收盘后增量刷新行情数据（systemd timer 调用）。

默认在 panel 更新后把日线跟进到 PostgreSQL/TimescaleDB（stock_daily），
PG 未配置或失败时不影响 panel 更新；可用 --no-sync-pg 关闭。
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.qqnotify import send_qq_text  # noqa: E402
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


def _pg_max_trade_date() -> str:
    """查询 PostgreSQL stock_daily 最新交易日（轻量单行查询）。"""
    try:
        from core import pg
        if not pg.configured():
            return ""
        with pg.get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT max(trade_date)::text FROM stock_daily")
            row = cur.fetchone()
            return row[0] if row and row[0] else ""
    except Exception as exc:
        print(f"[refresh] 查询 PG stock_daily 最新日期失败: {exc}",
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
        f"PG stock_daily 最新：{_pg_max_trade_date() or 'N/A'}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新行情面板")
    parser.add_argument("--workers", type=int, default=3,
                        help="抓取日线并发线程数（内存紧张时调低，如 3-4）")
    parser.add_argument("--no-sync-pg", action="store_true",
                        help="panel 刷新后不同步 PostgreSQL")
    parser.add_argument("--no-export-parquet", action="store_true",
                        help="PG 同步后不导出 Parquet（默认导出 stock_daily）")
    parser.add_argument("--no-rebuild-panel", action="store_true",
                        help="不重建股票 panel.parquet（默认从 PG 重建，稳定前复权）")
    parser.add_argument("--export-tables",
                        default="stock_daily,stock_basic,fina_indicator,income",
                        help="要导出的表（逗号分隔，默认行情/财务核心表；all=全部白名单）")
    parser.add_argument("--skip-stock-panel", action="store_true",
                        help="跳过腾讯股票日线刷新（股票行情由 Tushare PG 承担），"
                             "仅保留 ETF/基金/指数")
    args = parser.parse_args()
    started = datetime.now()
    send_qq_text(
        "【quant_ui】数据刷新开始 🔄\n"
        f"时间：{started.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "任务：ETF/基金/指数 + PG 日线 + Parquet 导出 + Panel 重建"
    )
    print("start refresh", flush=True)
    try:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        result = refresh_all(mode="incremental", end=end, max_workers=args.workers,
                             include_stocks=not args.skip_stock_panel,
                             sync_pg=not args.no_sync_pg,
                             rebuild_panel=not args.no_rebuild_panel,
                             export_parquet_tables=None if args.no_export_parquet
                             else ("all" if args.export_tables == "all"
                                   else args.export_tables))
        print(f"ok: {result}", flush=True)
        send_qq_text(_refresh_summary(started))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        send_qq_text(
            "【quant_ui】数据刷新失败 ❌\n"
            f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"错误：{exc}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
