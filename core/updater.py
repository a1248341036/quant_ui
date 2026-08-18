from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import pg
from .data import load_fund_nav
from .fetcher import update_data
from .fund_engine import build_fund_panel
from .store import save_fund_panel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "sync_postgres.py"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_pg_to_parquet.py"
REBUILD_SCRIPT = PROJECT_ROOT / "scripts" / "rebuild_stock_panel_from_pg.py"
# 运行时行情/财务来源：每日流式导出到 data/pg_parquet/，PG 不再承担运行读取
EXPORT_DEFAULT_TABLES = "stock_daily,stock_basic,fina_indicator,income"


def sync_postgres(end: str) -> None:
    """用 Tushare 不复权日线增量同步 PostgreSQL/TimescaleDB；未配置或失败不影响主流程。"""
    try:
        if not pg.configured():
            print("PG_DSN 未配置，跳过 PostgreSQL 日线同步", flush=True)
            return
        since = (pd.Timestamp(end) - pd.Timedelta(days=10)).strftime("%Y%m%d")
        # 股票日线由 Tushare 不复权价负责（--daily-since），不把腾讯/面板的
        # 前复权价写回 stock_daily：否则会覆盖 PG 里的不复权原始价，
        # 导致“历史价随最新行情漂移”和复权因子二次调整。
        r = subprocess.run(
            [sys.executable, str(SYNC_SCRIPT), "--daily-since", since],
            capture_output=True, text=True, timeout=3600,
        )
        print(r.stdout.strip(), flush=True)
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"PostgreSQL 日线同步失败（不影响 panel 更新）: {exc}",
              file=sys.stderr, flush=True)


def export_parquet(tables: str | None = None, batch: int = 50000) -> None:
    """PG 主要表流式导出到 Parquet；失败不影响主流程。"""
    if not pg.configured():
        print("PG_DSN 未配置，跳过 PG->Parquet 导出", flush=True)
        return
    tables = tables or EXPORT_DEFAULT_TABLES
    cmd = [sys.executable, str(EXPORT_SCRIPT), "--tables", tables,
           "--batch", str(batch)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(r.stdout.strip(), flush=True)
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr, flush=True)
        else:
            try:
                from . import db
                db.refresh_views()
            except Exception as exc:
                print(f"DuckDB 视图刷新失败: {exc}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"PG->Parquet 导出失败（不影响主流程）: {exc}",
              file=sys.stderr, flush=True)


def rebuild_stock_panel() -> None:
    """从 PG stock_daily 重建股票 panel.parquet（稳定前复权、原子替换）。"""
    if not pg.configured():
        print("PG_DSN 未配置，跳过股票 panel 重建", flush=True)
        return
    try:
        r = subprocess.run(
            [sys.executable, str(REBUILD_SCRIPT)],
            capture_output=True, text=True, timeout=3600,
        )
        print(r.stdout.strip(), flush=True)
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"股票 panel 重建失败（不影响主流程）: {exc}",
              file=sys.stderr, flush=True)


def rebuild_fund_panel() -> None:
    """从最新 fund_nav.parquet 重建基金衍生面板 fund_panel.parquet。"""
    try:
        nav = load_fund_nav()
        if len(nav) == 0:
            print("fund_nav 为空，跳过基金衍生面板重建", flush=True)
            return
        panel = build_fund_panel(nav)
        save_fund_panel(panel)
        print(f"fund_panel: rows={len(panel)} codes={panel['code'].nunique()}", flush=True)
    except Exception as exc:
        print(f"fund_panel 生成失败: {exc}", file=sys.stderr, flush=True)


def refresh_all(mode: str = "incremental", end: str | None = None,
                max_workers: int = 6, include_stocks: bool = True,
                sync_pg: bool = True, export_parquet_tables: str | None = "stock_daily",
                rebuild_panel: bool = True, progress=None) -> dict:
    """一键全量刷新：行情源数据 + PostgreSQL 同步 + 股票 panel 重建 +
    PG->Parquet + 基金衍生面板。"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    result = update_data(mode=mode, end=end, max_workers=max_workers,
                         progress=progress, include_stocks=include_stocks)
    if sync_pg:
        if progress:
            progress(7, 7, "同步 PostgreSQL 日线...")
        sync_postgres(end)
    if export_parquet_tables:
        if progress:
            progress(7, 7, "导出 PG 数据到 Parquet...")
        export_parquet(export_parquet_tables)
    if rebuild_panel:
        if progress:
            progress(7, 7, "从 PG 重建股票 panel...")
        rebuild_stock_panel()
    if progress:
        progress(7, 7, "重建基金衍生面板...")
    rebuild_fund_panel()
    if progress:
        progress(7, 7, "完成")
    return result
