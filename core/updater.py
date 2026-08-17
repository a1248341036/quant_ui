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
EXPORT_DEFAULT_TABLES = "stock_daily"


def sync_postgres(end: str) -> None:
    """把 panel 日线同步到 PostgreSQL/TimescaleDB；未配置或失败不影响主流程。"""
    try:
        if not pg.configured():
            print("PG_DSN 未配置，跳过 PostgreSQL 日线同步", flush=True)
            return
        since = (pd.Timestamp(end) - pd.Timedelta(days=10)).strftime("%Y%m%d")
        r1 = subprocess.run(
            [sys.executable, str(SYNC_SCRIPT), "--daily-from-panel"],
            capture_output=True, text=True, timeout=1800,
        )
        print(r1.stdout.strip(), flush=True)
        if r1.returncode != 0:
            print(r1.stderr.strip(), file=sys.stderr, flush=True)
            return
        r2 = subprocess.run(
            [sys.executable, str(SYNC_SCRIPT), "--daily-since", since],
            capture_output=True, text=True, timeout=3600,
        )
        print(r2.stdout.strip(), flush=True)
        if r2.returncode != 0:
            print(r2.stderr.strip(), file=sys.stderr, flush=True)
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
    except Exception as exc:
        print(f"PG->Parquet 导出失败（不影响主流程）: {exc}",
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
                progress=None) -> dict:
    """一键全量刷新：行情源数据 + PostgreSQL 同步 + PG->Parquet + 基金衍生面板。"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    result = update_data(mode=mode, end=end, max_workers=max_workers,
                         progress=progress, include_stocks=include_stocks)
    if sync_pg:
        if progress:
            progress(6, 6, "同步 PostgreSQL 日线...")
        sync_postgres(end)
    if export_parquet_tables:
        if progress:
            progress(6, 6, "导出 PG 数据到 Parquet...")
        export_parquet(export_parquet_tables)
    if progress:
        progress(6, 6, "重建基金衍生面板...")
    rebuild_fund_panel()
    if progress:
        progress(6, 6, "完成")
    return result
