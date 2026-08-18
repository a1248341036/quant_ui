from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from .data import load_fund_nav
from .fetcher import update_data
from .fund_engine import build_fund_panel
from .store import save_fund_panel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "sync_tushare_to_parquet.py"
REBUILD_SCRIPT = PROJECT_ROOT / "scripts" / "rebuild_stock_panel_from_pg.py"
STOCK_DAILY_PARQUET = PROJECT_ROOT / "data" / "pg_parquet" / "stock_daily.parquet"


def _parquet_max_trade_date() -> str:
    """查询 pg_parquet/stock_daily.parquet 最新交易日（轻量单行查询）。"""
    try:
        import duckdb
        con = duckdb.connect()
        try:
            row = con.execute(
                "SELECT strftime(max(trade_date), '%Y-%m-%d') FROM "
                "read_parquet(?)",
                [str(STOCK_DAILY_PARQUET)],
            ).fetchone()
            return row[0] if row and row[0] else ""
        finally:
            con.close()
    except Exception as exc:
        print(f"查询 stock_daily.parquet 最新日期失败: {exc}",
              file=sys.stderr, flush=True)
        return ""


def sync_market_data(end: str) -> None:
    """Tushare 直写 parquet：按交易日增量拉日线并重建复权因子锚点。"""
    try:
        latest = _parquet_max_trade_date()
        if latest:
            since = (pd.Timestamp(latest) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            since = (pd.Timestamp(end) - pd.Timedelta(days=10)).strftime("%Y%m%d")
        r = subprocess.run(
            [sys.executable, str(SYNC_SCRIPT), "--daily-since", since, "--last-adj"],
            capture_output=True, text=True, timeout=3600,
        )
        print(r.stdout.strip(), flush=True)
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"Tushare 日线同步失败（不影响 panel 更新）: {exc}",
              file=sys.stderr, flush=True)


def rebuild_stock_panel() -> None:
    """从 pg_parquet/stock_daily.parquet 重建/增量更新股票 panel.parquet。"""
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
                sync_tushare: bool = True, rebuild_panel: bool = True,
                progress=None) -> dict:
    """一键刷新：ETF/基金/指数源数据 + Tushare 直写 parquet + 股票/基金 panel。"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    # 股票 panel 最终会从 Tushare parquet 重建，update_data 里再抓一遍腾讯股票
    # 日线属于冗余，直接跳过，避免重复拉取和面板被腾讯 qfq 价覆盖。
    if include_stocks and rebuild_panel:
        print("股票 panel 将从 Tushare parquet 重建，跳过 update_data 中的腾讯股票日线抓取",
              flush=True)
        include_stocks = False
    result = update_data(mode=mode, end=end, max_workers=max_workers,
                         progress=progress, include_stocks=include_stocks)
    if sync_tushare:
        if progress:
            progress(7, 7, "同步 Tushare 日线到 Parquet...")
        sync_market_data(end)
    if rebuild_panel:
        if progress:
            progress(7, 7, "从 Tushare parquet 重建股票 panel...")
        rebuild_stock_panel()
    if progress:
        progress(7, 7, "重建基金衍生面板...")
    rebuild_fund_panel()
    if progress:
        progress(7, 7, "完成")
    return result
