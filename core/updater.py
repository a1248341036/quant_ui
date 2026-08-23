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
CNE_SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "sync_daily_to_cne.py"
REBUILD_SCRIPT = PROJECT_ROOT / "scripts" / "rebuild_stock_panel_from_pg.py"
FUND_FEES_SCRIPT = PROJECT_ROOT / "scripts" / "refresh_fund_fees.py"


def sync_market_data(end: str) -> None:
    """委托 CNEquity fetcher 增量合并日线到年度档案。"""
    try:
        r = subprocess.run(
            [sys.executable, str(CNE_SYNC_SCRIPT), "--end", end],
            capture_output=True, text=True, timeout=3600,
        )
        print(r.stdout.strip(), flush=True)
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"CNE 日线同步失败（不影响 panel 更新）: {exc}",
              file=sys.stderr, flush=True)


def rebuild_stock_panel() -> None:
    """从 CNE 年度 stock_daily 档案重建/增量更新股票 panel.parquet。"""
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


def refresh_fund_fees(max_workers: int = 4) -> None:
    """补齐缺失的基金费率；已有成功记录不会重复抓取。"""
    try:
        r = subprocess.run(
            [sys.executable, str(FUND_FEES_SCRIPT), "--workers", str(max(1, min(max_workers, 8)))],
            capture_output=True, text=True, timeout=3600,
        )
        if r.stdout.strip():
            print(r.stdout.strip(), flush=True)
        if r.returncode != 0 and r.stderr.strip():
            print(r.stderr.strip(), file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"基金费率同步失败（不影响主流程）: {exc}", file=sys.stderr, flush=True)


def refresh_all(mode: str = "incremental", end: str | None = None,
                max_workers: int = 6, include_stocks: bool = True,
                sync_tushare: bool = True, rebuild_panel: bool = True,
                sync_fund_fees: bool = True, progress=None) -> dict:
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
    if sync_fund_fees:
        if progress:
            progress(7, 7, "补齐基金费率...")
        refresh_fund_fees(max_workers=max_workers)
    if progress:
        progress(7, 7, "完成")
    return result
