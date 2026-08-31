"""一键更新入口：股票池 + 指数 + 日线 + ETF + 基金。"""
from __future__ import annotations

import sys
from typing import Callable

import pandas as pd

from ..store import (
    ETF_FILE, ETF_PANEL_FILE, FUND_FILE, LEGACY_DATA_DIR,
    TECH_FILE, UNIVERSE_FILE,
    save_csv, save_meta, save_panel,
)

from .stocks import (
    fetch_universe, fetch_tech_universe, fetch_daily_bars,
    _load_cached_universe, _add_rolling_factors, _compact_panel,
)
from .funds import (
    fetch_etf_universe, fetch_etf_daily_bars,
    fetch_fund_universe,
)


def update_data(
    mode: str = "incremental",
    start: str = "2020-01-01",
    end: str | None = None,
    max_workers: int = 6,
    progress: Callable[[int, int, str], None] | None = None,
    include_stocks: bool = True,
) -> dict:
    """一键更新：股票池 + 指数 + 日线缓存。"""
    from ..store import PANEL_FILE, save_panel as _save_panel

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    # include_stocks=False（股票由 Tushare parquet 承担）时不会刷新股票面板，
    # 初始化空表避免后续 save_meta 引用未定义变量。
    panel = pd.DataFrame(columns=["date", "code"])
    stage = {"text": "正在更新股票池..."}
    if progress:
        progress(0, 6, stage["text"])
    try:
        universe = fetch_universe()
        save_csv(universe, UNIVERSE_FILE)
    except Exception:
        universe = _load_cached_universe()
        if universe is None:
            raise
    if progress:
        progress(1, 6, "指数由 CNE 流水线维护（step_index_bars_external），跳过...")
    if progress:
        progress(2, 6, "更新行业分类...")
    tech = fetch_tech_universe(universe)
    save_csv(tech, TECH_FILE)
    if progress:
        progress(3, 6, "行业完成，增量更新日线...")

    if include_stocks:
        existing = None
        if mode != "full":
            if not PANEL_FILE.exists():
                legacy = LEGACY_DATA_DIR / "panel/turn20/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet"
                if legacy.exists():
                    _save_panel(_add_rolling_factors(pd.read_parquet(legacy)))
            if PANEL_FILE.exists():
                existing = pd.read_parquet(PANEL_FILE)
                existing["code"] = existing["code"].astype(str).str.zfill(6)
                existing = _compact_panel(existing)

        panel = fetch_daily_bars(
            sorted(universe["code"]),
            start=start,
            end=end,
            existing=existing,
            max_workers=max_workers,
            progress=lambda d, t, c: progress(3 + d / max(t, 1) * 0.9, 6, f"日线 {c} ({d}/{t})")
            if progress else None,
        )
        save_panel(panel)
    else:
        print("跳过腾讯股票日线（股票行情由 Tushare parquet 承担）", flush=True)

    # ---- ETF：池子 + 日线 ----
    if progress:
        progress(4.4, 6, "更新 ETF 列表...")
    try:
        etf = fetch_etf_universe()
        save_csv(etf, ETF_FILE)
        etf_existing = None
        if ETF_PANEL_FILE.exists():
            etf_existing = pd.read_parquet(ETF_PANEL_FILE)
            etf_existing["code"] = etf_existing["code"].astype(str).str.zfill(6)
            etf_existing = _compact_panel(etf_existing)
        etf_panel = fetch_etf_daily_bars(
            sorted(etf["code"]),
            start=start,
            end=end,
            existing=etf_existing,
            max_workers=max_workers,
            progress=lambda d, t, c: progress(4.5 + d / max(t, 1) * 0.5, 6,
                                               f"ETF {c} ({d}/{t})")
            if progress else None,
        )
        if len(etf_panel):
            ETF_PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            etf_panel.to_parquet(ETF_PANEL_FILE, index=False)
        etf_stats = {"n_codes": int(etf_panel["code"].nunique()) if len(etf_panel) else 0,
                     "n_rows": int(len(etf_panel))}
    except Exception as exc:
        print(f"[fetcher] ETF 更新失败（不影响股票面板）: {exc}", file=sys.stderr)
        etf_stats = {"n_codes": 0, "n_rows": 0}

    # ---- 场外基金池 ----
    # 净值本体已迁移到 CNE 流水线（step_fund_nav：EM 快照 → staging → compact
    # 直接合并回 fund_nav.parquet），这里只维护基金池清单供其过滤。
    if progress:
        progress(5.2, 6, "更新场外基金池...")
    try:
        fund = fetch_fund_universe()
        save_csv(fund, FUND_FILE)
        fund_stats = {"n_codes": int(len(fund)), "n_rows": 0}
    except Exception as exc:
        print(f"[fetcher] 场外基金池更新失败（不影响股票面板）: {exc}", file=sys.stderr)
        fund_stats = {"n_codes": 0, "n_rows": 0}
    if progress:
        progress(6, 6, "完成")

    if len(panel):
        # meta 记录面板真实覆盖范围（实际最后交易日），而不是请求的日历日；
        # 周末/节假日请求日（如 2026-08-16）不会产生行情，写请求日会误导展示。
        save_meta({"mode": mode,
                   "start": str(panel["date"].min().date()),
                   "end": str(panel["date"].max().date()),
                   "n_codes": int(panel["code"].nunique()),
                   "n_rows": int(len(panel))})
    return {"n_codes": int(panel["code"].nunique()) if len(panel) else 0,
            "n_rows": int(len(panel)),
            "etf": etf_stats, "fund": fund_stats}
