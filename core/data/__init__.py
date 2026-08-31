"""数据加载入口：股票面板 / 资产池 / ETF / 基金 / 指数 / 数据状态。

原 core/data.py（785行）拆分为 3 个子模块：
- panel.py  — 股票面板加载（CNE/预计算/DuckDB 多路径）、复权因子、单股详情
- assets.py — 资产池加载（股票池/行业/ETF/基金/指数）
- status.py — 数据源状态报告

本 __init__.py re-export 全部公共 API，外部 import 路径不变。
"""
from __future__ import annotations

# 从 store 透传的常量（外部代码可能 from core.data import PANEL_FILE 等）
from ..store import (
    DATA_DIR,
    PANEL_FILE,
    ETF_FILE,
    ETF_PANEL_FILE,
    FUND_FILE,
    FUND_FEE_FILE,
    FUND_NAV_FILE,
    FUND_PANEL_FILE,
    INDEX_FILE,
    LEGACY_DATA_DIR,
    PG_PARQUET_DIR,
    QUANT_DATASET_DIR,
    SENTIMENT_DIR,
    PRED_FILE,
    TECH_FILE,
    UNIVERSE_FILE,
    load_meta,
)

# ---------- panel.py ----------
from .panel import (
    PANEL_PATH,
    PANEL_START,
    FACTOR_BUFFER_DAYS,
    DATA_SOURCE,
    SIGNAL_LOOKBACK_DAYS,
    _CNE_DAILY_GLOB,
    _cne_daily_years,
    _cne_latest_year_file,
    _load_last_adj,
    reset_last_adj_cache,
    _finalize_stock_df,
    _finalize_panel_df,
    _duck_query,
    _panel_sql_where,
    _duck_panel_slice,
    _pg_parquet_end,
    _code_to_ts_map,
    _load_panel_pg_parquet,
    _load_panel_precomputed,
    _load_panel_impl,
    load_panel,
    load_signal_panel,
    load_panel_codes,
    load_stock_detail,
    load_pred_scores,
    reset_caches,
    duck_query,
)

# ---------- assets.py ----------
from .assets import (
    UNIVERSE_PATH,
    TECH_PATH,
    INDEX_PATH,
    load_universe,
    load_tech,
    load_etf,
    load_etf_panel,
    load_etf_panel_codes,
    load_fund,
    load_fund_nav,
    load_fund_nav_codes,
    load_fund_panel,
    load_index,
)

# ---------- status.py ----------
from .status import (
    _file_entry,
    data_status,
)

__all__ = [
    # 面板
    "load_panel",
    "load_signal_panel",
    "load_panel_codes",
    "load_stock_detail",
    "load_pred_scores",
    "reset_caches",
    "duck_query",
    "PANEL_PATH",
    "PANEL_START",
    "FACTOR_BUFFER_DAYS",
    "DATA_SOURCE",
    "SIGNAL_LOOKBACK_DAYS",
    # 资产池
    "load_universe",
    "load_tech",
    "load_etf",
    "load_etf_panel",
    "load_etf_panel_codes",
    "load_fund",
    "load_fund_nav",
    "load_fund_nav_codes",
    "load_fund_panel",
    "load_index",
    # 状态
    "data_status",
]
