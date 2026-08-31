"""行情数据抓取：腾讯 K线 / Tushare / AkShare。

原 core/fetcher.py（700行）拆分为 4 个子模块：
- kline.py  — 腾讯K线底层、符号转换、窗口拆分、面板压缩
- stocks.py — 股票池/行业/指数/日线抓取
- funds.py  — ETF/场外基金抓取
- update.py — 一键更新入口

本 __init__.py re-export 全部公共 API，外部 import 路径不变。
"""
from __future__ import annotations

# ---------- kline.py ----------
from .kline import (
    TX_URL,
    HEADERS,
    sina_symbol,
    split_windows,
    _kline_count,
    _fetch_kline,
    _fetch_kline_any,
    _fetch_index_tencent,
    _compact_panel,
    _add_rolling_factors,
)

# ---------- stocks.py ----------
from .stocks import (
    INDEX_SYMBOLS,
    fetch_universe,
    _load_cached_universe,
    fetch_indices,
    fetch_tech_universe,
    _load_cached_tech,
    fetch_daily_bars,
)

# ---------- funds.py ----------
from .funds import (
    FUND_TECH_KEYWORDS,
    fetch_etf_universe,
    fetch_etf_daily_bars,
    fetch_fund_universe,
    _parse_open_fund_daily,
    fetch_fund_navs,
)

# ---------- update.py ----------
from .update import (
    update_data,
)

__all__ = [
    # 股票
    "fetch_universe",
    "fetch_indices",
    "fetch_tech_universe",
    "fetch_daily_bars",
    "update_data",
    "INDEX_SYMBOLS",
    # ETF/基金
    "fetch_etf_universe",
    "fetch_etf_daily_bars",
    "fetch_fund_universe",
    "fetch_fund_navs",
    "FUND_TECH_KEYWORDS",
    # 工具
    "sina_symbol",
    "split_windows",
    "_compact_panel",
    "_add_rolling_factors",
]
