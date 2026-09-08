# -*- coding: utf-8 -*-
"""资金面数据插件：CNE curated `margin_trading`（融资融券，日频，东财源）。

两融余额反映杠杆资金的多空方向与拥挤度：
- 融资余额/融资买入额 = 多头杠杆情绪；
- 融券余额/融券卖出量 = 空头意愿（券源受限，规模远小于融资，解读需分开）。

日频表直接按 (symbol, trade_date) 对日期窗口过滤即可 join 核心行情面板，
不需要 PIT 网格（本数据集无公告/披露时点概念，T 日盘后即 T 日可用）。

列口径（单位均为源直给）：
- margin_balance    融资余额（元）
- margin_buy        融资买入额（元）
- short_balance     融券余额（元）
- short_sell_volume 融券卖出量（股）
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

# ── 列映射：CNE 原始列 → Panel 列名 ──────────────────────────────────

_COLUMN_MAP: dict[str, str] = {
    "margin_balance": "mgn_balance",
    "margin_buy": "mgn_buy",
    "short_balance": "mgn_short_balance",
    "short_sell_volume": "mgn_short_sell_vol",
}

_ALL_MGN_COLS = list(_COLUMN_MAP.values())

PLUGIN = DataSourcePlugin(
    name="margin",
    dataset="margin_trading",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map=_COLUMN_MAP,
    priority=51,
)


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """加载融资融券并按日期窗口裁剪（直接输出原始列名，由 registry 映射）。"""
    s, e = _pitlib.parse_window(start, end)
    raw = _pitlib.read_curated("margin_trading")

    keep = list(_COLUMN_MAP.keys()) + ["symbol", "trade_date"]
    df = raw.select([c for c in keep if c in raw.columns]).filter(
        (pl.col("trade_date") >= s) & (pl.col("trade_date") <= e)
    )
    if df.is_empty():
        raise ValueError(f"margin: 窗口内无数据 ({s}~{e})")
    df = df.unique(subset=["symbol", "trade_date"], keep="last").sort(
        ["trade_date", "symbol"]
    )
    pdf = df.to_pandas()
    logger.info("margin adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, s, e)
    return pdf
