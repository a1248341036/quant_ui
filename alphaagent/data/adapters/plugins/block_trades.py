# -*- coding: utf-8 -*-
"""大宗交易插件：CNE curated ``block_trades``（大宗/大宗交易折溢价，日频截面）。

数据：tushare 大宗交易记录（``symbol``/``trade_date``/``price``/``volume``/
``amount``/``premium_ratio``）。同一股票同一交易日可能有多笔。

产出列（每股每日聚合为一行）：
- ``bt_cnt``：当日大宗成交笔数
- ``bt_amount``：当日大宗成交总金额（万元比例口径，保留 a 股成交额一致量纲用元）
- ``bt_premium_wavg``：当日按金额加权的大宗折溢价率（%），无折溢价数据为 NaN
- ``bt_days_since``：距最近一次大宗交易的自然日（无则 NaN，行=仅已有大宗交易的股票）

内存设计：**按需薄列**——只输出 4 列并与主表 join，不携带原始长表；raw 在函数内
作用域即释放，不落面板全量。

背景：大宗交易折溢价反映大宗买方（机构/产业资本）的购买溢价意愿，可能蕴含对被成交
个股的短线价格预期，是独立信息面。折溢价非空率约 27.5%（其余为平价未披露）。
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 90

PLUGIN = DataSourcePlugin(
    name="block_trades",
    dataset="block_trades",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={
        "bt_cnt": "bt_cnt",
        "bt_amount": "bt_amount",
        "bt_premium_wavg": "bt_premium_wavg",
        "bt_days_since": "bt_days_since",
    },
    priority=52,
)


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """大宗交易记录 → 每股每日单行截面薄列。"""
    s, e = _pitlib.parse_window(start, end)
    ext = s - datetime.timedelta(days=_LOOKBACK_DAYS)
    raw = _pitlib.read_curated("block_trades", start=ext.isoformat(), end=e.isoformat())
    needed = {"symbol", "trade_date", "price", "volume", "amount", "premium_ratio"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"block_trades 缺列: {sorted(missing)}")

    daily = (
        raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("trade_date").cast(pl.Date),
            pl.col("amount").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("premium_ratio").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("trade_date").is_not_null())
        .group_by(["symbol", "trade_date"])
        .agg(
            pl.len().alias("bt_cnt"),
            pl.col("amount").sum().alias("amount_sum"),
            (
                (pl.col("premium_ratio") * pl.col("amount")).sum()
                / pl.col("amount").sum()
            ).alias("bt_premium_wavg"),
        )
        .sort(["symbol", "trade_date"])
    )

    # 距最近一次大宗交易的自然日（join_asof backward）
    days_since = (
        daily.select(
            pl.col("symbol"),
            pl.col("trade_date").alias("_bt_date"),
        )
        .unique()
        .sort(["symbol", "_bt_date"])
    )
    daily = daily.join_asof(
        days_since,
        left_on="trade_date",
        right_on="_bt_date",
        by="symbol",
        strategy="backward",
        suffix="_dummy",
    ).with_columns(
        (pl.col("trade_date") - pl.col("_bt_date"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("bt_days_since")
    ).drop("_bt_date")

    daily = daily.with_columns(
        pl.col("amount_sum").cast(pl.Float64).alias("bt_amount")
    ).drop("amount_sum")

    # 只在用户窗口输出（回看段仅用于播种 days_since 的初始状态）
    daily = daily.filter(pl.col("trade_date") >= s)
    if daily.is_empty():
        raise ValueError(f"block_trades: 窗口 {s}~{e} 内无大宗交易记录")

    out_cols = ["trade_date", "symbol", "bt_cnt", "bt_amount", "bt_premium_wavg", "bt_days_since"]
    pdf = daily.select(out_cols).to_pandas()
    logger.info("block_trades adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, s, e)
    return pdf