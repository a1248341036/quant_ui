# -*- coding: utf-8 -*-
"""披露/基本面数据插件：CNE curated `express`（业绩快报，PIT 日频阶跃）。

业绩快报是公司在定期报告前主动披露的初步财务数（主要在中报/年报前），以
``ann_date``（公告日）为 PIT 锚点展开为日频，公告日当天起引用最近一份快报。

源列口径（单位均源直给）：
- revenue / operate_profit / n_income        绝对额（元）
- diluted_eps                                每股收益（元）
- diluted_roe                                ROE（%）
- yoy_net_profit                             实际为**上年同期归母净利润**（绝对额，元），
  非同比%；插件用 (n_income / |yoy_net_profit| − 1) 计算 exp_netprofit_yoy（%）。

面板列（``exp_*``）：
- exp_revenue / exp_operate_profit / exp_net_profit
- exp_eps / exp_roe
- exp_netprofit_yoy   快报隐含归母净利同比（%，基准期亏损/为零时为 NaN）
- exp_days_since      距最近一次快报公告的自然日
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_ALL_EXP_COLS = [
    "exp_revenue", "exp_operate_profit", "exp_net_profit",
    "exp_eps", "exp_roe", "exp_netprofit_yoy", "exp_days_since",
]

PLUGIN = DataSourcePlugin(
    name="express",
    dataset="express",
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map={c: c for c in _ALL_EXP_COLS},
    priority=36,
)


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """加载业绩快报并按公告日 PIT 展开为日频阶跃。"""
    raw = _pitlib.read_curated("express").select(
        ["symbol", "ann_date", "end_date", "revenue", "operate_profit",
         "n_income", "diluted_eps", "diluted_roe", "yoy_net_profit"]
    )
    if raw.is_empty():
        raise ValueError("express: 无数据")

    events = (
        raw.with_columns(
            pl.when(
                pl.col("n_income").is_not_null()
                & pl.col("yoy_net_profit").is_not_null()
                & (pl.col("yoy_net_profit").abs() > 1e-9)
            )
            .then(pl.col("n_income") / pl.col("yoy_net_profit").abs() - 1.0)
            .otherwise(None)
            .alias("_yoy")
        )
        .rename({
            "revenue": "exp_revenue",
            "operate_profit": "exp_operate_profit",
            "n_income": "exp_net_profit",
            "diluted_eps": "exp_eps",
            "diluted_roe": "exp_roe",
        })
        # 同一公告日多份（不同报告期同日发布）取 end_date 最新一份
        .sort(["symbol", "ann_date", "end_date"])
        .group_by(["symbol", "ann_date"])
        .agg([pl.col(c).last() for c in _ALL_EXP_COLS
              if c not in ("exp_netprofit_yoy", "exp_days_since")]
             + [pl.col("_yoy").last().alias("exp_netprofit_yoy")])
        .rename({"ann_date": "_pit_date"})
        .filter(pl.col("_pit_date").is_not_null())
    )
    if events.is_empty():
        raise ValueError("express: 无有效快报记录")

    result = _pitlib.expand_pit_daily(
        events,
        start=start,
        end=end,
        anchor="_pit_date",
        since_col="exp_days_since",
        value_cols=_ALL_EXP_COLS,
    )
    pdf = result.to_pandas()
    logger.info("express adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
