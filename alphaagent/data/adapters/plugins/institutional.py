# -*- coding: utf-8 -*-
"""股东/机构面数据插件：CNE curated `institutional_holdings`（机构持仓，季频）。

数据口径：东财按季度汇总的机构持股（基金/QFII/社保/保险/券商/信托/银行/法人等），
行粒度 = (symbol, holder_type, report_period)。``holding_shares`` 实为**机构家数**，
``holding_mv`` 为持股市值（元），``holding_ratio`` 为持股比例（%，如 37.0 = 37%）。

PIT 锚点：源表只有 ``report_period``（如 2024Q1）没有公告日。股东信息随公司定期
报告公开，故与 curated ``earnings_disclosure_schedule`` 按报告期对齐，取该期
``actual_date``（定期报告实际披露日）作为可用时点，禁止按季末/期末前视。

面板列（``inst_*``）为日频阶跃：披露日后取最近一期值直到下一期披露。
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_ALL_INST_COLS = ["inst_count", "inst_ratio", "inst_mv", "inst_days_since"]

PLUGIN = DataSourcePlugin(
    name="institutional",
    dataset="institutional_holdings",
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map={c: c for c in _ALL_INST_COLS},
    priority=34,
)


def _disclosure_actual_by_period() -> pl.DataFrame:
    """disclosure schedule → (symbol, report_period, actual_date) 唯一映射。"""
    sched = _pitlib.read_curated("earnings_disclosure_schedule").select(
        ["symbol", "report_period", "actual_date"]
    )
    return (
        sched.filter(pl.col("actual_date").is_not_null())
        .sort(["symbol", "report_period", "actual_date"])
        .unique(subset=["symbol", "report_period"], keep="last")
    )


def _aggregate_per_period(raw: pl.DataFrame) -> pl.DataFrame:
    """(symbol, report_period) → 汇总家数/比例/市值。

    优先用源的 ``summary`` 行（东财直接给全类型汇总）；个别期缺 summary 时
    由各机构类型行累加兜底。
    """
    summary = (
        raw.filter(pl.col("holder_type") == "summary")
        .sort(["symbol", "report_period"])
        .unique(subset=["symbol", "report_period"], keep="last")
        .select(
            [
                pl.col("symbol"),
                pl.col("report_period"),
                pl.col("holding_shares").alias("s_count"),
                pl.col("holding_ratio").alias("s_ratio"),
                pl.col("holding_mv").alias("s_mv"),
            ]
        )
    )
    types = (
        raw.filter(pl.col("holder_type") != "summary")
        .group_by(["symbol", "report_period"])
        .agg(
            pl.col("holding_shares").sum().alias("t_count"),
            pl.col("holding_ratio").sum().alias("t_ratio"),
            pl.col("holding_mv").sum().alias("t_mv"),
        )
    )
    merged = summary.join(types, on=["symbol", "report_period"], how="full", coalesce=True)
    return merged.with_columns(
        pl.coalesce(["s_count", "t_count"]).alias("inst_count"),
        pl.coalesce(["s_ratio", "t_ratio"]).alias("inst_ratio"),
        pl.coalesce(["s_mv", "t_mv"]).alias("inst_mv"),
    ).select(["symbol", "report_period", "inst_count", "inst_ratio", "inst_mv"])


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """加载机构持仓并按定期报告实际披露日 PIT 展开为日频阶跃。"""
    raw = _pitlib.read_curated("institutional_holdings").select(
        ["symbol", "holder_type", "report_period", "holding_shares",
         "holding_ratio", "holding_mv"]
    )
    if raw.is_empty():
        raise ValueError("institutional: 无数据")
    agg = _aggregate_per_period(raw)
    actual = _disclosure_actual_by_period()
    events = agg.join(actual, on=["symbol", "report_period"], how="left")
    events = events.rename({"actual_date": "_pit_date"})
    events = events.filter(pl.col("_pit_date").is_not_null())
    if events.is_empty():
        raise ValueError("institutional: 无已披露的持仓期")

    result = _pitlib.expand_pit_daily(
        events,
        start=start,
        end=end,
        anchor="_pit_date",
        since_col="inst_days_since",
        value_cols=_ALL_INST_COLS,
    )
    pdf = result.to_pandas()
    logger.info("institutional adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
