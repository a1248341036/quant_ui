# -*- coding: utf-8 -*-
"""事件/披露日历数据插件：CNE curated `earnings_disclosure_schedule`。

预约披露时间表（镜像交易所披露日历）带三个日期：``first_scheduled_date``
（首次预约）、``scheduled_date``（当前有效预约）、``actual_date``（实际披露，
披露后回填）。源表是"现值语义"，预约变更覆盖 scheduled_date——没有记录每次
变更的发布时间戳，因此**不能**构造"距预约披露日还有 N 天"这类事前信号
（那需要知道时间表发布于哪天，硬算会把未来预约信息提前泄漏）。

本插件只输出实际披露后才成立的事后安全特征（以 actual_date 为 PIT 锚点）：
- ds_delay_days     最近一次实际披露相对其**最终预约日**推迟的天数（正=推迟，
  负=提前；披露当天起可见并保持到下一次披露）
- ds_delay_vs_first 相对**首次预约日**的推迟天数（含中途改约的累计信息）
- ds_days_since_actual 距最近一次实际披露的自然日

用法注意：与 holder_*/pred_* 同构的"距上次事件"序列，勿当作连续日频变量做
短窗差分；真正的事前披露提醒口径（预约日临近/逾期未披露）需 CNE 侧补充时间表
发布事件后再以插件 v2 提供。
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_ALL_DS_COLS = ["ds_delay_days", "ds_delay_vs_first", "ds_days_since_actual"]

PLUGIN = DataSourcePlugin(
    name="disclosure",
    dataset="earnings_disclosure_schedule",
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map={c: c for c in _ALL_DS_COLS},
    priority=37,
)


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """加载披露日历并按实际披露日 PIT 展开为日频阶跃。"""
    raw = _pitlib.read_curated("earnings_disclosure_schedule").select(
        ["symbol", "report_period", "scheduled_date",
         "first_scheduled_date", "actual_date"]
    )
    if raw.is_empty():
        raise ValueError("disclosure: 无数据")

    events = (
        raw.filter(pl.col("actual_date").is_not_null())
        .with_columns(
            (pl.col("actual_date") - pl.col("scheduled_date"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("ds_delay_days"),
            (pl.col("actual_date") - pl.col("first_scheduled_date"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("ds_delay_vs_first"),
        )
        # 同日多条（不同报告期同日披露）取推迟最大的一条作当日状态
        .sort(["symbol", "actual_date", "report_period"])
        .group_by(["symbol", "actual_date"])
        .agg([pl.col(c).last() for c in _ALL_DS_COLS if c != "ds_days_since_actual"])
        .rename({"actual_date": "_pit_date"})
        .filter(pl.col("_pit_date").is_not_null())
    )
    if events.is_empty():
        raise ValueError("disclosure: 无已披露记录")

    result = _pitlib.expand_pit_daily(
        events,
        start=start,
        end=end,
        anchor="_pit_date",
        since_col="ds_days_since_actual",
        value_cols=_ALL_DS_COLS,
    )
    pdf = result.to_pandas()
    logger.info("disclosure adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
