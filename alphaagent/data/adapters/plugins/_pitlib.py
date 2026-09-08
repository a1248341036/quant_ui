# -*- coding: utf-8 -*-
"""第一批数据插件共享工具：curated 读取 + 工作日网格 + PIT 展开基元。

供 plugins/ 下新增的季频/事件类插件复用（margin 等日频插件不需要网格）。
约定与 fundamental / forecast / shareholder_counts 插件一致：
- 事件表按 (symbol, PIT 锚点日) 聚合为 step 事件
- 用 symbol × 工作日 网格做 join_asof(backward) 广播为日频阶跃序列
- 首次事件之前的行整体丢弃（输出行即"已有披露"的股票与日期）
"""

from __future__ import annotations

import datetime
import logging
from typing import Iterable

import polars as pl

from alphaagent.data.adapters.plugins.fundamental import _curated_root

logger = logging.getLogger(__name__)

DEFAULT_START = datetime.date(2015, 1, 1)

# PIT 状态回看窗口：窗口起点前的该天数内的事件会保留，作为网格首日的初始状态。
PIT_LOOKBACK_DAYS = 800


def parse_window(
    start: str | None,
    end: str | None,
) -> tuple[datetime.date, datetime.date]:
    """把字符串日期窗口解析为 (start, end)，缺省补默认值。"""
    s = datetime.date.fromisoformat(start) if start else DEFAULT_START
    e = datetime.date.fromisoformat(end) if end else datetime.date.today()
    if s > e:
        raise ValueError(f"非法窗口: start={s} 晚于 end={e}")
    return s, e


def read_curated(dataset: str) -> pl.DataFrame:
    """读取整个 curated 数据集（与 fundamental 同口径，hive_partitioning=False）。"""
    root = _curated_root() / dataset
    return pl.read_parquet(root, hive_partitioning=False)


def weekdays_between(s: datetime.date, e: datetime.date) -> list[datetime.date]:
    days: list[datetime.date] = []
    d = s
    while d <= e:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def weekday_grid(
    symbols: Iterable[str],
    s: datetime.date,
    e: datetime.date,
) -> pl.DataFrame:
    """symbol × 工作日 笛卡尔网格（按 symbol, date 排序）。"""
    days = weekdays_between(s, e)
    syms = sorted(set(str(x) for x in symbols))
    return (
        pl.DataFrame({"date": pl.Series(days, dtype=pl.Date)})
        .join(pl.DataFrame({"symbol": pl.Series(syms, dtype=pl.Utf8)}), how="cross")
        .sort(["symbol", "date"])
    )


def expand_pit_daily(
    events: pl.DataFrame,
    *,
    start: str | None,
    end: str | None,
    anchor: str,
    since_col: str,
    value_cols: list[str],
    out_date: str = "trade_date",
    out_symbol: str = "ts_code",
) -> pl.DataFrame:
    """事件表 → 日频阶跃输出（网格 join_asof backward + days_since）。

    events 需已聚合到 (symbol, anchor) 粒度，含 ``value_cols``；
    输出日期范围 = [start, end]，首事件前的行丢弃。
    """
    s, e = parse_window(start, end)
    lo = s - datetime.timedelta(days=PIT_LOOKBACK_DAYS)
    events = events.filter(
        pl.col(anchor).is_not_null()
        & (pl.col(anchor) >= lo)
        & (pl.col(anchor) <= e)
    )
    if events.is_empty():
        raise ValueError("无有效事件记录")

    symbols = events["symbol"].unique().to_list()
    grid = weekday_grid(symbols, s, e)
    expanded = (
        grid.join_asof(
            events.sort(["symbol", anchor]),
            left_on="date",
            right_on=anchor,
            by="symbol",
            strategy="backward",
        )
        .with_columns(
            (pl.col("date") - pl.col(anchor))
            .dt.total_days()
            .cast(pl.Float64)
            .alias(since_col)
        )
        .filter(pl.col(since_col).is_not_null())
    )
    keep = list(value_cols)
    if since_col not in keep:
        keep.append(since_col)
    return expanded.select(
        [pl.col("date").alias(out_date), pl.col("symbol").alias(out_symbol), *keep]
    )
