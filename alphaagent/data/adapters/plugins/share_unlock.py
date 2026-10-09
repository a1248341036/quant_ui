# -*- coding: utf-8 -*-
"""限售解禁插件：CNE curated ``share_unlock_schedule``（解禁事件，PIT 展开）。

面板列
------
| 面板列 | 含义 |
|---|---|
| ``unlock_days_since`` | 距最近一次**已发生**解禁的自然日天数（行筛选锚点） |
| ``unlock_ratio_last`` | 最近一次解禁占流通股比例（%） |
| ``unlock_ratio_past180`` | 过去 180 天内已解禁比例合计（%），无解禁为 0 |
| ``unlock_days_to_next`` | 距下一次**计划**解禁的天数 |
| ``unlock_ratio_next`` | 下一次计划解禁比例（%） |

PIT 口径
--------
- ``past*`` 三列**严格 PIT 安全**：只用到 ``unlock_date <= 当日`` 的事实；
- ``*_next`` 两列按**计划口径**给出（解禁时间表在实践中提前公开）。但 curated 源表
  **没有公告日**，无法严格证明信息可知时点，故这两列仅作参考；要求严格 PIT 时只取 past 组。
- 与其它 PIT 插件一致：**首次解禁之前的行不产出**（``unlock_days_since`` 为空即丢弃），
  因此"刚上市、尚无解禁"的股票不出现在本列上。
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_PAST_COLS = ["unlock_days_since", "unlock_ratio_last", "unlock_ratio_past180"]
_NEXT_COLS = ["unlock_days_to_next", "unlock_ratio_next"]
_ALL_UNLOCK_COLS = [*_PAST_COLS, *_NEXT_COLS]

PLUGIN = DataSourcePlugin(
    name="share_unlock",
    dataset="share_unlock_schedule",
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map={c: c for c in _ALL_UNLOCK_COLS},
    priority=36,
)


def _events(start: str | None, end: str | None) -> pl.DataFrame:
    """解禁计划 → (symbol, unlock_date) 聚合（同日多次解禁比例求和）。"""
    raw = _pitlib.read_curated("share_unlock_schedule", start=start, end=end)
    needed = {"symbol", "unlock_date", "unlock_ratio"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"share_unlock_schedule 缺列: {sorted(missing)}")
    ev = (
        raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("unlock_date").cast(pl.Date),
            pl.col("unlock_ratio").cast(pl.Float64, strict=False).alias("ratio"),
        )
        .filter(pl.col("unlock_date").is_not_null())
        .group_by(["symbol", "unlock_date"])
        .agg(pl.col("ratio").sum().alias("ratio"))
        .sort(["symbol", "unlock_date"])
    )
    if ev.is_empty():
        raise ValueError("share_unlock: 窗口内无解禁记录")
    return ev


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """解禁计划 → 日频：最近/下一次解禁与过去 180 天解禁压力。"""
    s, e = _pitlib.parse_window(start, end)
    ev = _events(start, end)
    symbols = ev["symbol"].unique().to_list()
    grid = _pitlib.weekday_grid(symbols, s, e)

    # ── 过去：join_asof(backward) 最近一次已解禁 ──
    past = grid.sort(["symbol", "date"]).join_asof(
        ev, left_on="date", right_on="unlock_date", by="symbol", strategy="backward"
    )
    past = past.with_columns(
        (pl.col("date") - pl.col("unlock_date")).dt.total_days().cast(pl.Float64).alias("unlock_days_since"),
        pl.col("ratio").alias("unlock_ratio_last"),
    )

    # ── 过去 180 天解禁压力：把事件按 symbol 展开到网格后窗口内求和 ──
    pairs = grid.join(ev, on="symbol", how="inner").filter(
        (pl.col("unlock_date") <= pl.col("date"))
        & (pl.col("unlock_date") > pl.col("date") - pl.duration(days=180))
    )
    past180 = pairs.group_by(["symbol", "date"]).agg(
        pl.col("ratio").sum().alias("unlock_ratio_past180")
    )
    past = past.join(past180, on=["symbol", "date"], how="left").with_columns(
        pl.col("unlock_ratio_past180").fill_null(0.0)
    )

    # ── 下一次计划解禁：join_asof(forward) ──
    nxt = grid.sort(["symbol", "date"]).join_asof(
        ev, left_on="date", right_on="unlock_date", by="symbol", strategy="forward"
    ).select(
        pl.col("symbol"),
        pl.col("date"),
        (pl.col("unlock_date") - pl.col("date")).dt.total_days().cast(pl.Float64).alias("unlock_days_to_next"),
        pl.col("ratio").alias("unlock_ratio_next"),
    )

    out = past.join(nxt, on=["symbol", "date"], how="left")
    # 首次解禁之前不产出（与其它 PIT 插件同口径）
    out = out.filter(pl.col("unlock_days_since").is_not_null())
    if out.is_empty():
        raise ValueError("share_unlock: 过滤后无数据（窗口内可能没有任何已发生的解禁）")

    out = out.select(
        [pl.col("date").alias("trade_date"), pl.col("symbol").alias("ts_code"), *_ALL_UNLOCK_COLS]
    )
    pdf = out.to_pandas()
    logger.info("share_unlock adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, start, end)
    return pdf
