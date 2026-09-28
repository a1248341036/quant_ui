# -*- coding: utf-8 -*-
"""指数成分插件：CNE curated ``index_constituents``（PIT 月快照）。

**先看现状再解读**：curated 里当前只有两个指数的成分快照（``399001.SZ`` 深证成指 500 只、
``399006.SZ`` 创业板指 100 只），且 ``weight`` 全为 0（源端未给权重）。因此本插件只产出
真实存在的东西，不臆造 HS300/ZZ500/ZZ1000 标记：

- ``idx_szcomp``：深证成指成分（1/0，PIT 快照）
- ``idx_chinext``：创业板指成分（1/0，PIT 快照）
- ``idx_in_index``：任一指数成分（1/0）
- ``idx_asof_days``：距最近一期成分快照的自然日

要补 HS300/ZZ500/ZZ1000，需要 CNE 侧先把这些指数加进 ``index_constituents`` 的采集范围。
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

# 指数代码 → 面板标记列（curated 实测只有这两个）
_INDEX_FLAGS: dict[str, str] = {
    "399001.SZ": "idx_szcomp",
    "399006.SZ": "idx_chinext",
}
_FLAG_COLS = ["idx_szcomp", "idx_chinext", "idx_in_index"]

PLUGIN = DataSourcePlugin(
    name="index_membership",
    dataset="index_constituents",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={**{c: c for c in _FLAG_COLS}, "idx_asof_days": "idx_asof_days"},
    priority=64,
)


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """指数成分 → 按快照日 PIT 展开的日频标记列。"""
    raw = _pitlib.read_curated("index_constituents", start=start, end=end)
    needed = {"symbol", "index_symbol", "as_of_date"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"index_constituents 缺列: {sorted(missing)}")

    known = pl.DataFrame({"index_symbol": list(_INDEX_FLAGS)})
    snap = (
        raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("index_symbol").cast(pl.Utf8),
            pl.col("as_of_date").cast(pl.Date).alias("_pit_date"),
        )
        .filter(pl.col("_pit_date").is_not_null())
        .join(known, on="index_symbol", how="semi")
    )
    # 逐标记列透视：某 (symbol, 快照日) 是否属于该指数
    out = snap.select("symbol", "_pit_date").unique()
    for index_symbol, col in _INDEX_FLAGS.items():
        members = (
            snap.filter(pl.col("index_symbol") == index_symbol)
            .select("symbol", "_pit_date")
            .unique()
            .with_columns(pl.lit(1.0).alias(col))
        )
        out = out.join(members, on=["symbol", "_pit_date"], how="left")
    if out.is_empty():
        raise ValueError(f"index_membership: 窗口 {start}~{end} 内无目标指数成分")
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in _INDEX_FLAGS.values()])
    out = out.with_columns(
        (
            pl.sum_horizontal([pl.col(c) for c in _INDEX_FLAGS.values()]) > 0
        ).cast(pl.Float64).alias("idx_in_index")
    )

    expanded = _pitlib.expand_pit_daily(
        out.select(["symbol", "_pit_date", *_FLAG_COLS]),
        start=start,
        end=end,
        anchor="_pit_date",
        since_col="idx_asof_days",
        value_cols=_FLAG_COLS,
        out_symbol="symbol",
    )
    pdf = expanded.to_pandas()
    logger.info("index_membership adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, start, end)
    return pdf
