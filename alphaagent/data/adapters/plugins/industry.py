# -*- coding: utf-8 -*-
"""行业分类插件：CNE curated ``industry_members``（申万口径，PIT 快照）。

申万 6 位行业码是**前缀层级**（`240301` 铝 → `2403` 工业金属 → `24` 有色），
CNE 侧同样口径（见 `cnequity/derive/industry_index.py`）。本插件据此派生：

- ``industry_sw_l1``   一级行业离散码（6 位码前 2 位）；`CS_NEUTRALIZE(f, $industry_sw_l1)`
  即行业内去均值（离散组号，勿再套 `CS_BUCKET`）
- ``industry_sw_code`` 完整 6 位申万码（三级粒度分组键，如细分行业中性化）
- ``ind_asof_days``    距最近一期行业快照的自然日

PIT：只用 ``as_of_date <= 当日`` 的快照（join_asof backward），首次快照之前无值。
背景：提示词一直声明 ``$industry_sw_l1``，但面板从无供给 → 每 run 报"不可用字段"，
行业中性化（研报 A 股因子标配）此前完全无法落地。
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

PLUGIN = DataSourcePlugin(
    name="industry",
    dataset="industry_members",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={
        "industry_sw_l1": "industry_sw_l1",
        "industry_sw_code": "industry_sw_code",
        "ind_asof_days": "ind_asof_days",
    },
    priority=60,
)


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """按快照日 PIT 展开申万行业码为日频阶跃。"""
    raw = _pitlib.read_curated_pit("industry_members", start=start, end=end)
    needed = {"symbol", "classification_system", "industry_code", "as_of_date"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"industry_members 缺列: {sorted(missing)}")

    mem = (
        raw.filter(
            (pl.col("classification_system") == "sw")
            & pl.col("industry_code").is_not_null()
            & pl.col("as_of_date").is_not_null()
        )
        .select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("as_of_date").cast(pl.Date).alias("_pit_date"),
            pl.col("industry_code").cast(pl.Utf8),
        )
        .unique(subset=["symbol", "_pit_date"], keep="last")
        .with_columns(
            pl.col("industry_code").str.slice(0, 2).cast(pl.Int32, strict=False).cast(pl.Float64).alias("industry_sw_l1"),
            pl.col("industry_code").str.strip_chars_start("0").cast(pl.Int32, strict=False).cast(pl.Float64).alias("industry_sw_code"),
        )
        .select(["symbol", "_pit_date", "industry_sw_l1", "industry_sw_code"])
    )
    if mem.is_empty():
        raise ValueError(f"industry: 窗口 {start}~{end} 内无申万行业快照")

    out = _pitlib.expand_pit_daily(
        mem,
        start=start,
        end=end,
        anchor="_pit_date",
        since_col="ind_asof_days",
        value_cols=["industry_sw_l1", "industry_sw_code"],
        out_symbol="symbol",
    )
    pdf = out.to_pandas()
    logger.info("industry adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, start, end)
    return pdf
