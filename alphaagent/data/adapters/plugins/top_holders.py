# -*- coding: utf-8 -*-
"""股东/筹码面数据插件：CNE curated `top_holders`（十大股东/流通股东，季频）。

源口径（见 CNE docs）：一张表两个口径，``holder_scope=float`` = 前十大**流通**
股东（占流通股比例），``total`` = 前十大股东（占总股本比例），两者分母不同不可
混算。筹码集中度因子应使用 ``float`` 口径；本插件只产出 float 口径聚合。

PIT 锚点 = 每行 ``announce_date``（报告公告日），按 (symbol, announce_date)
聚合为 step 事件后做日频 asof 展开——公告日前不可见。

面板列（``th_*``，float 口径）：
- th_top1_pct    第一大流通股东持股占比（%）
- th_top10_pct   前十大流通股东持股占比合计（%）
- th_inst_pct    流通股东中机构持股占比合计（%）
- th_inst_count  流通股东中的机构家数
- th_days_since  距最近一次股东披露的自然日
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_ALL_TH_COLS = ["th_top1_pct", "th_top10_pct", "th_inst_pct", "th_inst_count", "th_days_since"]

PLUGIN = DataSourcePlugin(
    name="top_holders",
    dataset="top_holders",
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map={c: c for c in _ALL_TH_COLS},
    priority=35,
)


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """加载十大流通股东并按公告日 PIT 展开为日频阶跃。"""
    raw = _pitlib.read_curated("top_holders").select(
        ["symbol", "record_date", "holder_scope", "holder_rank",
         "holding_pct", "is_institution", "announce_date"]
    )
    if raw.is_empty():
        raise ValueError("top_holders: 无数据")

    float_rows = raw.filter(
        (pl.col("holder_scope") == "float")
        & pl.col("announce_date").is_not_null()
        & pl.col("holding_pct").is_not_null()
    )
    if float_rows.is_empty():
        raise ValueError("top_holders: float 口径无数据")

    # 同一 (symbol, announce_date) 可能存在多张报表快照（纠正性披露），
    # 只保留 record_date 最新的一张，避免同公告日快照被重复合计。
    float_rows = float_rows.filter(
        pl.col("record_date")
        == pl.col("record_date").max().over(["symbol", "announce_date"])
    )

    per_period = float_rows.group_by(["symbol", "announce_date"]).agg(
        pl.col("holding_pct")
        .filter(pl.col("holder_rank") == 1)
        .max()
        .alias("th_top1_pct"),
        pl.col("holding_pct").sum().alias("th_top10_pct"),
        pl.col("holding_pct")
        .filter(pl.col("is_institution"))
        .sum()
        .alias("th_inst_pct"),
        pl.col("is_institution")
        .filter(pl.col("is_institution"))
        .count()
        .alias("th_inst_count"),
    )
    per_period = (
        per_period.rename({"announce_date": "_pit_date"})
        .filter(pl.col("_pit_date").is_not_null())
    )
    if per_period.is_empty():
        raise ValueError("top_holders: 聚合后无有效披露期")

    result = _pitlib.expand_pit_daily(
        per_period,
        start=start,
        end=end,
        anchor="_pit_date",
        since_col="th_days_since",
        value_cols=_ALL_TH_COLS,
    )
    pdf = result.to_pandas()
    logger.info("top_holders adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
