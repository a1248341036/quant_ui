# -*- coding: utf-8 -*-
"""公告事件插件：CNE curated ``announcement_index``（个股公告索引，PIT 日频）。

数据：cninfo/东财口径公告索引（``announcement_id``/``symbol``/``title``/``announce_date``/
``category``），按 **公告日** PIT 展开为日频阶跃——当日及之前已公告的才可见。

产出列：
- ``ann_cnt_5d`` / ``ann_cnt_20d``：近 5 / 20 个**交易日**（工作日网格）公告条数
- ``ann_days_since``：距最近一条公告的自然日
- ``ann_flag_*``：窗口内是否出现过该类公告（关键词命中 title+category，1/0）

关键词表见 ``_FLAG_PATTERNS``（可调）；计数与打标都在日频网格上用"累计和差分"
实现（`cum_sum - cum_sum.shift(N)`，按 symbol 分组），避免 polars 时间窗 API 差异。

背景：研报课题 96 次点名 announcement 数据面，而挖掘面板此前没有任何公告列。
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

# 事件打标关键词（对 title + category 做小写包含匹配）；命中任一即置 1
_FLAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "ann_flag_reduce": ("减持", "减仓", "清仓"),
    "ann_flag_increase": ("增持", "增仓", "举牌"),
    "ann_flag_buyback": ("回购",),
    "ann_flag_pledge": ("质押", "冻结"),
    "ann_flag_lawsuit": ("诉讼", "仲裁", "处罚", "立案", "违规"),
    "ann_flag_inquiry": ("问询", "关注函", "监管函", "警示函"),
    "ann_flag_forecast": ("业绩预告", "预增", "预减", "预盈", "预亏", "业绩快报"),
}

PLUGIN = DataSourcePlugin(
    name="announcement",
    dataset="announcement_index",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={
        "ann_cnt_5d": "ann_cnt_5d",
        "ann_cnt_20d": "ann_cnt_20d",
        "ann_days_since": "ann_days_since",
        **{col: col for col in _FLAG_PATTERNS},
    },
    priority=61,
)

_CNT_WINDOWS = ((5, "ann_cnt_5d"), (20, "ann_cnt_20d"))
_FLAG_WINDOW = 20


def _events(raw: pl.DataFrame) -> pl.DataFrame:
    """公告索引 → 每 (symbol, 公告日) 的条数 + 打标（0/1）。"""
    text = (
        pl.col("title").cast(pl.Utf8).fill_null("").str.to_lowercase()
        + pl.lit(" ")
        + pl.col("category").cast(pl.Utf8).fill_null("").str.to_lowercase()
    )
    expr = [pl.len().alias("ann_cnt")]
    for col, keys in _FLAG_PATTERNS.items():
        pat = "|".join(keys)
        expr.append(text.str.contains(pat).any().cast(pl.Int8).alias(col))
    return (
        raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("announce_date").cast(pl.Date).alias("trade_date"),
            pl.col("title").cast(pl.Utf8),
            pl.col("category").cast(pl.Utf8),
        )
        .group_by(["symbol", "trade_date"])
        .agg(expr)
    )


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """公告索引 → 日频公告密度/打标面板列。"""
    raw = _pitlib.read_curated("announcement_index", start=start, end=end)
    needed = {"symbol", "announce_date", "title", "category"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"announcement_index 缺列: {sorted(missing)}")

    spec, ee = _pitlib.parse_window(start, end)
    events = _events(raw).filter(
        (pl.col("trade_date") >= spec) & (pl.col("trade_date") <= ee)
    )
    if events.is_empty():
        raise ValueError(f"announcement: 窗口 {spec}~{ee} 内无公告")

    days = _pitlib.weekdays_between(spec, ee)
    syms = sorted(set(events["symbol"].to_list()))
    dense = (
        pl.DataFrame({"trade_date": pl.Series(days, dtype=pl.Date)})
        .join(pl.DataFrame({"symbol": pl.Series(syms, dtype=pl.Utf8)}), how="cross")
        .join(events, on=["symbol", "trade_date"], how="left")
        .with_columns(pl.col("ann_cnt").fill_null(0))
        .sort(["symbol", "trade_date"])
    )

    # 计数：累计和差分（窗口=行数，即交易日数）；打标：窗口内累计 >0
    dense = dense.with_columns(
        pl.col("ann_cnt").cum_sum().over("symbol").alias("_cum"),
        *[
            pl.col(col).fill_null(0).cum_sum().over("symbol").alias(f"_cum_{col}")
            for col in _FLAG_PATTERNS
        ],
    )
    count_exprs = [
        (pl.col("_cum") - pl.col("_cum").shift(n).over("symbol").fill_null(0))
        .cast(pl.Float64)
        .alias(out)
        for n, out in _CNT_WINDOWS
    ]
    flag_exprs = [
        (
            pl.col(f"_cum_{col}") - pl.col(f"_cum_{col}").shift(_FLAG_WINDOW).over("symbol").fill_null(0)
        )
        .gt(0)
        .cast(pl.Int8)
        .alias(col)
        for col in _FLAG_PATTERNS
    ]
    dense = dense.with_columns(count_exprs + flag_exprs)

    # 距最近一条公告的自然日
    last_ann = events.select(["symbol", pl.col("trade_date").alias("_ann_date")]).unique()
    dense = dense.join_asof(
        last_ann.sort(["symbol", "_ann_date"]),
        left_on="trade_date",
        right_on="_ann_date",
        by="symbol",
        strategy="backward",
    ).with_columns(
        (pl.col("trade_date") - pl.col("_ann_date")).dt.total_days().cast(pl.Float64).alias("ann_days_since")
    )

    keep = ["trade_date", "symbol", *[out for _, out in _CNT_WINDOWS], *_FLAG_PATTERNS, "ann_days_since"]
    pdf = dense.select(keep).to_pandas()
    logger.info("announcement adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, spec, ee)
    return pdf
