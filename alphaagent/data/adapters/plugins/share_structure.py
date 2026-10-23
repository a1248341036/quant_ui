# -*- coding: utf-8 -*-
"""股本结构变化插件：CNE curated ``share_structure``（股本变动事件）。

数据：东财口径股本变动（``symbol``/``change_date``/``total_shares``/``float_shares``/
``restricted_shares``/``free_float_shares``/``change_reason``/``announce_date``），按
``change_date`` 解释为**股本结构变化事件**，日频播报。

与 share_unlock（限售解禁计划）互补：本数据集按 change_reason 区分股本事件类型，
并提供「流通股占总股本比例」这种连续状态，而 share_unlock 只给解禁比例。

产出列：
- ``ss_cnt_20d``：近 20 个**交易日**股本变化事件条数
- ``ss_days_since``：距最近一次股本变化事件的自然日
- ``ss_float_ratio``：最近一次股本事件后的流通股占总股本比例（float_shares/total_shares，
  total_shares<=0 时 NaN）
- ``ss_flag_ipo`` / ``ss_flag_plac`` / ``ss_flag_stock`` / ``ss_flag_other``：
  最近一次股本事件分属 首发限售 / 增发配售 / 转增拆股 / 其他（1/0）

内存设计：事件按 (symbol, change_date) 聚合，日频网格只保留有事件的 symbol，稀疏薄列，
原始长表不落全量面板。
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 60

# change_reason → 类别关键词（小写包含匹配，事件命中首个类别即归其类）
_FLAG_RULES: dict[str, tuple[str, ...]] = {
    "ss_flag_ipo": ("首发", "新三板上市", "上市前股本", "股改限售"),
    "ss_flag_plac": ("增发", "配售", "配股", "自主行权"),
    "ss_flag_stock": ("转增", "拆股"),
    "ss_flag_other": ("股权激励", "高管", "回购", "网下", "其他"),
}
_FLAG_ORDER = list(_FLAG_RULES)

PLUGIN = DataSourcePlugin(
    name="share_structure",
    dataset="share_structure",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={
        "ss_cnt_20d": "ss_cnt_20d",
        "ss_days_since": "ss_days_since",
        "ss_float_ratio": "ss_float_ratio",
        **{col: col for col in _FLAG_ORDER},
    },
    priority=54,
)

_CNT_WINDOW = 20


def _events(raw: pl.DataFrame) -> pl.DataFrame:
    """股本变化 → 每 (symbol, change_date)：条数 / 流通比例 / 首个命中类别。"""
    lower = pl.col("change_reason").cast(pl.Utf8).fill_null("").str.to_lowercase()
    chained: pl.Expr = pl.lit("")
    for col, keys in _FLAG_RULES.items():
        chained = pl.when((chained.eq("")) & lower.str.contains("|".join(keys))).then(pl.lit(col)).otherwise(chained)

    ev = (
        raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("change_date").cast(pl.Date),
            pl.col("float_shares").cast(pl.Float64),
            pl.col("total_shares").cast(pl.Float64),
        )
        .filter(pl.col("change_date").is_not_null())
        .group_by(["symbol", "change_date"])
        .agg(
            pl.len().alias("ss_cnt"),
            pl.col("float_shares").sum().alias("_fs"),
            pl.col("total_shares").sum().alias("_ts"),
        )
        .with_columns(
            pl.when(pl.col("_ts") > 0)
            .then(pl.col("_fs") / pl.col("_ts"))
            .otherwise(None)
            .alias("ss_float_ratio")
        )
    )
    # 类别基于同一 (symbol, change_date) 的原文——回填第一条 reason 后分类
    ev = ev.join(
        (
            raw.select(
                pl.col("symbol").cast(pl.Utf8),
                pl.col("change_date").cast(pl.Date),
                lower,  # 命名 change_reason_lower
            )
            .group_by(["symbol", "change_date"])
            .agg(pl.col("change_reason").first().alias("_reason"))
        ),
        on=["symbol", "change_date"],
        how="left",
    ).with_columns(pl.col("_reason").fill_null("").alias("_reason2"))
    rl = pl.col("_reason2").str.to_lowercase()
    flag_chained: pl.Expr = pl.lit("")
    for col, keys in _FLAG_RULES.items():
        flag_chained = pl.when((flag_chained.eq("")) & rl.str.contains("|".join(keys))).then(pl.lit(col)).otherwise(flag_chained)
    ev = ev.with_columns(
        *[(flag_chained.eq(col)).cast(pl.Int8).alias(col) for col in _FLAG_ORDER]
    )
    return ev.drop(["_fs", "_ts", "_reason", "_reason2"])


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """股本变化 → 日频事件播报/流通比例面板列。"""
    s, e = _pitlib.parse_window(start, end)
    ext = s - datetime.timedelta(days=_LOOKBACK_DAYS)
    raw = _pitlib.read_curated("share_structure", start=ext.isoformat(), end=e.isoformat())
    needed = {"symbol", "change_date", "float_shares", "total_shares", "change_reason"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"share_structure 缺列: {sorted(missing)}")

    events = _events(raw).filter(
        (pl.col("change_date") >= ext) & (pl.col("change_date") <= e)
    )
    if events.is_empty():
        raise ValueError(f"share_structure: 窗口 {s}~{e} 内无股本变化事件")

    days = _pitlib.weekdays_between(ext, e)
    syms = sorted(set(events["symbol"].to_list()))
    dense = (
        pl.DataFrame({"trade_date": pl.Series(days, dtype=pl.Date)})
        .join(pl.DataFrame({"symbol": pl.Series(syms, dtype=pl.Utf8)}), how="cross")
        .join(
            events,
            left_on=["symbol", "trade_date"],
            right_on=["symbol", "change_date"],
            how="left",
        )
        .sort(["symbol", "trade_date"])
    )

    flag_cols = list(_FLAG_ORDER)
    dense = dense.with_columns(
        pl.col("ss_cnt").fill_null(0).cast(pl.Int32).alias("ss_cnt"),
        pl.col("ss_cnt").fill_null(0).cum_sum().over("symbol").alias("_cum_cnt"),
    )
    dense = dense.with_columns(
        (pl.col("_cum_cnt") - pl.col("_cum_cnt").shift(_CNT_WINDOW).over("symbol").fill_null(0))
        .cast(pl.Float64)
        .alias("ss_cnt_20d")
    )

    # 最近一次股本事件后的流通比例（backward 播报）
    state = events.select(["symbol", pl.col("change_date").alias("_ev_date"), "ss_float_ratio"])
    dense = dense.join_asof(
        state.sort(["symbol", "_ev_date"]),
        left_on="trade_date",
        right_on="_ev_date",
        by="symbol",
        strategy="backward",
    ).drop("_ev_date")

    # 距最近一次股本事件的自然日
    last = events.select(["symbol", pl.col("change_date").alias("_ev_date")]).unique()
    dense = dense.join_asof(
        last.sort(["symbol", "_ev_date"]),
        left_on="trade_date",
        right_on="_ev_date",
        by="symbol",
        strategy="backward",
    ).with_columns(
        (pl.col("trade_date") - pl.col("_ev_date"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("ss_days_since")
    ).drop("_ev_date")

    # 标志列透传到日频（backward 播报，未到首个事件前为空）
    dense = dense.join_asof(
        events.select(["symbol", pl.col("change_date").alias("_ev_date"), *flag_cols]),
        left_on="trade_date",
        right_on="_ev_date",
        by="symbol",
        strategy="backward",
    ).with_columns(
        *[pl.col(c).fill_null(0).cast(pl.Int8).alias(c) for c in flag_cols]
    ).drop("_ev_date")

    keep = ["trade_date", "symbol", "ss_cnt_20d", "ss_days_since", "ss_float_ratio", *flag_cols]
    pdf = dense.filter(pl.col("trade_date") >= s).select(keep).to_pandas()
    logger.info("share_structure adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, s, e)
    return pdf