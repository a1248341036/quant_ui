# -*- coding: utf-8 -*-
"""监管事件插件：CNE curated ``regulatory_events``（立案/警示函/处罚/监管函/纪律处分）。

数据：东财口径监管事件（``symbol``/``event_date``/``event_type``/``title``），按事件日
PIT 展开为日频稀疏列。事件类型五类：investigation（立案）/ warning_letter（警示函）/
regulatory_letter（监管函）/ disciplinary（纪律处分）/ penalty（行政处罚）。

产出列：
- ``reg_cnt_20d``：近 20 个**交易日**监管事件条数（累计和差分）
- ``reg_days_since``：距最近一次监管事件的自然日
- ``reg_flag_*``：近 20 交易日是否出现过该类事件（1/0，五类各一列）
- ``reg_severity``：近 360 自然日窗口内出现过的最高事件严重度档
  （investigation=5 > disciplinary=4 > penalty=3 > regulatory_letter=2 >
  warning_letter=1，无事件=0）

内存设计：事件日志按 (symbol, event_date) 聚合后只保留有事件的 symbol 的日频网格，
输出为稀疏日频薄列并与主表 join，原始长表不落全量面板。
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 400  # 覆盖 20 交易日窗口 + severity 360 自然日 + 播种

# 事件类型 → 严重度分档（越高越严重）
_SEVERITY = {
    "warning_letter": 1,
    "regulatory_letter": 2,
    "penalty": 3,
    "disciplinary": 4,
    "investigation": 5,
}
_TYPES = tuple(_SEVERITY)

PLUGIN = DataSourcePlugin(
    name="regulatory_events",
    dataset="regulatory_events",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={
        "reg_cnt_20d": "reg_cnt_20d",
        "reg_days_since": "reg_days_since",
        "reg_severity": "reg_severity",
        **{f"reg_flag_{t}": f"reg_flag_{t}" for t in _TYPES},
    },
    priority=53,
)

_CNT_WINDOW = 20
_SEV_WINDOW_DAYS = 360


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """监管事件 → 日频事件密度/严重度/类型打标面板列。"""
    s, e = _pitlib.parse_window(start, end)
    ext = s - datetime.timedelta(days=_LOOKBACK_DAYS)
    raw = _pitlib.read_curated("regulatory_events", start=ext.isoformat(), end=e.isoformat())
    needed = {"symbol", "event_date", "event_type"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"regulatory_events 缺列: {sorted(missing)}")

    # 事件 → (symbol, 事件日) 聚合：条数 / 分类型命中 / 当日最高严重度
    sev_expr = (
        pl.when(pl.col("event_type").is_in(_SEVERITY))
        .then(pl.col("event_type").cast(pl.Utf8).replace_strict(_SEVERITY))
        .otherwise(0)
        .cast(pl.Int8)
    )
    events = (
        raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("event_date").cast(pl.Date),
            sev_expr.alias("_sev"),
        )
        .group_by(["symbol", "event_date"])
        .agg(
            pl.len().alias("reg_cnt_day"),
            pl.col("_sev").max().alias("_day_sev"),
            *[
                (pl.col("_sev") == _SEVERITY[t]).sum().cast(pl.Int8).alias(f"reg_flag_{t}")
                for t in _TYPES
            ],
        )
        .filter((pl.col("event_date") >= ext) & (pl.col("event_date") <= e))
        # 必须显式排序：group_by().agg() 的输出顺序不确定，而下面第 155 行的
        # cum_max().over("symbol") 是**按行序**累积的。不排序时累积顺序随分组数
        # （即随 end）变化，同一交易日的 reg_severity 会因请求右端不同而不同
        # ——2026-11-22 实测：end=10-15 与 end=10-29 两次构建，2026-09-01 的
        # 000567.SZ 得到 4 与 3。排序后与 end 无关。
        .sort(["symbol", "event_date"])
    )
    if events.is_empty():
        raise ValueError(f"regulatory_events: 窗口 {s}~{e} 内无监管事件")

    days = _pitlib.weekdays_between(ext, e)
    syms = sorted(set(events["symbol"].to_list()))
    dense = (
        pl.DataFrame({"trade_date": pl.Series(days, dtype=pl.Date)})
        .join(pl.DataFrame({"symbol": pl.Series(syms, dtype=pl.Utf8)}), how="cross")
        .join(
            events,
            left_on=["symbol", "trade_date"],
            right_on=["symbol", "event_date"],
            how="left",
        )
        .sort(["symbol", "trade_date"])
    )

    flag_cols = [f"reg_flag_{t}" for t in _TYPES]
    dense = dense.with_columns(
        pl.col("reg_cnt_day").fill_null(0).alias("reg_cnt_day"),
        *[pl.col(c).fill_null(0).cast(pl.Int8).alias(c) for c in flag_cols],
        pl.col("_day_sev").fill_null(0).cast(pl.Int8).alias("_day_sev"),
    )
    # 滚动窗口前先播种回看段，保证窗口头部的窗口计数正确（累计和 - 差分法）
    dense = dense.with_columns(
        pl.col("reg_cnt_day").cum_sum().over("symbol").alias("_cum_cnt"),
        *[pl.col(c).cum_sum().over("symbol").alias(f"_cum_{c}") for c in flag_cols],
    )
    dense = dense.with_columns(
        (
            pl.col("_cum_cnt") - pl.col("_cum_cnt").shift(_CNT_WINDOW).over("symbol").fill_null(0)
        )
        .cast(pl.Float64)
        .alias("reg_cnt_20d"),
        *[
            (
                pl.col(f"_cum_{c}")
                - pl.col(f"_cum_{c}").shift(_CNT_WINDOW).over("symbol").fill_null(0)
            )
            .gt(0)
            .cast(pl.Int8)
            .alias(c)
            for c in flag_cols
        ],
    )

    # 最高严重度累积序列：事件日 → 截至该日的最高严重度，再 join_asof backward 阶梯化。
    # 注意 1：events 已在上面按 (symbol, event_date) 排序，cum_max 因此有确定语义——
    #         「截至该事件日的全历史最高严重度」。排序缺失时结果依赖行序（见上方注释）。
    # 注意 2：`_SEV_WINDOW_DAYS = 360` 是**死常量**（全文件未引用），当初注释写的
    #         "近 360 自然日窗口内最高严重度" 并未实现，实际是全历史 max。改成真滚动
    #         窗口会改变历史值（口径变化），需要单独决策，故此处只修确定性、不动语义。
    sev_cum = (
        events.select(["symbol", "event_date", "_day_sev"])
        .with_columns(pl.col("_day_sev").cum_max().over("symbol").alias("_sev_cum_max"))
        .select(["symbol", "event_date", "_sev_cum_max"])
    )
    dense = dense.join_asof(
        sev_cum.sort(["symbol", "event_date"]),
        left_on="trade_date",
        right_on="event_date",
        by="symbol",
        strategy="backward",
    ).with_columns(
        pl.col("_sev_cum_max").fill_null(0).cast(pl.Int8).alias("reg_severity")
    ).drop("_sev_cum_max")

    # 距最近一次监管事件的自然日
    last = events.select(["symbol", pl.col("event_date").alias("_ev_date")]).unique()
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
        .alias("reg_days_since")
    ).drop("_ev_date")

    keep = ["trade_date", "symbol", "reg_cnt_20d", "reg_severity", "reg_days_since", *flag_cols]
    # 首监管事件前的行不产出（PIT：事件不可知；与 share_unlock 同口径）
    pdf = (
        dense.filter(
            (pl.col("trade_date") >= s) & pl.col("reg_days_since").is_not_null()
        )
        .select(keep)
        .to_pandas()
    )
    logger.info("regulatory_events adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, s, e)
    return pdf