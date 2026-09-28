# -*- coding: utf-8 -*-
"""研报视角插件：卖方一致预期（``analyst_consensus``）+ 个股研报预测（``report_rc``）。

两个 CNE curated 数据集合并成一组 PIT 面板列：

一致预期（按 ``forecast_date`` PIT，取最近预测年度）：
- ``ac_eps_fy`` / ``ac_pe_fy`` / ``ac_target_price``：一致预期 EPS / 对应 PE / 目标价（元）
- ``ac_rating``：评级分值（buy=2 / overweight=1 / neutral=0，其它 NaN）
- ``ac_analyst_count``：覆盖分析师家数
- ``ac_days_since``：距预期更新日的自然日

研报预测（按 ``report_date`` PIT，同 (symbol, report_date) 取均值）：
- ``rc_eps_y`` / ``rc_pe_y`` / ``rc_roe_y``：最近研报给出的 EPS / PE / ROE 预测
- ``rc_tp_wan``：预测利润总额（**万元**，非目标价）
- ``rc_target_price``：目标价（元）= (max_price+min_price)/2，源端多数为空
- ``rc_rating``：评级分值（买入/强推/强烈推荐/推荐=2；增持/跑赢行业/优于大市=1；其它 NaN）
- ``rc_cnt_90d`` / ``rc_days_since``：近 90 个交易日研报条数（拥挤度）/ 距最近研报的自然日

背景：研报类课题直接点名"一致预期/评级/覆盖度"，而挖掘面板此前一个研报列都没有。
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

_AC_VALUE_COLS = ["ac_eps_fy", "ac_pe_fy", "ac_target_price", "ac_rating", "ac_analyst_count"]
_RC_VALUE_COLS = ["rc_eps_y", "rc_pe_y", "rc_roe_y", "rc_tp_wan", "rc_target_price", "rc_rating"]
_ALL_COLS = [*_AC_VALUE_COLS, "ac_days_since", *_RC_VALUE_COLS, "rc_cnt_90d", "rc_days_since"]

PLUGIN = DataSourcePlugin(
    name="research_views",
    dataset="analyst_consensus",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={c: c for c in _ALL_COLS},
    priority=62,
)

# 评级 → 分值（东财英文口径 / 国内中文口径各一套）
_AC_RATING_SCORE = {"buy": 2.0, "overweight": 1.0, "neutral": 0.0}
_RC_RATING_SCORE = {
    "买入": 2.0, "强推": 2.0, "强烈推荐": 2.0, "推荐": 2.0,
    "增持": 1.0, "跑赢行业": 1.0, "优于大市": 1.0, "审慎增持": 1.0,
    "中性": 0.0, "持有": 0.0, "谨慎推荐": 0.0,
}
_RC_WINDOW = 90


def _rating_score(col: str, mapping: dict[str, float]) -> pl.Expr:
    return (
        pl.col(col).cast(pl.Utf8).fill_null("").str.strip_chars()
        .replace_strict(mapping, default=None, return_dtype=pl.Float64)
    )


def _ac_events(raw: pl.DataFrame) -> pl.DataFrame:
    """一致预期 → 每 (symbol, 预期日) 一行（取最近预测年度）。"""
    df = raw
    if "forecast_year" in df.columns:
        df = df.with_columns(pl.col("forecast_year").cast(pl.Utf8).fill_null(""))
        df = df.sort(["symbol", "forecast_date", "forecast_year"]).unique(
            subset=["symbol", "forecast_date"], keep="last"
        )
    return (
        df.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("forecast_date").cast(pl.Date).alias("_pit_date"),
            pl.col("eps_forecast").cast(pl.Float64, strict=False).alias("ac_eps_fy"),
            pl.col("pe_forecast").cast(pl.Float64, strict=False).alias("ac_pe_fy"),
            pl.col("target_price").cast(pl.Float64, strict=False).alias("ac_target_price"),
            _rating_score("rating", _AC_RATING_SCORE).alias("ac_rating"),
            pl.col("analyst_count").cast(pl.Float64, strict=False).alias("ac_analyst_count"),
        )
        .filter(pl.col("_pit_date").is_not_null())
        .unique(subset=["symbol", "_pit_date"], keep="last")
    )


def _rc_events(raw: pl.DataFrame) -> pl.DataFrame:
    """研报预测 → 每 (symbol, 研报日) 一行（多份同日取均值）+ 目标价。"""
    tp = None
    if {"max_price", "min_price"} <= set(raw.columns):
        tp = (
            (pl.col("max_price").cast(pl.Float64, strict=False) + pl.col("min_price").cast(pl.Float64, strict=False))
            / 2.0
        ).alias("rc_target_price")
    exprs = [
        pl.col("eps").cast(pl.Float64, strict=False).mean().alias("rc_eps_y"),
        pl.col("pe").cast(pl.Float64, strict=False).mean().alias("rc_pe_y"),
        pl.col("roe").cast(pl.Float64, strict=False).mean().alias("rc_roe_y"),
        pl.col("tp").cast(pl.Float64, strict=False).mean().alias("rc_tp_wan"),
        _rating_score("rating", _RC_RATING_SCORE).mean().alias("rc_rating"),
    ]
    if tp is not None:
        exprs.append(tp.mean())
    else:
        exprs.append(pl.lit(None, dtype=pl.Float64).alias("rc_target_price"))
    return (
        raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("report_date").cast(pl.Date).alias("_pit_date"),
            pl.col("eps"),
            pl.col("pe"),
            pl.col("roe"),
            pl.col("tp"),
            pl.col("rating"),
            *([] if tp is None else [pl.col("max_price"), pl.col("min_price")]),
        )
        .filter(pl.col("_pit_date").is_not_null())
        .group_by(["symbol", "_pit_date"])
        .agg(exprs)
    )


def _rc_density(events: pl.DataFrame, *, start: str, end: str) -> pl.DataFrame:
    """研报条数密度：近 90 个交易日条数 + 距最近一份研报的自然日。"""
    s, e = _pitlib.parse_window(start, end)
    days = _pitlib.weekdays_between(s, e)
    syms = sorted(set(events["symbol"].to_list()))
    dense = (
        pl.DataFrame({"trade_date": pl.Series(days, dtype=pl.Date)})
        .join(pl.DataFrame({"symbol": pl.Series(syms, dtype=pl.Utf8)}), how="cross")
    )
    per_day = events.group_by(["symbol", "_pit_date"]).len().rename({"_pit_date": "trade_date"})
    dense = (
        dense.join(per_day, on=["symbol", "trade_date"], how="left")
        .with_columns(pl.col("len").fill_null(0))
        .sort(["symbol", "trade_date"])
        .with_columns(pl.col("len").cum_sum().over("symbol").alias("_cum"))
        .with_columns(
            (pl.col("_cum") - pl.col("_cum").shift(_RC_WINDOW).over("symbol").fill_null(0))
            .cast(pl.Float64)
            .alias("rc_cnt_90d")
        )
    )
    last = events.select(["symbol", pl.col("_pit_date").alias("_rc_date")]).unique()
    dense = dense.join_asof(
        last.sort(["symbol", "_rc_date"]),
        left_on="trade_date",
        right_on="_rc_date",
        by="symbol",
        strategy="backward",
    ).with_columns(
        (pl.col("trade_date") - pl.col("_rc_date")).dt.total_days().cast(pl.Float64).alias("rc_days_since")
    )
    return dense.select(["trade_date", "symbol", "rc_cnt_90d", "rc_days_since"])


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """一致预期 + 研报预测 → 日频 PIT 面板列。"""
    frames: list[pl.DataFrame] = []

    try:
        ac_raw = _pitlib.read_curated_pit("analyst_consensus", start=start, end=end)
        ac = _pitlib.expand_pit_daily(
            _ac_events(ac_raw), start=start, end=end, anchor="_pit_date",
            since_col="ac_days_since", value_cols=_AC_VALUE_COLS, out_symbol="symbol",
        )
        frames.append(ac)
    except Exception as exc:  # noqa: BLE001 — 单侧源缺数据不应让另一侧也失效
        logger.warning("research_views: analyst_consensus 跳过: %s", exc)

    try:
        rc_raw = _pitlib.read_curated_pit("report_rc", start=start, end=end)
        rc_events = _rc_events(rc_raw)
        s, e = _pitlib.parse_window(start, end)
        rc_events = rc_events.filter(
            (pl.col("_pit_date") >= s - datetime.timedelta(days=_pitlib.PIT_LOOKBACK_DAYS))
            & (pl.col("_pit_date") <= e)
        )
        if rc_events.is_empty():
            raise ValueError("report_rc 窗口内无研报记录")
        rc_daily = _pitlib.expand_pit_daily(
            rc_events, start=start, end=end, anchor="_pit_date",
            since_col="rc_days_since", value_cols=_RC_VALUE_COLS, out_symbol="symbol",
        )
        rc_daily = rc_daily.join(
            _rc_density(rc_events, start=start, end=end), on=["trade_date", "symbol"], how="left"
        )
        frames.append(rc_daily)
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_views: report_rc 跳过: %s", exc)

    if not frames:
        # 两个源的历史底都很浅（analyst_consensus 2026-09 起、report_rc 2026-08 起）：
        # 窗口早于其历史底时返回**零行但列齐全**的帧，让面板列存在（全 NaN）而不是
        # 让辅助插件加载失败——后者会触发"本次不写面板缓存"，把历史窗口每次拖成重建。
        logger.warning("research_views: 窗口 %s~%s 早于源历史底，返回空列", start, end)
        return pl.DataFrame(
            {"trade_date": [], "symbol": []} | {c: [] for c in _ALL_COLS}
        ).to_pandas()

    out = frames[0]
    for nxt in frames[1:]:
        out = out.join(nxt, on=["trade_date", "symbol"], how="full", coalesce=True)
    for col in _ALL_COLS:
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    pdf = out.select(["trade_date", "symbol", *_ALL_COLS]).to_pandas()
    logger.info("research_views adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, start, end)
    return pdf
