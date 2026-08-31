"""Tushare fetchers for the capital/signals datasets, mapped onto existing schemas.

``fund_flow`` / ``dragon_tiger`` / ``block_trades`` were historically
EastMoney-sourced and ``shareholder_counts`` was frozen (EastMoney skip step).
The Tushare middleware serves the equivalents — ``moneyflow``, ``top_list``,
``block_trade``, ``stk_holdernumber`` — as full-market per-day batch queries,
so these fetchers produce rows in the EXISTING lake schemas with the SAME
units as the EastMoney history they join:

- fund_flow: net inflows in 元 (Tushare moneyflow reports 万元 → ×1e4)
- dragon_tiger: amounts in 元 (Tushare top_list already 元)
- block_trades: price 元, volume 万股, amount 万元 (identical to EastMoney);
  ``premium_ratio`` is computed as ``price/close - 1`` against the Tushare
  wide archive (EastMoney publishes it, Tushare does not)
- shareholder_counts: holder_count 户, change pct %, avg_float_shares 股/户,
  avg_holding_value 元/户 — the derived columns are recomputed from prior
  curated counts and the wide archive (Tushare only publishes holder_num)

Rows are stamped ``source="tushare"`` by the normal write path, so the
EastMoney history and Tushare increments coexist in one schema.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from cnequity.config import Config
from cnequity.external.tushare_fetch import _fetch_with_retry, _get_pro

logger = logging.getLogger(__name__)

# Widest window the wide archive is scanned backward from the earliest count
# date, so an as-of join can reach the last session at or before a 旬末/月末
# count that fell on a weekend.
_WIDE_LOOKBACK_DAYS = 10

_HOLDER_SCHEMA_COLS = [
    "symbol",
    "count_date",
    "holder_count",
    "holder_count_change_pct",
    "avg_float_shares",
    "avg_holding_value",
    "announce_date",
]


def _ts_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def _fetch(config: Config, api: str, **params) -> pl.DataFrame:
    pro = _get_pro(config)
    return _fetch_with_retry(pro, api, interval=config.external_tushare_wide_interval, **params)


def _wide_daily(config: Config, start: date, end: date) -> pl.DataFrame:
    """(symbol, trade_date, close, float_share, circ_mv) from the wide archive.

    Empty on any archive failure — premium/avg columns degrade to null rather
    than failing the whole fetch.
    """
    try:
        from cnequity.external.tushare_wide_raw import ADAPTER as _wide_adapter

        lf = _wide_adapter.scan(config, "stock_daily_wide", start=start, end=end)
        return (
            lf.select(
                "symbol",
                "trade_date",
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col("float_share").cast(pl.Float64, strict=False),
                pl.col("circ_mv").cast(pl.Float64, strict=False),
            )
            .unique(subset=["symbol", "trade_date"], keep="last")
            .collect()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("tushare_capital: wide archive scan failed (%s..%s): %s",
                       start.isoformat(), end.isoformat(), exc)
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "trade_date": pl.Date,
                "close": pl.Float64,
                "float_share": pl.Float64,
                "circ_mv": pl.Float64,
            }
        )


# ── fund_flow (Tushare moneyflow) ────────────────────────────────────────


def _net_amount_expr(cols: set[str], kind: str) -> pl.Expr:
    """净流入表达式：优先 buy/sell 差值（两个方向列 always present per docs），
    缺 sell 列时退回 vendor 的 net_* 列。"""
    if f"sell_{kind}_amount" in cols:
        return pl.col(f"buy_{kind}_amount").cast(pl.Float64, strict=False) - pl.col(
            f"sell_{kind}_amount"
        ).cast(pl.Float64, strict=False)
    return pl.col(f"net_{kind}_amount").cast(pl.Float64, strict=False)


def fetch_fund_flow_tushare(trade_date: date, *, config: Config) -> pl.DataFrame:
    raw = _fetch(config, "moneyflow", trade_date=_ts_date(trade_date))
    if raw.is_empty():
        return pl.DataFrame()
    cols = set(raw.columns)
    elg, lg, md, sm = (_net_amount_expr(cols, k) for k in ("elg", "lg", "md", "sm"))
    # moneyflow 金额单位是万元；curated 东财历史是元 → ×1e4 对齐
    return (
        raw.select(
            pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
            pl.lit(trade_date).alias("trade_date"),
            ((elg + lg) * 1e4).alias("main_net_inflow"),
            (elg * 1e4).alias("super_large_net_inflow"),
            (lg * 1e4).alias("large_net_inflow"),
            (md * 1e4).alias("medium_net_inflow"),
            (sm * 1e4).alias("small_net_inflow"),
        )
        .unique(subset=["symbol", "trade_date"], keep="last")
    )


# ── dragon_tiger (Tushare top_list) ─────────────────────────────────────


def fetch_dragon_tiger_tushare(trade_date: date, *, config: Config) -> pl.DataFrame:
    raw = _fetch(config, "top_list", trade_date=_ts_date(trade_date))
    if raw.is_empty():
        return pl.DataFrame()
    cols = set(raw.columns)
    net = (
        pl.col("net_amount").cast(pl.Float64, strict=False)
        if "net_amount" in cols
        else pl.col("l_buy").cast(pl.Float64, strict=False)
        - pl.col("l_sell").cast(pl.Float64, strict=False)
    )
    return (
        raw.select(
            pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
            pl.lit(trade_date).alias("trade_date"),
            pl.col("reason").cast(pl.Utf8).alias("reason"),
            pl.col("l_buy").cast(pl.Float64, strict=False).alias("buy_amount"),
            pl.col("l_sell").cast(pl.Float64, strict=False).alias("sell_amount"),
            net.alias("net_amount"),
        )
        # reason 是主键的一部分且校验要求非空：空原因的行无法入库，直接丢弃
        .filter(pl.col("reason").is_not_null() & (pl.col("reason").str.strip_chars() != ""))
        .unique(subset=["symbol", "trade_date", "reason"], keep="last")
    )


# ── block_trades (Tushare block_trade) ──────────────────────────────────


def fetch_block_trades_tushare(trade_date: date, *, config: Config) -> pl.DataFrame:
    raw = _fetch(config, "block_trade", trade_date=_ts_date(trade_date))
    if raw.is_empty():
        return pl.DataFrame()
    out = (
        raw.select(
            pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
            pl.lit(trade_date).alias("trade_date"),
            pl.col("price").cast(pl.Float64, strict=False),
            pl.col("vol").cast(pl.Float64, strict=False).alias("volume"),  # 万股
            pl.col("amount").cast(pl.Float64, strict=False),  # 万元
        )
        # price/volume 是主键且校验要求非空
        .filter(pl.col("price").is_not_null() & pl.col("volume").is_not_null())
    )
    wide = _wide_daily(config, trade_date, trade_date)
    if wide.is_empty():
        return out.with_columns(pl.lit(None, dtype=pl.Float64).alias("premium_ratio")).unique(
            subset=["symbol", "trade_date", "price", "volume"], keep="last"
        )
    return (
        out.join(wide.select("symbol", pl.col("close").alias("_close")), on="symbol", how="left")
        .with_columns(
            pl.when(pl.col("_close") > 0)
            .then(pl.col("price") / pl.col("_close") - 1.0)
            .otherwise(None)
            .alias("premium_ratio")
        )
        .drop("_close")
        .unique(subset=["symbol", "trade_date", "price", "volume"], keep="last")
    )


# ── shareholder_counts (Tushare stk_holdernumber) ───────────────────────


def _curated_holder_history(config: Config) -> pl.DataFrame:
    """(symbol, count_date, holder_count) already in curated, for chaining."""
    schema = {"symbol": pl.Utf8, "count_date": pl.Date, "holder_count": pl.Float64}
    root = config.curated_root / "shareholder_counts"
    files = sorted(root.rglob("*.parquet")) if root.exists() else []
    if not files:
        return pl.DataFrame(schema=schema)
    try:
        df = pl.read_parquet(files, hive_partitioning=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("tushare_capital: curated shareholder_counts scan failed: %s", exc)
        return pl.DataFrame(schema=schema)
    return df.select(
        pl.col("symbol").cast(pl.Utf8),
        pl.col("count_date").cast(pl.Date, strict=False),
        pl.col("holder_count").cast(pl.Float64, strict=False),
    ).drop_nulls()


def _enrich_holder_counts(raw: pl.DataFrame, config: Config) -> pl.DataFrame:
    base = raw.select(
        pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
        pl.col("end_date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias("count_date"),
        pl.col("holder_num").cast(pl.Float64, strict=False).alias("holder_count"),
        pl.col("ann_date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias("announce_date"),
    ).drop_nulls(subset=["symbol", "count_date", "holder_count", "announce_date"])
    if base.is_empty():
        return pl.DataFrame(schema={c: pl.Utf8 if c == "symbol" else pl.Date if "date" in c else pl.Float64 for c in _HOLDER_SCHEMA_COLS})

    # 1. 环比变化：与 curated 历史链式相接后按 (symbol, count_date) 取 shift。
    #    同窗口内多次披露（旬末+月末）也能链上；首次披露为 null（与东财一致）。
    combined = (
        pl.concat(
            [_curated_holder_history(config), base.select("symbol", "count_date", "holder_count")],
            how="vertical_relaxed",
        )
        .unique(subset=["symbol", "count_date"], keep="last")
        .sort(["symbol", "count_date"])
    )
    chg = combined.select(
        "symbol",
        "count_date",
        ((pl.col("holder_count") / pl.col("holder_count").shift(1).over("symbol") - 1.0) * 100.0).alias("_chg_pct"),
    )
    base = base.join(chg, on=["symbol", "count_date"], how="left").rename({"_chg_pct": "holder_count_change_pct"})

    # 2. 户均流通股 / 户均市值：count_date 当日（asof 回看至最近交易日）的
    #    float_share(万股)/circ_mv(万元) → 股/户、元/户。
    wide = _wide_daily(
        config,
        base["count_date"].min() - timedelta(days=_WIDE_LOOKBACK_DAYS),
        base["count_date"].max(),
    )
    if wide.is_empty():
        base = base.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("avg_float_shares"),
            pl.lit(None, dtype=pl.Float64).alias("avg_holding_value"),
        )
    else:
        base = (
            base.sort(["symbol", "count_date"])
            .join_asof(
                wide.sort(["symbol", "trade_date"]),
                left_on="count_date",
                right_on="trade_date",
                by="symbol",
                strategy="backward",
            )
            .drop("trade_date")
            .with_columns(
                pl.when(pl.col("holder_count") > 0)
                .then(pl.col("float_share") * 1e4 / pl.col("holder_count"))
                .otherwise(None)
                .alias("avg_float_shares"),
                pl.when(pl.col("holder_count") > 0)
                .then(pl.col("circ_mv") * 1e4 / pl.col("holder_count"))
                .otherwise(None)
                .alias("avg_holding_value"),
            )
            .drop("float_share", "circ_mv")
        )
    return base.select(_HOLDER_SCHEMA_COLS).unique(subset=["symbol", "count_date", "announce_date"], keep="last")


def fetch_holder_counts_tushare(start: date, end: date, *, config: Config) -> pl.DataFrame:
    """股东户数，公告日落在 [start, end] 的全市场批量（一天一次调用）。"""
    pro = _get_pro(config)
    interval = config.external_tushare_wide_interval
    frames: list[pl.DataFrame] = []
    day = start
    while day <= end:
        raw = _fetch_with_retry(pro, "stk_holdernumber", interval=interval, ann_date=_ts_date(day))
        if raw is not None and not raw.is_empty():
            frames.append(raw)
        day += timedelta(days=1)
    if not frames:
        return pl.DataFrame()
    merged = pl.concat(frames, how="diagonal_relaxed").unique(
        subset=["ts_code", "ann_date", "end_date"], keep="last"
    )
    return _enrich_holder_counts(merged, config)
