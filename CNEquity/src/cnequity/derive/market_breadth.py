"""Market breadth metrics computed from daily_bars via the standard reader."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from cnequity.config import Config
from cnequity.query.parquet_scan import collect_parquet_root

MARKET_BREADTH_METRICS = (
    "advance_count",
    "decline_count",
    "flat_count",
    "limit_up_count",
    "limit_down_count",
    "advance_ratio",
    "total_count",
)


def _read_bars(config: Config, trade_date: date) -> pl.DataFrame:
    """Load one day's raw bars, external archive first with curated fallback.

    The standard reader resolves the tushare-wide archive this deployment
    writes to; older/offline lakes only have curated hive partitions.
    """
    from cnequity.query.parquet_scan import collect_parquet_root
    from cnequity.query.reader import load

    try:
        df = load("daily_bars", start=trade_date, end=trade_date, config=config)
        if not df.is_empty():
            return df
    except Exception:  # noqa: BLE001 — no external source here, fall through
        pass

    root = config.curated_root / "daily_bars"
    if not root.exists():
        return pl.DataFrame()
    try:
        df = collect_parquet_root(
            root,
            partition_col="trade_date",
            start=trade_date,
            end=trade_date,
        )
    except FileNotFoundError:
        return pl.DataFrame()
    if all(col in df.columns for col in ("symbol", "trade_date")):
        if "fetched_at" in df.columns:
            df = df.sort("fetched_at")
        df = df.unique(subset=["symbol", "trade_date"], keep="last")
    return df.filter(pl.col("trade_date") == trade_date)


def _prev_trading_date(config: Config, trade_date: date) -> date | None:
    cal_root = config.curated_root / "trading_calendar"
    if not cal_root.exists():
        return None
    try:
        cal = collect_parquet_root(
            cal_root,
            partition_col="trade_date",
            end=trade_date,
        )
    except FileNotFoundError:
        return None
    if cal.is_empty() or not {"trade_date", "is_trading"}.issubset(cal.columns):
        return None
    if "fetched_at" in cal.columns:
        cal = cal.sort("fetched_at")
    cal = cal.unique(subset=["trade_date"], keep="last")
    prior = cal.filter((pl.col("trade_date") < trade_date) & pl.col("is_trading")).sort(
        "trade_date", descending=True
    )
    if prior.is_empty():
        return None
    return prior["trade_date"][0]


def _read_trading_status(root: Path, trade_date: date) -> pl.DataFrame:
    """Read optional same-day status evidence without scanning all history."""
    if not root.exists():
        return pl.DataFrame()
    try:
        df = collect_parquet_root(
            root,
            partition_col="trade_date",
            start=trade_date,
            end=trade_date,
        )
    except FileNotFoundError:
        return pl.DataFrame()
    required = {"symbol", "trade_date", "status"}
    if not required.issubset(df.columns):
        return pl.DataFrame()
    if "fetched_at" in df.columns:
        df = df.sort("fetched_at")
    return df.unique(subset=["symbol", "trade_date"], keep="last").select(
        ["symbol", "trade_date", "status"]
    )


def _limit_threshold(symbol: str, status: str | None) -> float:
    """Return a conservative daily limit threshold for a symbol."""
    if str(status or "").strip().lower() in {"st", "*st"}:
        return 0.045
    code, _, exchange = str(symbol).partition(".")
    if exchange == "BJ":
        return 0.295
    if code.startswith("30") or code.startswith("688"):
        return 0.195
    return 0.095


def compute_market_breadth(config: Config, trade_date: date) -> pl.DataFrame:
    today = _read_bars(config, trade_date)
    if today.is_empty():
        return pl.DataFrame()

    # A suspended security still has an OHLC placeholder in daily_bars, but
    # the lake contract marks it with volume=0 and amount=0.  Counting that
    # carried-forward close as flat would dilute every breadth ratio and make
    # ``total_count`` depend on the day's suspension population.  Breadth is
    # about names that actually traded, so remove no-trade rows before joining
    # against the prior close.  Keep this volume-based guard independent of
    # trading_status: historical status is intentionally sparse and may not
    # exist yet for an otherwise valid daily-bars window.
    if "volume" in today.columns:
        today = today.filter((pl.col("volume") > 0) | pl.col("volume").is_null())
    if today.is_empty():
        return pl.DataFrame()

    prev_date = _prev_trading_date(config, trade_date)
    if prev_date is None:
        return pl.DataFrame()

    prev = _read_bars(config, prev_date)
    if prev.is_empty():
        return pl.DataFrame()

    joined = today.select(["symbol", "close", "trade_date"]).join(
        prev.select(["symbol", pl.col("close").alias("prev_close")]),
        on="symbol",
        how="inner",
    )
    status = _read_trading_status(config.curated_root / "trading_status", trade_date)
    if status.is_empty():
        joined = joined.with_columns(pl.lit(None, dtype=pl.Utf8).alias("status"))
    else:
        joined = joined.join(status.select(["symbol", "status"]), on="symbol", how="left")
    joined = joined.with_columns(
        ((pl.col("close") - pl.col("prev_close")) / pl.col("prev_close")).alias("pct"),
        pl.struct(["symbol", "status"])
        .map_elements(
            lambda row: _limit_threshold(row["symbol"], row["status"]),
            return_dtype=pl.Float64,
        )
        .alias("limit_threshold"),
    )
    joined = joined.filter(pl.col("prev_close") > 0)

    total = joined.height
    advance = joined.filter(pl.col("pct") > 0).height
    decline = joined.filter(pl.col("pct") < 0).height
    flat = joined.filter(pl.col("pct") == 0).height
    limit_up = joined.filter(pl.col("pct") >= pl.col("limit_threshold")).height
    limit_down = joined.filter(pl.col("pct") <= -pl.col("limit_threshold")).height
    ratio = advance / total if total else 0.0

    values = {
        "advance_count": float(advance),
        "decline_count": float(decline),
        "flat_count": float(flat),
        "limit_up_count": float(limit_up),
        "limit_down_count": float(limit_down),
        "advance_ratio": ratio,
        "total_count": float(total),
    }
    rows = [
        {"trade_date": trade_date, "metric_id": metric_id, "value": val}
        for metric_id, val in values.items()
    ]
    return pl.DataFrame(rows)
