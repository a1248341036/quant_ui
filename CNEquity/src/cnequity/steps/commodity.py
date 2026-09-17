"""Commodity futures continuous bars (L1-adjacent)."""

from __future__ import annotations

from datetime import date

import polars as pl

from cnequity.adapters.eastmoney.commodity_bars import (
    DEFAULT_BACKFILL_START,
    fetch_commodity_bars,
)
from cnequity.config import Config
from cnequity.orchestrator.registry import register_step
from cnequity.steps.http_common import run_incremental_fetched, write_fetched


@register_step("commodity_bars", group="macro_risk")
def step_commodity_bars(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    em = bool(config.sources.get("eastmoney", True))
    sina = bool(config.sources.get("sina", True))
    if not em and not sina:
        raise RuntimeError("commodity_bars: both eastmoney and sina sources disabled")
    if getattr(config, "_backfill", False):
        start = getattr(config, "_backfill_start", None) or DEFAULT_BACKFILL_START
        end = getattr(config, "_backfill_end", None) or trade_date
        df = fetch_commodity_bars(trade_date, config=config, strict=True)
        if df.is_empty():
            raise RuntimeError(
                f"commodity_bars: no rows returned for {start.isoformat()}..{end.isoformat()}"
            )
        dates = df.get_column("trade_date").cast(pl.Date, strict=False)
        out_of_bounds = int((dates.is_null() | (dates < start) | (dates > end)).sum())
        if out_of_bounds:
            raise RuntimeError(
                f"commodity_bars: {out_of_bounds} row(s) outside {start.isoformat()}..{end.isoformat()}"
            )
        return write_fetched(config, run_id, "commodity_bars", df, source="sina")
    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "commodity_bars",
        lambda d: fetch_commodity_bars(d, config=config, strict=True),
        # Row-level ``source`` is set by adapters (eastmoney / sina); this is
        # only the fallback stamp when a frame lacks the column.
        source="eastmoney",
        allow_empty=False,
    )
