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
    missing_contracts: list[str] = []

    def _fetch(d: date) -> pl.DataFrame:
        # 单日缺失合约在 adapter 内降级为部分成功（不 raise）；这里接住缺失清单，
        # 让 step 以 warning 收尾（failed_contracts 触发 _has_partial_failures），
        # 已写入合约保留、缺口留给 retry / stale。
        return fetch_commodity_bars(
            d, config=config, strict=True, missing_out=missing_contracts
        )

    result = run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "commodity_bars",
        _fetch,
        # Row-level ``source`` is set by adapters (eastmoney / sina); this is
        # only the fallback stamp when a frame lacks the column.
        source="eastmoney",
        allow_empty=False,
    )
    if missing_contracts:
        result["failed_contracts"] = sorted(set(missing_contracts))
    return result
