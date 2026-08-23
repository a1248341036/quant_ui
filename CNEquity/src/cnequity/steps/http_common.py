"""Shared step helper for EastMoney / CNINFO HTTP datasets."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.domain.schemas import data_version_for, with_provenance
from cnequity.steps.common import fetch_incremental_daily, write_simple


def write_fetched(
    config: Config,
    run_id: str,
    dataset: str,
    df: pl.DataFrame,
    *,
    source: str,
    batch_id: str = "batch-0",
) -> dict:
    df = with_provenance(df, source=source, data_version=data_version_for(dataset))
    return write_simple(config, run_id, dataset, df, batch_id=batch_id)


def run_incremental_fetched(
    config: Config,
    trade_date: date,
    run_id: str,
    dataset: str,
    fetch_fn: Callable[[date], pl.DataFrame],
    *,
    source: str,
    allow_empty: bool = False,
    universe: set[str] | None = None,
    date_col: str | None = None,
) -> dict:
    df, findings = fetch_incremental_daily(
        config,
        dataset,
        trade_date,
        fetch_fn,
        allow_empty=allow_empty,
        date_col=date_col,
    )
    if universe and not df.is_empty():
        # Constrain a live snapshot (e.g. EastMoney valuation clist) to the
        # tradable universe: the source returns delisted / never-traded names the
        # lake must not carry. An empty universe means "cannot reconcile" — skip
        # filtering rather than dropping every row.
        source_rows = df.height
        if "symbol" not in df.columns:
            raise RuntimeError(f"{dataset}: cannot reconcile source rows without a symbol column")
        df = df.filter(pl.col("symbol").is_in(list(universe)))
        if df.is_empty():
            raise RuntimeError(
                f"{dataset}: source returned {source_rows} row(s), but none matched the "
                f"reconciled universe ({len(universe)} symbol(s))"
            )
    if df.is_empty():
        out: dict = {"rows_read": 0, "rows_written": 0}
        if findings:
            out["context_updates"] = {"audit_findings": findings}
            # A session-dense dataset is only complete when every requested
            # trading session has a response. Keep the staged subset visible,
            # but make the step retryable and block compact from checkpointing
            # past the missing session. Snapshot coverage gaps are deliberately
            # not promoted here: those missed historical snapshots cannot be
            # replayed without manufacturing point-in-time values.
            if any(f.get("check") == "session_dense_empty_days" for f in findings):
                out["status"] = "warning"
        return out
    result = write_fetched(config, run_id, dataset, df, source=source)
    if findings:
        result["context_updates"] = {"audit_findings": findings}
        if any(f.get("check") == "session_dense_empty_days" for f in findings):
            result["status"] = "warning"
    return result


def empty_ok(df: pl.DataFrame, dataset: str, trade_date: date) -> None:
    if df.is_empty():
        raise RuntimeError(f"{dataset}: no rows returned for {trade_date.isoformat()}")
