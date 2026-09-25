"""Intraday steps: minute_bars.

Kept out of ``steps/bars.py`` because it shares almost nothing with the daily
path — different horizon, different scope, different schedule — and because a
reader looking for what runs on the daily waves should not have to skip past a
step that never does.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from cnequity.adapters.tdx_protocol.client import (
    fetch_auction_series,
    fetch_minute_bars,
    normalize_with_source,
)
from cnequity.adapters.tdx_protocol.minute_bars import pages_for_window
from cnequity.config import Config
from cnequity.domain.datasets import get_dataset, intraday_datasets
from cnequity.orchestrator.registry import register_step
from cnequity.steps.common import incremental_window, load_symbols
from cnequity.storage import StagingWriter

logger = logging.getLogger(__name__)

# Symbols per staged batch. Small enough that a killed backfill loses minutes
# rather than hours, large enough that the parquet footers stay negligible.
#
# Also the reconnect unit: `fetch_minute_bars` opens fresh TDX connections per
# call, so this many symbols is also how often a full-market sweep pays for a
# TCP handshake. At 50, a 7,747-symbol seed reconnects ~155 times; one of
# those handshakes timed out under sustained load (measured) and — before the
# per-batch try/except below existed — took the whole step down with it. 200
# keeps the same order-of-magnitude "loses minutes, not hours" property while
# cutting reconnects roughly 4x.
_BATCH_SYMBOLS = 200


def _validate_minute_batch(
    df: pl.DataFrame,
    symbols: list[str],
    start: date,
    end: date,
    frequency: str,
) -> pl.DataFrame:
    """Reject a vendor frame that escapes its symbol/date/frequency scope."""
    required = ("symbol", "trade_date", "frequency")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"{frequency}: minute response is missing {missing}")
    normalized = df.with_columns(
        pl.col("symbol").cast(pl.Utf8, strict=False),
        pl.col("trade_date").cast(pl.Date, strict=False),
        pl.col("frequency").cast(pl.Utf8, strict=False),
    )
    returned_symbols = normalized.get_column("symbol")
    if returned_symbols.null_count():
        raise RuntimeError(f"{frequency}: minute response returned a null symbol")
    unexpected = sorted(set(returned_symbols.to_list()) - set(symbols))
    if unexpected:
        raise RuntimeError(
            f"{frequency}: minute response returned unexpected symbol(s): "
            + ", ".join(unexpected[:5])
        )
    dates = normalized.get_column("trade_date")
    invalid_dates = (
        dates.is_null() | (dates < start).fill_null(False) | (dates > end).fill_null(False)
    )
    if normalized.filter(invalid_dates).height:
        raise RuntimeError(
            f"{frequency}: minute response returned row(s) outside "
            f"requested window {start.isoformat()}..{end.isoformat()}"
        )
    returned_frequencies = set(normalized.get_column("frequency").drop_nulls().to_list())
    if returned_frequencies != {frequency}:
        raise RuntimeError(
            f"{frequency}: minute response returned unexpected frequency values "
            f"{sorted(returned_frequencies)}"
        )
    return normalized


class MinuteBarsScopeError(RuntimeError):
    """Raised when the configured scope cannot be resolved to symbols."""


def _index_members(config: Config, index_symbol: str, scope_label: str = "minute_bars") -> list[str]:
    """Latest known constituents of *index_symbol* from ``index_constituents``."""
    from cnequity.query.parquet_scan import dataset_has_parquet, scan_parquet_root

    root = config.curated_root / "index_constituents"
    if not dataset_has_parquet(root):
        raise MinuteBarsScopeError(
            f"{scope_label} scope 'index:{index_symbol}' needs the index_constituents "
            "dataset, which is empty — run `cne run daily` (or `cne backfill "
            f"index_constituents`) first, or set [{scope_label}].scope = 'watchlist'"
        )
    df = (
        scan_parquet_root(root, partition_col="as_of_date", hive=False)
        .filter(pl.col("index_symbol") == index_symbol)
        .select("symbol", "as_of_date")
        .collect()
    )
    if df.is_empty():
        raise MinuteBarsScopeError(
            f"index_constituents holds no rows for {index_symbol!r}; "
            "check the index symbol or pick another scope"
        )
    latest = df["as_of_date"].max()
    return sorted(df.filter(pl.col("as_of_date") == latest)["symbol"].unique().to_list())


def resolve_scope(config: Config) -> list[str]:
    """Symbols the intraday capture covers, per ``[minute_bars].scope``.

    ``index:<symbol>`` — that index's latest constituents (the default;
    沪深300 is ~300 names, about 2MB a day at 1m).
    ``watchlist`` — exactly ``[minute_bars].symbols``.
    ``all`` — the whole universe. ~1.3M rows and ~30MB a day; opt in knowingly.
    """
    scope = (config.minute_bars_scope or "").strip()
    if scope == "all":
        # BJ has no TDX intraday route at all, so it would be all failures.
        return [s for s in load_symbols(config) if not s.endswith(".BJ")]
    if scope == "watchlist":
        symbols = [s.strip() for s in config.minute_bars_symbols if s.strip()]
        if not symbols:
            raise MinuteBarsScopeError(
                "[minute_bars].scope = 'watchlist' but [minute_bars].symbols is empty"
            )
        return symbols
    if scope.startswith("index:"):
        return _index_members(config, scope.split(":", 1)[1].strip())
    raise MinuteBarsScopeError(
        f"unknown [minute_bars].scope {scope!r} (expected 'all', 'watchlist', or 'index:<symbol>')"
    )


def resolve_auction_scope(config: Config) -> list[str]:
    """Symbols the auction capture covers, per ``[auction_series].scope``.

    Same shapes as ``[minute_bars].scope``, but the default is ``watchlist``:
    indices have no auction process, so an index default would fail empty.
    """
    scope = (config.auction_series_scope or "").strip()
    if scope == "all":
        return [s for s in load_symbols(config) if not s.endswith(".BJ")]
    if scope == "watchlist":
        symbols = [s.strip() for s in config.auction_series_symbols if s.strip()]
        if not symbols:
            raise MinuteBarsScopeError(
                "[auction_series].scope = 'watchlist' but [auction_series].symbols is empty"
            )
        return symbols
    if scope.startswith("index:"):
        return _index_members(config, scope.split(":", 1)[1].strip(), "auction_series")
    raise MinuteBarsScopeError(
        f"unknown [auction_series].scope {scope!r} (expected 'all', 'watchlist', or 'index:<symbol>')"
    )


def horizon_start(dataset: str, today: date) -> date | None:
    """Earliest date the source still serves, or None when unbounded."""
    return get_dataset(dataset).earliest_available(today)


def _window(config: Config, dataset: str, trade_date: date) -> tuple[date, date]:
    """Fetch window, clamped to the source's retention horizon.

    Clamping rather than failing: a first run legitimately asks for more than
    the source has, and the honest answer is "here is everything that exists",
    with the clamp logged so it is not mistaken for complete history.
    """
    if getattr(config, "_backfill", False):
        end = getattr(config, "_backfill_end", None) or trade_date
        start = getattr(config, "_backfill_start", None) or (end - timedelta(days=365))
    else:
        start = incremental_window(config, dataset, trade_date)
        end = trade_date

    earliest = horizon_start(dataset, trade_date)
    if earliest is not None and start < earliest:
        logger.warning(
            "%s: requested start %s is older than the source horizon "
            "(~%s, %d trading days); clamping to %s",
            dataset,
            start,
            earliest,
            get_dataset(dataset).history_horizon_days,
            earliest,
        )
        start = earliest
    return start, min(end, trade_date)


def capture_intraday_bars(
    config: Config,
    trade_date: date,
    run_id: str,
    *,
    dataset: str,
    frequency: str,
) -> dict:
    """Capture *frequency* bars for the configured scope into *dataset*.

    Never on the default daily waves. Full-market 1m is ~35MB a day and would
    change what `cne init` costs a user who never asked for it, so this runs
    only when a config opts in, only over the scope that config names, and only
    for the frequencies it lists.
    """
    if not config.minute_bars_enabled:
        return {
            "rows_read": 0,
            "rows_written": 0,
            "note": "intraday capture disabled ([minute_bars].enabled = false)",
        }
    if frequency not in config.minute_bars_frequencies:
        return {
            "rows_read": 0,
            "rows_written": 0,
            "note": (
                f"{frequency} not in [minute_bars].frequencies "
                f"({', '.join(config.minute_bars_frequencies) or 'empty'})"
            ),
        }

    symbols = resolve_scope(config)
    start, end = _window(config, dataset, trade_date)
    if start > end:
        return {"rows_read": 0, "rows_written": 0, "note": f"empty window {start}..{end}"}

    # Bound the page walk: without it, every symbol is paged back to its full
    # retention depth and the extra pages are then discarded by the window
    # filter — 29 requests where 1 would do on the daily path.
    #
    # The depth is trade_date -> start, NOT end -> start. The wire always pages
    # back from the live tip (offset 0 = today), regardless of what `end` is,
    # so a backfill slice near the historical edge still has to walk through
    # everything between today and its start before reaching it — a shallow
    # 10-day-wide slice sitting 140 days back needs ~30 pages, not ~4. Using
    # the slice's own width here made every page land after `end`, get
    # discarded by the date filter, and every symbol come back with zero rows
    # — silently, with no error, indistinguishable from "TDX has nothing here"
    # until traced back to a raw wire probe.
    trading_days = max(1, _approx_trading_days(config, start, trade_date))
    max_pages = pages_for_window(frequency, trading_days)

    logger.info(
        "%s: %d symbol(s) %s, %s..%s (~%d trading days, ≤%d page(s)/symbol)",
        dataset,
        len(symbols),
        frequency,
        start,
        end,
        trading_days,
        max_pages,
    )

    writer = StagingWriter(config.staging_root)
    rate_limit = config.tdx_rate_limit_spec()
    written = 0
    failed: list[str] = []
    with_rows: set[str] = set()

    for index in range(0, len(symbols), _BATCH_SYMBOLS):
        chunk = symbols[index : index + _BATCH_SYMBOLS]
        try:
            df, chunk_failed = fetch_minute_bars(
                chunk,
                start,
                end,
                frequency=frequency,
                rate_limit=rate_limit,
                backfill=getattr(config, "_backfill", False),
                config=config,
                max_pages=max_pages,
                require_complete=True,
                workers=config.minute_bars_fetch_workers,
            )
        except Exception as exc:  # noqa: BLE001 — recorded, sweep continues
            # A batch failing outright (e.g. a connect timeout after hundreds
            # of prior reconnects on a full-market sweep) must cost this batch,
            # not the whole step — the same contract as a single symbol's
            # failure, just at a coarser grain. None of these symbols got a
            # chance to succeed or fail individually, so all of them count as
            # failed rather than silently vanishing from the totals.
            logger.warning(
                "%s: batch of %d symbol(s) failed outright (%s..%s): %s",
                dataset,
                len(chunk),
                chunk[0],
                chunk[-1],
                exc,
            )
            failed.extend(chunk)
            continue
        failed.extend(chunk_failed)
        if df.is_empty():
            continue
        df = _validate_minute_batch(df, chunk, start, end, frequency)
        with_rows.update(df["symbol"].unique().to_list())
        df = normalize_with_source(df, dataset=dataset)
        writer.write_batch(dataset, run_id, f"intraday-{index // _BATCH_SYMBOLS:04d}", df)
        written += df.height
        logger.info(
            "%s: %d/%d symbols, %d rows staged",
            dataset,
            min(index + _BATCH_SYMBOLS, len(symbols)),
            len(symbols),
            written,
        )

    # A symbol can come back empty without failing: a name suspended for the
    # whole window genuinely has no intraday bars. That is the right answer, but
    # it is indistinguishable from a silent fetch hole unless the count is
    # reported, so record both rather than only the failures.
    result: dict = {
        "rows_read": written,
        "rows_written": written,
        "symbols": len(symbols),
        "symbols_with_rows": len(with_rows),
        "failed_symbols": len(failed),
        "note": f"{frequency} {start}..{end} scope={config.minute_bars_scope}",
    }
    silent = len(symbols) - len(with_rows) - len(failed)
    if silent > 0:
        logger.info(
            "%s: %d symbol(s) returned no bars without erroring "
            "(suspended for the whole window, or never traded it)",
            dataset,
            silent,
        )
    if failed:
        result["context_updates"] = {
            "audit_findings": [
                {
                    "dataset": dataset,
                    "severity": "warning",
                    "check": "minute_bars_symbol_fetch",
                    "message": (
                        f"{len(failed)}/{len(symbols)} symbol(s) returned no {frequency} "
                        f"bars for {start}..{end} (e.g. {', '.join(failed[:5])})"
                    ),
                }
            ]
        }
    if written == 0 and symbols:
        raise RuntimeError(
            f"{dataset}: no rows for any of {len(symbols)} symbol(s) over {start}..{end} "
            "— check TDX reachability and that the window is inside the source horizon"
        )
    return result


def capture_auction_series(
    config: Config,
    trade_date: date,
    run_id: str,
) -> dict:
    """集合竞价过程快照 for the configured scope (0x056A, opt-in).

    One request per symbol per day. The source serves it from the money-flow
    host group and historical depth varies by host, so this is a same-day
    capture, not a backfill target — like minute_bars, it never runs on the
    default daily waves.
    """
    if not config.auction_series_enabled:
        return {
            "rows_read": 0,
            "rows_written": 0,
            "note": "auction capture disabled ([auction_series].enabled = false)",
        }

    symbols = resolve_auction_scope(config)
    logger.info(
        "auction_series: %d symbol(s) on %s scope=%s",
        len(symbols),
        trade_date,
        config.auction_series_scope,
    )

    writer = StagingWriter(config.staging_root)
    rate_limit = config.tdx_rate_limit_spec()
    df, failed = fetch_auction_series(
        symbols, trade_date, rate_limit=rate_limit, config=config
    )
    if df.is_empty():
        if failed:
            raise RuntimeError(
                f"auction_series: no rows and {len(failed)}/{len(symbols)} symbol(s) "
                f"failed on {trade_date} — check TDX reachability and auction host coverage"
            )
        return {
            "rows_read": 0,
            "rows_written": 0,
            "symbols": len(symbols),
            "failed_symbols": 0,
            "note": f"no auction snapshots returned for {trade_date}",
        }

    df = normalize_with_source(df, dataset="auction_series")
    writer.write_batch("auction_series", run_id, "batch-0", df)
    result: dict = {
        "rows_read": df.height,
        "rows_written": df.height,
        "symbols": len(symbols),
        "failed_symbols": len(failed),
        "note": f"scope={config.auction_series_scope}",
    }
    if failed:
        result["context_updates"] = {
            "audit_findings": [
                {
                    "dataset": "auction_series",
                    "severity": "warning",
                    "check": "auction_symbol_fetch",
                    "message": (
                        f"{len(failed)}/{len(symbols)} symbol(s) returned no auction "
                        f"snapshots for {trade_date} (e.g. {', '.join(failed[:5])})"
                    ),
                }
            ]
        }
    return result


def _register_intraday_steps() -> None:
    """One step per registered intraday dataset, named after the dataset.

    Generated rather than written out so that adding a frequency stays a single
    registry entry. The step name must equal the dataset name — `cne backfill
    <dataset>` and the compact/watermark plumbing both key on that.
    """
    for frequency, dataset in sorted(intraday_datasets().items()):

        def _step(
            config: Config,
            trade_date: date,
            run_id: str,
            context: dict,
            *,
            _dataset: str = dataset,
            _frequency: str = frequency,
        ) -> dict:
            return capture_intraday_bars(
                config, trade_date, run_id, dataset=_dataset, frequency=_frequency
            )

        _step.__name__ = f"step_{dataset}"
        _step.__doc__ = f"Capture {frequency} bars for the configured scope (opt-in)."
        register_step(dataset, group="intraday", depends_on=["instruments"])(_step)


_register_intraday_steps()


def _step_auction_series(
    config: Config,
    trade_date: date,
    run_id: str,
    context: dict,
) -> dict:
    return capture_auction_series(config, trade_date, run_id)


_step_auction_series.__name__ = "auction_series"
_step_auction_series.__doc__ = "Capture 集合竞价过程快照 for the configured scope (opt-in)."
register_step("auction_series", group="intraday", depends_on=["instruments"])(
    _step_auction_series
)


def _approx_trading_days(config: Config, start: date, end: date) -> int:
    """Trading days in [start, end] from the authoritative calendar."""
    from cnequity.steps.common import list_trading_dates

    return max(1, len(list_trading_dates(config, start, end)))
