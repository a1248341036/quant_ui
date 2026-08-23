"""L2 corporate-event steps: corporate_actions, announcement_index,
earnings_disclosure_schedule."""

from __future__ import annotations

import json
import logging
from datetime import date

import polars as pl

from cnequity.adapters.cninfo.announcements import fetch_announcement_index
from cnequity.adapters.eastmoney.corporate_actions import fetch_corporate_actions_eastmoney
from cnequity.adapters.eastmoney.earnings_disclosure import (
    _backfill_report_dates,
    fetch_earnings_disclosure_schedule,
)
from cnequity.adapters.tdx_protocol.client import fetch_corporate_actions
from cnequity.config import Config
from cnequity.domain.schemas import with_provenance
from cnequity.orchestrator.manifest import Manifest
from cnequity.orchestrator.registry import register_step
from cnequity.quality.failover import (
    snapshot_corporate_actions_backup,
    snapshot_corporate_actions_tdx_backup,
)
from cnequity.steps.common import (
    BACKFILL_START,
    fetch_incremental_daily,
    load_symbols,
    write_simple,
)
from cnequity.steps.http_common import run_incremental_fetched, write_fetched

# TDX xdxr is per-symbol (backfill); EastMoney datacenter supports ex-date filter (daily).
_CANONICAL_BACKFILL = "tdx_protocol"
_CANONICAL_DAILY = "eastmoney"
_CORPORATE_ACTIONS_CHUNK_TASK = "corporate_actions_chunk"
_MIN_EARNINGS_SCHEDULE_SYMBOLS_PER_PERIOD = 100


logger = logging.getLogger(__name__)


def _validate_earnings_schedule_snapshot(df: pl.DataFrame) -> pl.DataFrame:
    """Reject a non-empty but obviously truncated report-period snapshot."""
    if df.is_empty():
        return df
    required = {"symbol", "report_period"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "earnings_disclosure_schedule: response is missing required column(s): "
            + ", ".join(missing)
        )
    counts = (
        df.unique(subset=["symbol", "report_period"])
        .group_by("report_period")
        .agg(pl.len().alias("_symbol_count"))
        .filter(pl.col("_symbol_count") < _MIN_EARNINGS_SCHEDULE_SYMBOLS_PER_PERIOD)
    )
    if not counts.is_empty():
        details = ", ".join(
            f"{row['report_period']}={row['_symbol_count']}" for row in counts.iter_rows(named=True)
        )
        raise RuntimeError(
            "earnings_disclosure_schedule: incomplete report-period snapshot; each "
            f"observed period needs at least {_MIN_EARNINGS_SCHEDULE_SYMBOLS_PER_PERIOD} "
            f"unique symbol(s) ({details})"
        )
    return df


@register_step("corporate_actions", group="core", depends_on=["instruments"])
def step_corporate_actions(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    rl = config.tdx_rate_limit_spec()
    backfill = getattr(config, "_backfill", False)
    findings: list[dict] = []
    failed_symbols: list[str] = []

    if backfill:
        symbols = list(context.get("_retry_symbols") or load_symbols(config))
        batch_id = context.get("_batch_id")
        manifest = Manifest(config.manifest_path) if batch_id else None

        if manifest is not None:
            manifest.set_batch_symbols(run_id, batch_id, symbols)

        completed_symbols: set[str] = set()
        if manifest is not None and context.get("_retry_symbols"):
            for row in manifest.get_batches_for_run(run_id):
                if row["task_id"] != _CORPORATE_ACTIONS_CHUNK_TASK or row["status"] != "success":
                    continue
                completed_symbols.update(json.loads(row["symbols_json"] or "[]"))
        remaining_symbols = [symbol for symbol in symbols if symbol not in completed_symbols]

        def on_progress(done: int, total: int) -> None:
            if manifest is not None and (done % 50 == 0 or done == total):
                try:
                    manifest.touch_batch_heartbeat(run_id, batch_id)
                except Exception as exc:  # noqa: BLE001 — heartbeat is auxiliary
                    logger.warning(
                        "corporate_actions: heartbeat update failed at %d/%d: %s",
                        done,
                        total,
                        exc,
                    )

        if config.failover_enabled:
            # Best-effort: this writes an EastMoney snapshot for cross-source
            # audit, not the canonical rows. It must never decide whether the
            # backfill runs — when EastMoney changed its filter grammar the
            # raise from here aborted the whole step before TDX, the actual
            # primary, was contacted at all.
            try:
                snapshot_corporate_actions_backup(
                    config, trade_date=trade_date, run_id=run_id, backfill=True
                )
            except Exception as exc:  # noqa: BLE001 — audit artifact, not the data
                logger.warning(
                    "corporate_actions: backup snapshot failed (%s: %s); "
                    "continuing with the canonical TDX fetch",
                    type(exc).__name__,
                    exc,
                )
        frames: list[pl.DataFrame] = []
        failed_symbols: list[str] = []
        batch_size = max(1, config.batch_size)
        for chunk_index in range(0, len(remaining_symbols), batch_size):
            chunk = remaining_symbols[chunk_index : chunk_index + batch_size]
            chunk_number = chunk_index // batch_size
            chunk_batch_id = f"{batch_id or 'batch-0'}-chunk-{chunk_number:04d}"
            try:
                df_chunk = fetch_corporate_actions(
                    trade_date,
                    symbols=chunk,
                    backfill=True,
                    rate_limit=rl,
                    allow_mock=config.tdx_allow_mock,
                    primary_only=True,
                    config=config,
                    on_progress=lambda done, total, offset=chunk_index: on_progress(
                        offset + done, len(remaining_symbols)
                    ),
                    fail_loud=True,
                    allow_empty=True,
                )
            except Exception as exc:  # noqa: BLE001 — preserve completed chunks for retry
                failed_symbols.extend(chunk)
                logger.warning(
                    "corporate_actions chunk %d failed for %d symbols: %s",
                    chunk_number,
                    len(chunk),
                    exc,
                )
                continue

            if not df_chunk.is_empty():
                frames.append(df_chunk)
            if manifest is not None:
                # The fetch and staging happen before this success receipt. If
                # the process dies earlier, the parent batch remains retryable;
                # an unreceipted chunk is safely fetched again.
                staged_chunk = with_provenance(
                    df_chunk,
                    source=_CANONICAL_BACKFILL,
                    data_version="v1",
                )
                write_simple(
                    config,
                    run_id,
                    "corporate_actions",
                    staged_chunk,
                    batch_id=chunk_batch_id,
                )
                manifest.start_batch(
                    run_id,
                    chunk_batch_id,
                    task_id=_CORPORATE_ACTIONS_CHUNK_TASK,
                    dataset="corporate_actions",
                    symbols=chunk,
                    window_start=getattr(config, "_backfill_start", None).isoformat()
                    if getattr(config, "_backfill_start", None)
                    else None,
                    window_end=getattr(config, "_backfill_end", None).isoformat()
                    if getattr(config, "_backfill_end", None)
                    else trade_date.isoformat(),
                    blocks_compaction=False,
                )
                manifest.finish_batch(
                    run_id,
                    chunk_batch_id,
                    "success",
                    rows_read=df_chunk.height,
                    rows_written=df_chunk.height,
                )

        if frames:
            df = pl.concat(frames, how="diagonal_relaxed")
        else:
            df = pl.DataFrame()
        if failed_symbols:
            failed_symbols = list(dict.fromkeys(failed_symbols))
        if failed_symbols:
            logger.warning(
                "corporate_actions backfill incomplete: %d/%d symbols failed; "
                "successful chunks are staged and will be skipped on retry",
                len(failed_symbols),
                len(symbols),
            )
        canonical_source = _CANONICAL_BACKFILL
    else:
        if not config.sources.get("eastmoney", True):
            raise RuntimeError("corporate_actions daily: eastmoney source disabled in config")
        df, findings = fetch_incremental_daily(
            config,
            "corporate_actions",
            trade_date,
            lambda d: fetch_corporate_actions_eastmoney(d, backfill=False, config=config),
            allow_empty=True,
            date_col="ex_date",
        )
        canonical_source = _CANONICAL_DAILY
        if config.failover_enabled and df.height:
            ex_today = df.filter(pl.col("ex_date") == trade_date)
            if ex_today.height:
                snapshot_corporate_actions_tdx_backup(
                    config,
                    trade_date=trade_date,
                    symbols=ex_today["symbol"].unique().to_list(),
                    run_id=run_id,
                    rate_limit=rl,
                )

    if backfill and not df.is_empty():
        start = getattr(config, "_backfill_start", None) or BACKFILL_START
        end = getattr(config, "_backfill_end", None) or trade_date
        if "ex_date" not in df.columns:
            raise RuntimeError("corporate_actions: backfill response has no ex_date column")
        parsed_dates = df.get_column("ex_date").cast(pl.Date, strict=False)
        invalid = (
            parsed_dates.is_null()
            | (parsed_dates < start).fill_null(False)
            | (parsed_dates > end).fill_null(False)
        )
        if int(invalid.sum()):
            raise RuntimeError(
                "corporate_actions: backfill response returned row(s) outside "
                f"requested window {start.isoformat()}..{end.isoformat()}"
            )
        df = df.with_columns(parsed_dates.alias("ex_date"))

    context_updates: dict = {"symbols_to_rebackfill": []}
    if findings:
        context_updates["audit_findings"] = findings
    if df.is_empty():
        result = {"rows_read": 0, "rows_written": 0, "context_updates": context_updates}
        if backfill and failed_symbols:
            result["failed_symbols"] = failed_symbols
            result["status"] = "failed"
        return result

    df = with_provenance(df, source=canonical_source, data_version="v1")

    rebackfill: list[str] = []
    if df.height and "symbol" in df.columns and "ex_date" in df.columns:
        today = df.filter(pl.col("ex_date") == trade_date)
        if today.height:
            rebackfill = today["symbol"].unique().to_list()

    context_updates["symbols_to_rebackfill"] = rebackfill
    if backfill and manifest is not None:
        result = {
            "rows_read": df.height,
            "rows_written": df.height,
        }
    else:
        result = write_simple(config, run_id, "corporate_actions", df)
    if backfill and failed_symbols:
        result["failed_symbols"] = failed_symbols
        result["status"] = "failed"
    result["context_updates"] = context_updates
    return result


@register_step("earnings_disclosure_schedule", group="fundamentals", depends_on=["instruments"])
def step_earnings_disclosure_schedule(
    config: Config, trade_date: date, run_id: str, context: dict
) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("earnings_disclosure_schedule: eastmoney source disabled in config")
    # Period-keyed like financial_statement_items (watermark=False): daily runs
    # refresh the open disclosure windows; backfill walks every period 2016+.
    backfill = getattr(config, "_backfill", False)
    df = _validate_earnings_schedule_snapshot(
        fetch_earnings_disclosure_schedule(trade_date, backfill=backfill, config=config)
    )
    missing_periods: set[str] = set()
    if backfill:
        expected = {
            f"{period[:4]}Q{(int(period[5:7]) - 1) // 3 + 1}"
            for period in _backfill_report_dates(
                trade_date,
                start=getattr(config, "_backfill_start", None),
                end=getattr(config, "_backfill_end", None),
            )
        }
        observed = (
            set(df.get_column("report_period").drop_nulls().to_list())
            if not df.is_empty() and "report_period" in df.columns
            else set()
        )
        missing_periods = expected - observed
    if backfill and missing_periods:
        result: dict
        if df.is_empty():
            result = {"rows_read": 0, "rows_written": 0}
        else:
            result = write_fetched(
                config, run_id, "earnings_disclosure_schedule", df, source="eastmoney"
            )
        result["status"] = "warning"
        result["missing_periods"] = len(missing_periods)
        result["context_updates"] = {
            "audit_findings": [
                {
                    "dataset": "earnings_disclosure_schedule",
                    "severity": "warning",
                    "check": "backfill_missing_report_periods",
                    "message": (
                        f"earnings disclosure schedule missing {len(missing_periods)} "
                        f"requested report period(s): {', '.join(sorted(missing_periods)[:8])}"
                    ),
                    "missing_periods": sorted(missing_periods),
                }
            ]
        }
        return result
    if df.is_empty():
        return {"rows_read": 0, "rows_written": 0}
    return write_fetched(config, run_id, "earnings_disclosure_schedule", df, source="eastmoney")


@register_step("announcement_index", group="capital", depends_on=["instruments"])
def step_announcement_index(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if not config.sources.get("cninfo", True):
        raise RuntimeError("announcement_index: cninfo source disabled in config")
    if getattr(config, "_backfill", False):
        from cnequity.steps.common import walk_day_backfill

        return walk_day_backfill(
            config,
            trade_date,
            run_id,
            "announcement_index",
            lambda d: fetch_announcement_index(d, config=config),
            source="cninfo",
            date_col="announce_date",
            floor=date(2010, 1, 1),
        )
    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "announcement_index",
        lambda d: fetch_announcement_index(d, config=config),
        source="cninfo",
        date_col="announce_date",
    )
