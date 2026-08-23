"""Maintenance steps for externally-owned datasets."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from cnequity.config import Config
from cnequity.external import tushare_fetch
from cnequity.external.tushare_wide import ADAPTER as _tushare_wide_adapter
from cnequity.orchestrator.registry import register_step
from cnequity.storage.state import StateStore

logger = logging.getLogger(__name__)


# The local_assets_daily step is deprecated. ETF and fund bars are now fetched
# directly by the ``fund_bars`` step (registered in ``bars.py``) via Tushare
# ``fund_daily`` and written to curated ``etf_bars`` / ``fund_bars`` datasets.
# The step is kept for backward compatibility but is a no-op: it short-circuits
# with a warning when invoked.
@register_step(
    "local_assets_daily",
    group="assets",
    parallelizable=False,
    description="[DEPRECATED] No-op. ETF/fund bars now served by fund_bars step via Tushare fund_daily.",
)
def step_local_assets_daily(
    config: Config,
    trade_date: date,
    run_id: str,
    context: dict,
) -> dict:
    """Deprecated: ETF/fund bars are now natively managed by CNE (see fund_bars step)."""
    return {
        "status": "warning",
        "dataset": "local_assets",
        "error": "local_assets_daily is deprecated; etf_bars/fund_bars served by fund_bars step",
    }


@register_step(
    "external_tushare_wide_status",
    group="external",
    parallelizable=False,
    description="Refresh coverage and watermark for the external Tushare-wide daily archive.",
)
def step_external_tushare_wide_status(
    config: Config,
    trade_date: date,
    run_id: str,
    context: dict,
) -> dict:
    """Register the archive's current coverage without copying or mutating it."""
    if not _tushare_wide_adapter.enabled(config, "daily_bars"):
        return {
            "status": "warning",
            "dataset": "daily_bars",
            "error": "external_tushare_wide is disabled",
        }
    start, end = _tushare_wide_adapter.coverage_bounds(config, "daily_bars")
    if start is None or end is None:
        return {
            "status": "warning",
            "dataset": "daily_bars",
            "error": "external Tushare-wide archive has no readable daily bars",
        }
    # Use set_date (overwrite) rather than update_max_date (monotonic max).
    # This step reflects the archive's *actual* coverage end, which can shrink
    # if files are repaired or pruned. update_max_date would ignore a receded
    # boundary and leave a stale watermark that blocks gap detection.
    StateStore(config.meta_root).set_date("daily_bars", end)
    return {
        "dataset": "daily_bars",
        "rows_read": 0,
        "rows_written": 0,
        "context_updates": {
            "external_tushare_wide": {
                "run_id": run_id,
                "coverage_start": start.isoformat(),
                "coverage_end": end.isoformat(),
                "observed_at": trade_date.isoformat(),
            }
        },
    }


@register_step(
    "tushare_wide_daily",
    group="core",
    depends_on=["external_tushare_wide_status"],
    parallelizable=False,
    description="Fetch daily bars + daily_basic + adj_factor from Tushare, merge into yearly archive.",
)
def step_tushare_wide_daily(
    config: Config,
    trade_date: date,
    run_id: str,
    context: dict,
) -> dict:
    """Fetch missing trade dates from Tushare and atomically merge into the archive.

    This step:
    1. Reads the watermark left by ``external_tushare_wide_status``.
    2. Identifies missing trading days between watermark+1 and ``trade_date``.
    3. Fetches each missing day (daily + daily_basic + adj_factor) from Tushare.
    4. Atomically merges the result into the corresponding yearly parquet file.
    5. Advances the watermark only if every missing day succeeded.
    """
    if not _tushare_wide_adapter.enabled(config, "daily_bars"):
        return {
            "status": "warning",
            "dataset": "daily_bars",
            "error": "external_tushare_wide is disabled",
        }

    if not config.external_tushare_wide_token and not os.environ.get("TUSHARE_TOKEN"):
        return {
            "status": "warning",
            "dataset": "daily_bars",
            "error": "tushare_token not configured; skipping daily fetch",
        }

    # The status step ran before us and set the watermark to the archive's
    # actual coverage end. We need the day after that.
    state = StateStore(config.meta_root)
    watermark = state.get_date("daily_bars")

    # Determine the fetch window.
    if getattr(config, "_backfill", False):
        start = getattr(config, "_backfill_start", None)
        end = getattr(config, "_backfill_end", None) or trade_date
        if start is None:
            start = watermark + timedelta(days=1) if watermark else trade_date - timedelta(days=5)
        else:
            start = start if isinstance(start, date) else date.fromisoformat(str(start))
    else:
        if watermark is None:
            # No archived data at all — start from a few days back.
            start = trade_date - timedelta(days=5)
        else:
            start = watermark + timedelta(days=1)
        end = trade_date

    if start > end:
        logger.info(
            "tushare_wide_daily: already up to date (watermark=%s, trade_date=%s)",
            watermark,
            trade_date,
        )
        return {
            "dataset": "daily_bars",
            "rows_read": 0,
            "rows_written": 0,
        }

    # Enumerate missing trading days. Use CNE's built-in calendar; if it
    # fails, fall back to weekdays (rare — the seed calendar covers 2026).
    from cnequity.steps.common import list_trading_dates

    try:
        missing_dates = list_trading_dates(config, start, end)
    except Exception as exc:
        logger.warning("list_trading_dates failed (%s); falling back to weekdays", exc)
        missing_dates = []
        cur = start
        while cur <= end:
            if cur.weekday() < 5:  # Mon-Fri
                missing_dates.append(cur)
            cur += timedelta(days=1)

    if not missing_dates:
        logger.info("tushare_wide_daily: no trading days in [%s, %s]", start, end)
        return {
            "dataset": "daily_bars",
            "rows_read": 0,
            "rows_written": 0,
        }

    logger.info(
        "tushare_wide_daily: %d missing trade date(s) to fetch: %s ~ %s",
        len(missing_dates),
        missing_dates[0],
        missing_dates[-1],
    )

    total_read = 0
    total_written = 0
    failed_dates: list[str] = []
    last_success_date: date | None = None

    for d in missing_dates:
        result = tushare_fetch.fetch_and_merge_one_date(config, d)
        if result.get("status") == "success":
            total_read += result["rows_read"]
            total_written += result["rows_written"]
            last_success_date = d
        else:
            failed_dates.append(d.isoformat())
            logger.error(
                "tushare_wide_daily %s failed: %s",
                d,
                result.get("error", "unknown"),
            )

    # Advance watermark only to the last *contiguous* success. If days fail
    # in the middle, the watermark stops at the gap so the next run retries.
    if last_success_date is not None and not failed_dates:
        state.set_date("daily_bars", last_success_date)
    elif last_success_date is not None:
        # Partial success: advance watermark to the last date before the
        # first failure, so we don't skip gaps on the next run.
        contiguous_end = last_success_date
        for d in missing_dates:
            if d in [date.fromisoformat(x) for x in failed_dates]:
                break
            contiguous_end = d
        if contiguous_end > (watermark or date.min):
            state.set_date("daily_bars", contiguous_end)

    status = "success" if not failed_dates else "warning"
    result = {
        "dataset": "daily_bars",
        "rows_read": total_read,
        "rows_written": total_written,
        "context_updates": {
            "external_tushare_wide": {
                "run_id": run_id,
                "fetched_dates": len(missing_dates) - len(failed_dates),
                "failed_dates": failed_dates,
                "last_success": last_success_date.isoformat() if last_success_date else None,
            }
        },
    }
    if failed_dates:
        result["error"] = f"{len(failed_dates)} trade date(s) failed: {failed_dates[:5]}"
    return {"status": status, **result}
