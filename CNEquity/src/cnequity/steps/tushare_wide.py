"""Tushare wide-table steps: financial statements, corporate events, surveys.

Three fetch strategies, chosen per-dataset based on Tushare middleware
(``t.xiaodefa.top``) API capabilities — eliminates the 7-hour full-history
refetch problem:

1. **Full-market ann_date batch** (dividend, share_float, stk_surv):
   The middleware accepts ``ann_date`` without ``ts_code``, returning all rows
   market-wide for that date.  We iterate calendar days (watermark+1 →
   trade_date) — O(N_days) calls instead of O(N_stocks).

2. **Full-market date-range batch** (namechange):
   The middleware accepts ``start_date``/``end_date`` without ``ts_code``,
   returning all name changes in the window — a single call.

3. **Per-stock incremental** (balancesheet, income, cashflow, fina_indicator,
   forecast, express):
   The middleware requires ``ts_code``, but ``start_date``/``end_date``
   filters to only rows announced in the window.  We pass watermark+1 →
   trade_date so each stock returns only new rows (0–6 vs 100+ full).
   On a normal day most stocks return 0 rows, making the 7780-call sweep
   fast (~4 min/dataset at 10 workers).

All schemas are vendor-controlled (Tushare may add/remove columns), so they
are deliberately NOT registered in ``DATASET_SCHEMAS`` —
``validate_dataframe`` passes them through untouched.  Primary keys are
registered in ``PRIMARY_KEYS`` for compact dedup.  Provenance columns
(``source``, ``data_version``, ``fetched_at``) are added manually.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Callable

import polars as pl

from cnequity.config import Config
from cnequity.domain.datasets import should_fetch
from cnequity.domain.schemas import data_version_for, with_provenance
from cnequity.orchestrator.registry import register_step
from cnequity.storage import StagingWriter
from cnequity.storage.state import StateStore

logger = logging.getLogger(__name__)

# Concurrency for per-stock fetches. The Tushare middleware (xiaodefa.top)
# supports up to ~10 concurrent requests without triggering rate limits.
_DEFAULT_WORKERS = 10


# ─── Helpers ──────────────────────────────────────────────────────────


def _get_pro(config: Config):
    """Return a Tushare pro API client (reuses tushare_fetch helpers)."""
    from cnequity.external.tushare_fetch import _get_pro

    return _get_pro(config)


def _ts_date(d: date) -> str:
    """Format a date as Tushare expects: YYYYMMDD."""
    return d.strftime("%Y%m%d")


def _norm_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Cast all YYYYMMDD string date columns to pl.Date."""
    date_cols = []
    for col in df.columns:
        if col == "ts_code" or col == "symbol":
            continue
        if "date" in col.lower():
            date_cols.append(col)
    if not date_cols:
        return df
    exprs = []
    for col in date_cols:
        if df.schema[col] == pl.Utf8:
            exprs.append(
                pl.col(col).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias(col)
            )
        elif df.schema[col] == pl.Date:
            pass
        else:
            exprs.append(
                pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias(col)
            )
    if exprs:
        df = df.with_columns(exprs)
    return df


def _load_symbols(config: Config) -> list[str]:
    from cnequity.steps.common import load_symbols

    return load_symbols(config)


def _finalize_and_write(
    combined: pl.DataFrame,
    dataset: str,
    run_id: str,
    trade_date: date,
    config: Config,
) -> dict:
    """Common post-processing: rename, drop id, norm dates, provenance, write."""
    if combined.is_empty():
        logger.warning("%s: no data fetched", dataset)
        return {"rows_read": 0, "rows_written": 0}

    if "ts_code" in combined.columns:
        combined = combined.rename({"ts_code": "symbol"})
    if "id" in combined.columns:
        combined = combined.drop("id")

    combined = _norm_dates(combined)
    combined = with_provenance(
        combined, source="tushare", data_version=data_version_for(dataset)
    )

    writer = StagingWriter(config.staging_root)
    writer.write_batch(dataset, run_id, f"tushare-{trade_date.isoformat()}", combined)

    logger.info("%s: %d rows written to staging", dataset, combined.height)
    return {"rows_read": combined.height, "rows_written": combined.height}


def _calendar_days(start: date, end: date) -> list[date]:
    """All calendar days from start to end inclusive."""
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


# ─── Strategy 1: Full-market ann_date batch ───────────────────────────


def _make_ann_date_step(
    dataset: str,
    api: str,
    depends_on: list[str],
    group: str,
) -> Callable:
    """Fetch all rows for each calendar day via ``pro.api(ann_date=...)``.

    Iterates from watermark+1 to trade_date, one API call per day.
    Replaces 7780 per-stock calls with ≤ a few dozen day-level calls.
    """

    @register_step(dataset, group=group, depends_on=depends_on, parallelizable=False)
    def step(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
        if not config.external_tushare_wide_token and not os.environ.get("TUSHARE_TOKEN"):
            return {"status": "warning", "dataset": dataset, "error": "tushare_token not configured"}

        from cnequity.external.tushare_fetch import _fetch_with_retry

        state = StateStore(config.meta_root)
        watermark = state.get_date(dataset)
        if not should_fetch(dataset, watermark, trade_date):
            logger.info(
                "%s: cadence skip (watermark %s in same period as trade_date %s)",
                dataset,
                watermark.isoformat() if watermark else "None",
                trade_date.isoformat(),
            )
            state.set_date(dataset, trade_date)
            return {"rows_read": 0, "rows_written": 0, "status": "success"}

        start = watermark + timedelta(days=1) if watermark else trade_date - timedelta(days=7)
        days = _calendar_days(start, trade_date)

        logger.info(
            "%s: fetching ann_date %s..%s (%d days)",
            dataset, start.isoformat(), trade_date.isoformat(), len(days),
        )

        pro = _get_pro(config)
        interval = config.external_tushare_wide_interval
        frames: list[pl.DataFrame] = []

        for d in days:
            raw = _fetch_with_retry(pro, api, interval=interval, ann_date=_ts_date(d))
            if raw is not None and not raw.is_empty():
                frames.append(raw)

        combined = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
        result = _finalize_and_write(combined, dataset, run_id, trade_date, config)

        state.set_date(dataset, trade_date)
        logger.info("%s: watermark → %s", dataset, trade_date.isoformat())
        return result

    return step


# ─── Strategy 2: Full-market date-range batch ─────────────────────────


def _make_date_range_step(
    dataset: str,
    api: str,
    depends_on: list[str],
    group: str,
) -> Callable:
    """Fetch all rows in a date window via ``pro.api(start_date=..., end_date=...)``.

    Single API call for the entire incremental window.  Used for APIs that
    support full-market date-range queries (e.g. namechange).
    """

    @register_step(dataset, group=group, depends_on=depends_on, parallelizable=False)
    def step(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
        if not config.external_tushare_wide_token and not os.environ.get("TUSHARE_TOKEN"):
            return {"status": "warning", "dataset": dataset, "error": "tushare_token not configured"}

        from cnequity.external.tushare_fetch import _fetch_with_retry

        state = StateStore(config.meta_root)
        watermark = state.get_date(dataset)
        if not should_fetch(dataset, watermark, trade_date):
            logger.info(
                "%s: cadence skip (watermark %s in same period as trade_date %s)",
                dataset,
                watermark.isoformat() if watermark else "None",
                trade_date.isoformat(),
            )
            state.set_date(dataset, trade_date)
            return {"rows_read": 0, "rows_written": 0, "status": "success"}

        start = watermark + timedelta(days=1) if watermark else trade_date - timedelta(days=30)
        end = trade_date

        logger.info(
            "%s: fetching date-range %s..%s",
            dataset, start.isoformat(), end.isoformat(),
        )

        pro = _get_pro(config)
        interval = config.external_tushare_wide_interval
        raw = _fetch_with_retry(
            pro, api, interval=interval,
            start_date=_ts_date(start), end_date=_ts_date(end),
        )

        combined = raw if raw is not None else pl.DataFrame()
        result = _finalize_and_write(combined, dataset, run_id, trade_date, config)

        state.set_date(dataset, trade_date)
        logger.info("%s: watermark → %s", dataset, trade_date.isoformat())
        return result

    return step


# ─── Strategy 3: Per-stock incremental ────────────────────────────────


def _make_incremental_step(
    dataset: str,
    api: str,
    depends_on: list[str],
    group: str,
    initial_lookback_days: int = 30,
    max_workers: int | None = None,
) -> Callable:
    """Fetch per-stock with ``start_date``/``end_date`` incremental filter.

    On first run (no watermark), we use a lookback window as ``start_date``
    instead of an unbounded full fetch.  Historical data is already in the
    curated layer from the one-time import script.

    ``max_workers`` overrides the default concurrency (_DEFAULT_WORKERS=10).
    Use a lower value (e.g. 3) for APIs that are more aggressively rate-
    limited by the Tushare middleware (express, forecast).
    """

    @register_step(dataset, group=group, depends_on=depends_on, parallelizable=False)
    def step(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
        if not config.external_tushare_wide_token and not os.environ.get("TUSHARE_TOKEN"):
            return {"status": "warning", "dataset": dataset, "error": "tushare_token not configured"}

        symbols = _load_symbols(config)
        if not symbols:
            logger.warning("%s: no symbols loaded (run instruments step first)", dataset)
            return {"rows_read": 0, "rows_written": 0}

        state = StateStore(config.meta_root)
        watermark = state.get_date(dataset)

        # Cadence check: skip the per-stock sweep entirely when the dataset's
        # publication period has not changed since the watermark.
        if not should_fetch(dataset, watermark, trade_date):
            logger.info(
                "%s: cadence skip (watermark %s in same period as trade_date %s)",
                dataset,
                watermark.isoformat() if watermark else "None",
                trade_date.isoformat(),
            )
            state.set_date(dataset, trade_date)
            return {"rows_read": 0, "rows_written": 0, "status": "success"}

        if watermark:
            start_str = _ts_date(watermark + timedelta(days=1))
            mode = "incremental"
        else:
            # First run: use a lookback window instead of unbounded full
            # history.  Historical data is already in curated from the
            # migration script; we only need recent rows going forward.
            start_str = _ts_date(trade_date - timedelta(days=initial_lookback_days))
            mode = f"initial lookback ({initial_lookback_days}d)"
        end_str = _ts_date(trade_date)

        # If the waterfall start is already past trade_date (watermark is at or
        # after trade_date), there is nothing to fetch — skip the entire per-
        # stock sweep to avoid 7780 pointless API calls that trigger rate
        # limiting on the middleware for zero rows.
        if start_str > end_str:
            logger.info(
                "%s: watermark >= trade_date, nothing to fetch (skip %d symbols)",
                dataset, len(symbols),
            )
            state.set_date(dataset, trade_date)
            logger.info("%s: watermark → %s (up-to-date)", dataset, trade_date.isoformat())
            return {"rows_read": 0, "rows_written": 0, "status": "success"}

        logger.info(
            "%s: %s %s..%s for %d symbols",
            dataset, mode, start_str, end_str, len(symbols),
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from cnequity.external.tushare_fetch import _fetch_with_retry

        pro = _get_pro(config)
        interval = config.external_tushare_wide_interval
        frames: list[pl.DataFrame] = []
        done = 0
        workers = min(max_workers or _DEFAULT_WORKERS, len(symbols))

        def _fetch_one(ts_code: str) -> pl.DataFrame:
            kwargs: dict = {"ts_code": ts_code, "start_date": start_str, "end_date": end_str}
            try:
                return _fetch_with_retry(pro, api, interval=interval, **kwargs)
            except Exception as exc:
                logger.warning("%s: %s failed: %s", dataset, ts_code, exc)
                return pl.DataFrame()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fetch_one, s): s for s in symbols}
            for fut in as_completed(futs):
                code = futs[fut]
                try:
                    sub = fut.result()
                except Exception as exc:
                    logger.warning("%s: %s failed: %s", dataset, code, exc)
                    continue
                if sub is not None and not sub.is_empty():
                    frames.append(sub)
                done += 1
                if done % 1000 == 0 or done == len(symbols):
                    logger.info("%s: progress %d/%d", dataset, done, len(symbols))

        combined = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
        result = _finalize_and_write(combined, dataset, run_id, trade_date, config)

        state.set_date(dataset, trade_date)
        logger.info("%s: watermark → %s", dataset, trade_date.isoformat())
        return result

    return step


# ─── Step registrations ───────────────────────────────────────────────

# Skip-step factory (defined early so all skip registrations can use it).
# Advances watermark without fetching — for datasets where the source is
# unreliable (EastMoney) or the middleware doesn't support efficient queries
# (express/forecast).  Historical data is in the curated layer.
# The cadence field on DatasetSpec controls whether this step actually skips:
# cadence="skip" → always skip; cadence="quarterly" → skip within the same
# quarter, fetch (via the skip-step's no-op) when the quarter changes.
def _make_skip_step(dataset: str, group: str, depends_on: list[str]) -> Callable:
    @register_step(dataset, group=group, depends_on=depends_on, parallelizable=False)
    def step(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
        state = StateStore(config.meta_root)
        wm = state.get_date(dataset)
        if not should_fetch(dataset, wm, trade_date):
            if wm is None:
                logger.info("%s: skipping (historical data in curated)", dataset)
            else:
                logger.info("%s: skipping (watermark already %s)", dataset, wm.isoformat())
            state.set_date(dataset, trade_date)
            logger.info("%s: watermark → %s (skip)", dataset, trade_date.isoformat())
            return {"rows_read": 0, "rows_written": 0, "status": "success"}
        # Cadence says fetch, but skip-step has no fetch implementation —
        # still advance the watermark.  This covers EastMoney datasets with
        # cadence="quarterly" where the source is unreliable: the first run
        # of a new quarter advances the watermark without fetching.
        logger.info("%s: cadence fetch due, but skip-step (source unreliable)", dataset)
        state.set_date(dataset, trade_date)
        logger.info("%s: watermark → %s (skip-step)", dataset, trade_date.isoformat())
        return {"rows_read": 0, "rows_written": 0, "status": "success"}
    return step

# L3 fundamentals — per-stock incremental (serial wave, 5 workers to respect
# middleware rate limits; 4 datasets × 10 workers = 40 concurrent was too aggressive)
_make_incremental_step("balancesheet",   api="balancesheet",   depends_on=["instruments"], group="fundamentals", max_workers=5)
_make_incremental_step("income",         api="income",         depends_on=["instruments"], group="fundamentals", max_workers=5)
_make_incremental_step("cashflow",       api="cashflow",       depends_on=["instruments"], group="fundamentals", max_workers=5)
_make_incremental_step("fina_indicator", api="fina_indicator", depends_on=["instruments"], group="fundamentals", max_workers=5)

# EastMoney fundamentals — skip (source unreliable, historical data in curated)
# financial_statement_items, earnings_disclosure_schedule, share_structure,
# shareholder_counts, top_holders all fetch from EastMoney on every run (no
# watermark check).  EastMoney frequently times out or returns partial data.
# Historical data is already in the curated layer (cover to 2026-08-22).
# These steps advance the watermark without fetching; run `cne backfill` for
# a full refresh when needed.
_make_skip_step("financial_statement_items", group="fundamentals", depends_on=["instruments"])
_make_skip_step("earnings_disclosure_schedule", group="fundamentals", depends_on=["instruments"])
_make_skip_step("share_structure", group="fundamentals", depends_on=["instruments"])
_make_skip_step("shareholder_counts", group="fundamentals", depends_on=["instruments"])
_make_skip_step("top_holders", group="fundamentals", depends_on=["instruments"])

# L2/L8 corporate events — full-market ann_date batch
_make_ann_date_step("dividend",            api="dividend",      depends_on=["instruments"], group="events")
_make_ann_date_step("share_float_external", api="share_float",  depends_on=["instruments"], group="events")
_make_ann_date_step("stk_surv",            api="stk_surv",      depends_on=["instruments"], group="events")

# L2 corporate events — full-market date-range batch
_make_date_range_step("namechange", api="namechange", depends_on=["instruments"], group="events")

# L3 earnings forecast / express — skip (middleware limitation)
# The Tushare middleware (t.xiaodefa.top) does not support efficient
# querying for these APIs:
#   - ann_date without ts_code → "必填参数, 标的" (rejected)
#   - ts_code + start_date/end_date → 0 rows (date filter ignored)
#   - ts_code only → full history (7780 calls × slow = 13+ min)
# Historical data is already in the curated layer from the import script.
_make_skip_step("forecast", group="events", depends_on=["instruments"])
_make_skip_step("express",  group="events", depends_on=["instruments"])


# ─── Snapshot step (report_rc) ────────────────────────────────────────

@register_step("report_rc", group="fundamentals", depends_on=["instruments"], parallelizable=False)
def step_report_rc(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    """Fetch report_rc (一致预期研究报告) — single full-market API call."""
    if not config.external_tushare_wide_token and not os.environ.get("TUSHARE_TOKEN"):
        return {"status": "warning", "dataset": "report_rc", "error": "tushare_token not configured"}

    state = StateStore(config.meta_root)
    watermark = state.get_date("report_rc")
    if not should_fetch("report_rc", watermark, trade_date):
        logger.info(
            "report_rc: cadence skip (watermark %s in same period as trade_date %s)",
            watermark.isoformat() if watermark else "None",
            trade_date.isoformat(),
        )
        state.set_date("report_rc", trade_date)
        return {"rows_read": 0, "rows_written": 0, "status": "success"}

    from cnequity.external.tushare_fetch import _fetch_with_retry

    pro = _get_pro(config)
    interval = config.external_tushare_wide_interval

    logger.info("report_rc: fetching full-market snapshot via Tushare")
    raw = _fetch_with_retry(pro, "report_rc", interval=interval)
    if raw.is_empty():
        logger.warning("report_rc: no data returned")
        return {"rows_read": 0, "rows_written": 0}

    if "ts_code" in raw.columns:
        raw = raw.rename({"ts_code": "symbol"})
    if "id" in raw.columns:
        raw = raw.drop("id")

    raw = _norm_dates(raw)
    raw = raw.unique(
        subset=["symbol", "report_date", "org_name", "report_title"],
        keep="last",
    )
    raw = with_provenance(raw, source="tushare", data_version=data_version_for("report_rc"))

    writer = StagingWriter(config.staging_root)
    writer.write_batch("report_rc", run_id, f"tushare-{trade_date.isoformat()}", raw)

    state.set_date("report_rc", trade_date)
    logger.info("report_rc: %d rows written to staging, watermark → %s", raw.height, trade_date.isoformat())
    return {"rows_read": raw.height, "rows_written": raw.height}
