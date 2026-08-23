"""Tushare wide-table steps: financial statements, corporate events, surveys.

These datasets were previously synced by ``scripts/sync_tushare_to_parquet.py``
to ``data/pg_parquet/`` and mounted read-only via the ``pg_parquet`` external
adapter.  Now they are curated datasets fetched directly inside the CNE daily
run, so the entire update pipeline goes through ``cne run daily``.

Two patterns:

1. **Per-stock** (10 datasets): fetch one ``ts_code`` at a time from the
   Tushare API, concat, rename ``ts_code`` → ``symbol``, add provenance, write
   to staging.  The generic ``_make_per_stock_step`` factory builds the step
   function for each dataset.

2. **Snapshot** (report_rc): single API call returns all rows; full overwrite
   each run.

All schemas are vendor-controlled (Tushare may add/remove columns), so they
are deliberately NOT registered in ``DATASET_SCHEMAS`` —
``validate_dataframe`` passes them through untouched.  Primary keys are
registered in ``PRIMARY_KEYS`` for compact dedup.  Provenance columns
(``source``, ``data_version``, ``fetched_at``) are added manually.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Callable

import polars as pl

from cnequity.config import Config
from cnequity.domain.schemas import data_version_for, with_provenance
from cnequity.orchestrator.registry import register_step
from cnequity.storage import StagingWriter

logger = logging.getLogger(__name__)

# Concurrency for per-stock fetches. The Tushare proxy rate-limits hard, so we
# keep this modest — the old script used 3 workers and that worked.
_DEFAULT_WORKERS = 3


def _get_pro(config: Config):
    """Return a Tushare pro API client (reuses tushare_fetch helpers)."""
    from cnequity.external.tushare_fetch import _get_pro

    return _get_pro(config)


def _fetch_one_stock(pro, config: Config, api: str, ts_code: str) -> pl.DataFrame:
    """Fetch one Tushare API for one stock code, returning a polars DataFrame."""
    from cnequity.external.tushare_fetch import _fetch_with_retry

    interval = config.external_tushare_wide_interval
    try:
        return _fetch_with_retry(pro, api, interval=interval, ts_code=ts_code)
    except Exception as exc:
        logger.warning("Tushare %s failed for %s: %s", api, ts_code, exc)
        return pl.DataFrame()


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
            pass  # already Date
        else:
            # Tushare sometimes returns dates as integers (YYYYMMDD as int)
            exprs.append(
                pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias(col)
            )
    if exprs:
        df = df.with_columns(exprs)
    return df


def _load_symbols(config: Config) -> list[str]:
    """Load all stock ts_codes from curated instruments."""
    from cnequity.steps.common import load_symbols

    return load_symbols(config)


def _make_per_stock_step(
    dataset: str,
    api: str,
    depends_on: list[str],
    group: str = "fundamentals",
) -> Callable:
    """Build a step function that fetches one Tushare API per stock.

    The resulting function:
    1. Loads the stock universe from curated instruments.
    2. Calls ``pro.<api>(ts_code=<code>)`` for each stock concurrently.
    3. Concats, renames ts_code → symbol, normalizes dates, adds provenance.
    4. Writes a single batch to staging.
    """
    @register_step(dataset, group=group, depends_on=depends_on, parallelizable=False)
    def step(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
        if not config.external_tushare_wide_token and not os.environ.get("TUSHARE_TOKEN"):
            return {
                "status": "warning",
                "dataset": dataset,
                "error": "tushare_token not configured",
            }

        # Backfill mode: fetch all history per stock (Tushare's default
        # behaviour — no start/end date means all periods).
        # Daily mode: same — Tushare per-stock APIs return the full history
        # for that stock. Watermark-based incremental is handled by compact
        # dedup (only new/changed rows survive PK unique keep=last).
        symbols = _load_symbols(config)
        if not symbols:
            logger.warning("%s: no symbols loaded (run instruments step first)", dataset)
            return {"rows_read": 0, "rows_written": 0}

        logger.info(
            "%s: fetching %s for %d symbols via Tushare",
            dataset, api, len(symbols),
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        pro = _get_pro(config)
        frames: list[pl.DataFrame] = []
        done = 0
        workers = min(_DEFAULT_WORKERS, len(symbols))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_fetch_one_stock, pro, config, api, s): s
                for s in symbols
            }
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
                if done % 500 == 0 or done == len(symbols):
                    logger.info("%s: progress %d/%d", dataset, done, len(symbols))

        if not frames:
            logger.warning("%s: no data fetched", dataset)
            return {"rows_read": 0, "rows_written": 0}

        combined = pl.concat(frames, how="diagonal_relaxed")

        # Rename ts_code → symbol for CNE compatibility
        if "ts_code" in combined.columns:
            combined = combined.rename({"ts_code": "symbol"})

        # Drop the 'id' column if present (Tushare returns it but it's
        # meaningless as a unique key across runs).
        if "id" in combined.columns:
            combined = combined.drop("id")

        # Normalize date columns from YYYYMMDD strings to pl.Date
        combined = _norm_dates(combined)

        # Add provenance
        combined = with_provenance(
            combined, source="tushare", data_version=data_version_for(dataset)
        )

        # Write to staging as a single batch
        writer = StagingWriter(config.staging_root)
        writer.write_batch(dataset, run_id, f"tushare-{trade_date.isoformat()}", combined)

        logger.info(
            "%s: %d rows written to staging (%d symbols)",
            dataset, combined.height, len(symbols),
        )

        return {
            "rows_read": combined.height,
            "rows_written": combined.height,
        }

    return step


# ─── Per-stock steps (10 datasets) ────────────────────────────────────

# L3 fundamentals
_make_per_stock_step(
    "balancesheet", api="balancesheet",
    depends_on=["instruments"], group="fundamentals",
)
_make_per_stock_step(
    "income", api="income",
    depends_on=["instruments"], group="fundamentals",
)
_make_per_stock_step(
    "cashflow", api="cashflow",
    depends_on=["instruments"], group="fundamentals",
)
_make_per_stock_step(
    "fina_indicator", api="fina_indicator",
    depends_on=["instruments"], group="fundamentals",
)

# L2 corporate events
_make_per_stock_step(
    "dividend", api="dividend",
    depends_on=["instruments"], group="events",
)
_make_per_stock_step(
    "share_float_external", api="share_float",
    depends_on=["instruments"], group="events",
)
_make_per_stock_step(
    "namechange", api="namechange",
    depends_on=["instruments"], group="events",
)

# L3/L7 earnings forecast / express / survey
_make_per_stock_step(
    "forecast", api="forecast",
    depends_on=["instruments"], group="events",
)
_make_per_stock_step(
    "express", api="express",
    depends_on=["instruments"], group="events",
)
_make_per_stock_step(
    "stk_surv", api="stk_surv",
    depends_on=["instruments"], group="events",
)


# ─── Snapshot step (report_rc) ────────────────────────────────────────

@register_step(
    "report_rc",
    group="fundamentals",
    depends_on=["instruments"],
    parallelizable=False,
)
def step_report_rc(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    """Fetch report_rc (一致预期研究报告) — single full-market API call."""
    if not config.external_tushare_wide_token and not os.environ.get("TUSHARE_TOKEN"):
        return {
            "status": "warning",
            "dataset": "report_rc",
            "error": "tushare_token not configured",
        }

    from cnequity.external.tushare_fetch import _fetch_with_retry

    pro = _get_pro(config)
    interval = config.external_tushare_wide_interval

    logger.info("report_rc: fetching full-market snapshot via Tushare")
    raw = _fetch_with_retry(pro, "report_rc", interval=interval)
    if raw.is_empty():
        logger.warning("report_rc: no data returned")
        return {"rows_read": 0, "rows_written": 0}

    # Rename ts_code → symbol, drop id
    if "ts_code" in raw.columns:
        raw = raw.rename({"ts_code": "symbol"})
    if "id" in raw.columns:
        raw = raw.drop("id")

    # Normalize dates
    raw = _norm_dates(raw)

    # Deduplicate by PK
    raw = raw.unique(
        subset=["symbol", "report_date", "org_name", "report_title"],
        keep="last",
    )

    # Add provenance
    raw = with_provenance(raw, source="tushare", data_version=data_version_for("report_rc"))

    writer = StagingWriter(config.staging_root)
    writer.write_batch("report_rc", run_id, f"tushare-{trade_date.isoformat()}", raw)

    logger.info("report_rc: %d rows written to staging", raw.height)

    return {
        "rows_read": raw.height,
        "rows_written": raw.height,
    }
