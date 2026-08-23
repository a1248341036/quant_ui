"""Tushare daily fetcher for the external wide-table archive.

Fetches ``daily`` + ``daily_basic`` + ``adj_factor`` from Tushare for a single
trade date and atomically merges the result into the corresponding yearly
``stock_daily.parquet`` file under ``<root>/<year>/<year>/day/``.

The merge preserves the existing 33-column wide-table format — rows are
deduplicated on ``(ts_code, trade_date)`` with the new data taking precedence.
No CNE ``curated/`` copy is created; the yearly file remains the single source
of truth.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from datetime import date
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Column order matching the existing quant_dataset wide-table layout.
WIDE_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "up_limit",
    "down_limit",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
    "adj_factor",
    "suspend_timing",
    "suspend_type",
    "is_st",
    "listed_days",
]


def _ts_date(d: date) -> str:
    """Format a date as Tushare expects: YYYYMMDD."""
    return d.strftime("%Y%m%d")


def _get_pro(config):
    """Return a Tushare pro API client.

    Token and URL come from [external_tushare_wide] in the config. If the token
    is not set there, we fall back to the TUSHARE_TOKEN env var so existing
    ``.env`` files keep working without config duplication.
    """
    import tushare as ts

    token = config.external_tushare_wide_token or os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Tushare token not configured: set [external_tushare_wide].tushare_token "
            "or the TUSHARE_TOKEN environment variable"
        )
    url = config.external_tushare_wide_url or os.environ.get("TUSHARE_URL", "")
    pro = ts.pro_api(token)
    if url:
        pro._DataApi__http_url = url
    return pro


def _fetch_with_retry(
    pro, api: str, retries: int = 4, interval: float = 0.3, **kwargs
) -> pl.DataFrame:
    """Fetch a Tushare API with sleep, exponential backoff retry."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        time.sleep(max(0.0, interval))
        try:
            pdf = getattr(pro, api)(**kwargs)
            if pdf is None or (hasattr(pdf, "empty") and pdf.empty):
                return pl.DataFrame()
            return pl.from_pandas(pdf)
        except Exception as exc:
            last_err = exc
            if attempt == retries:
                break
            delay = min(30.0, 1.5 * (2 ** (attempt - 1)))
            logger.warning(
                "%s failed (%s: %s), retry %d/%d in %.1fs",
                api,
                exc.__class__.__name__,
                exc,
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def fetch_one_trade_date(config, trade_date: date) -> pl.DataFrame:
    """Fetch daily + daily_basic + adj_factor for one trade date.

    Returns a polars DataFrame with the wide-table columns. Missing optional
    columns (up_limit, suspend_timing, etc.) are filled with null.
    """
    pro = _get_pro(config)
    interval = config.external_tushare_wide_interval
    d = _ts_date(trade_date)

    daily = _fetch_with_retry(pro, "daily", interval=interval, trade_date=d)
    if daily.is_empty():
        return pl.DataFrame()

    basic = _fetch_with_retry(pro, "daily_basic", interval=interval, trade_date=d)
    adj = _fetch_with_retry(
        pro,
        "adj_factor",
        interval=interval,
        trade_date=d,
        fields="ts_code,trade_date,adj_factor",
    )

    # daily_basic has a ``close`` column that duplicates daily's; drop it
    # before the left-join to avoid column name conflicts.
    if not basic.is_empty() and "close" in basic.columns:
        basic = basic.drop("close")

    merged = daily
    if not basic.is_empty():
        merged = merged.join(basic, on=["ts_code", "trade_date"], how="left")
    if not adj.is_empty():
        merged = merged.join(adj, on=["ts_code", "trade_date"], how="left")

    # Normalise the trade_date column to Date type.
    if "trade_date" in merged.columns:
        merged = merged.with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format="%Y%m%d")
            .alias("trade_date")
        )

    # Add placeholder columns the existing archive has but Tushare's
    # daily/daily_basic/adj_factor APIs do not return. These are left as
    # null so the schema stays consistent across all yearly files.
    for col in WIDE_COLUMNS:
        if col not in merged.columns:
            merged = merged.with_columns(pl.lit(None).alias(col))

    # Cast known string columns.
    for col in ("ts_code", "suspend_timing", "suspend_type"):
        if col in merged.columns:
            merged = merged.with_columns(pl.col(col).cast(pl.Utf8))

    # Cast numeric columns.
    numeric_cols = [
        c
        for c in WIDE_COLUMNS
        if c not in ("ts_code", "trade_date", "suspend_timing", "suspend_type")
    ]
    exprs = []
    for col in numeric_cols:
        if col in merged.columns:
            if col == "is_st":
                exprs.append(pl.col(col).cast(pl.Int64, strict=False).fill_null(0).alias(col))
            elif col == "listed_days":
                exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
            else:
                exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    if exprs:
        merged = merged.with_columns(exprs)

    return merged.select(WIDE_COLUMNS)


def _yearly_file_path(root: Path, trade_date: date) -> Path:
    """Return the parquet path for the year containing ``trade_date``."""
    year = trade_date.year
    return root / str(year) / str(year) / "day" / "stock_daily.parquet"


def _atomic_merge(existing_path: Path, new_df: pl.DataFrame) -> int:
    """Merge ``new_df`` into ``existing_path`` using ANTI JOIN + UNION.

    Rows in the existing file whose (ts_code, trade_date) appears in
    ``new_df`` are replaced; the new rows are appended. The merge is atomic:
    a temporary file is written then ``os.replace`` swaps it in.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(existing_path.parent),
        prefix=".merge-",
        suffix=".parquet",
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    try:
        if existing_path.exists():
            old = pl.scan_parquet(existing_path)
            old_schema = old.collect_schema()
            old_trade_date_type = old_schema.get("trade_date")

            # Align new_df's trade_date to the existing file's type. Historical
            # files store datetime[ns] (legacy Tushare format), while new rows
            # are cast to Date in fetch_one_trade_date. The join requires both
            # sides to share the same key type.
            new_aligned = new_df
            if old_trade_date_type is not None:
                new_aligned = new_df.with_columns(pl.col("trade_date").cast(old_trade_date_type))

            # Remove rows from old that have the same (ts_code, trade_date) as new.
            old_without_overlap = old.join(
                new_aligned.lazy().select("ts_code", "trade_date").unique(),
                on=["ts_code", "trade_date"],
                how="anti",
            )
            combined = pl.concat(
                [old_without_overlap.collect(), new_aligned],
                how="diagonal_relaxed",
            )
        else:
            combined = new_df

        combined.write_parquet(tmp_path, compression="zstd")
        os.replace(tmp_path, existing_path)
        return combined.height
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def fetch_and_merge_one_date(config, trade_date: date) -> dict:
    """Fetch one trade date from Tushare and merge into the yearly archive.

    Returns a result dict with keys: rows_read, rows_written, status, error.
    """
    root = config.external_tushare_wide_root
    if root is None:
        return {
            "status": "error",
            "error": "external_tushare_wide_root not set",
            "rows_read": 0,
            "rows_written": 0,
        }

    try:
        new_df = fetch_one_trade_date(config, trade_date)
    except Exception as exc:
        logger.error("fetch_one_trade_date(%s) failed: %s", trade_date, exc)
        return {"status": "error", "error": str(exc), "rows_read": 0, "rows_written": 0}

    if new_df.is_empty():
        logger.warning("Tushare returned 0 rows for %s (non-trading day or holiday?)", trade_date)
        return {"status": "success", "rows_read": 0, "rows_written": 0}

    target = _yearly_file_path(root, trade_date)
    target.parent.mkdir(parents=True, exist_ok=True)

    rows_written = _atomic_merge(target, new_df)
    logger.info(
        "tushare_wide_daily %s: %d new rows merged -> %s (total %d)",
        trade_date,
        new_df.height,
        target,
        rows_written,
    )
    return {
        "status": "success",
        "rows_read": new_df.height,
        "rows_written": rows_written,
    }
