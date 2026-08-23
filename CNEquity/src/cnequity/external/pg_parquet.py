"""Read-only adapter for pre-built Tushare parquet exports in ``pg_parquet/``.

These files come from a PostgreSQL→Parquet export.  Each file maps to one CNE
dataset.  The mapping is intentionally light: we rename a few columns (ts_code
→ symbol, ann_date → announce_date) and add provenance, but we do NOT try to
unpivot the wide financial tables into CNE's long-format
``financial_statement_items`` — that would be a destructive transformation
that belongs in a derive step, not in a read adapter.  Instead we register
new wide-table datasets so the data is visible and browsable in the dashboard.

File → dataset mapping
----------------------
    stock_basic.parquet   → ``instruments_external``  (snapshot, L0)
    trade_cal.parquet     → ``trading_calendar_external``  (L0)

Financial statements, corporate events and survey datasets were migrated to
curated steps (steps/tushare_wide.py) and are no longer served by this adapter.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from cnequity.config import Config
from cnequity.query.parquet_scan import scan_parquet_files

logger = logging.getLogger(__name__)

SOURCE = "pg_parquet"
VERSION = "pg-parquet-v1"
_EXTERNAL_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# ── file → dataset name ──────────────────────────────────────────────
_FILE_MAP: dict[str, str] = {
    "stock_basic.parquet": "instruments_external",
    "trade_cal.parquet": "trading_calendar_external",
}

# reverse lookup
_DATASET_TO_FILE: dict[str, str] = {v: k for k, v in _FILE_MAP.items()}

# ── column renames per dataset ───────────────────────────────────────
_RENAMES: dict[str, dict[str, str]] = {
    "instruments_external": {},
    "trading_calendar_external": {"cal_date": "trade_date"},
}

# ── date columns per dataset (for coverage_bounds and partitioning) ─
_DATE_COL: dict[str, str] = {
    "instruments_external": "list_date",
    "trading_calendar_external": "trade_date",
}


class PgParquetAdapter:
    """Adapter for pre-built Tushare parquet exports (pg_parquet/)."""

    is_native = False

    DATASETS = tuple(_FILE_MAP.values())

    # ── read protocol ──────────────────────────────────────────────────

    def enabled(self, config: Config, dataset: str) -> bool:
        return (
            dataset in _DATASET_TO_FILE
            and getattr(config, "external_pg_parquet_enabled", False)
            and getattr(config, "external_pg_parquet_root", None) is not None
        )

    def _file_for(self, config: Config, dataset: str) -> Path | None:
        root = config.external_pg_parquet_root
        if root is None:
            return None
        fname = _DATASET_TO_FILE.get(dataset)
        if fname is None:
            return None
        path = root / fname
        return path if path.is_file() else None

    def has_data(self, config: Config, dataset: str) -> bool:
        return self._file_for(config, dataset) is not None

    def files(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Path]:
        path = self._file_for(config, dataset)
        return [path] if path is not None else []

    def scan(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
        symbols: list[str] | None = None,
    ) -> pl.LazyFrame:
        paths = self.files(config, dataset)
        if not paths:
            raise FileNotFoundError(f"no pg_parquet file for dataset {dataset!r}")

        raw = scan_parquet_files(paths, missing_columns="insert", extra_columns="ignore")
        renames = _RENAMES.get(dataset, {})
        if renames:
            existing = {k: v for k, v in renames.items() if k in raw.collect_schema().names()}
            if existing:
                raw = raw.rename(existing)

        schema_names = raw.collect_schema().names()
        out = raw.with_columns(
            pl.lit(SOURCE).alias("source"),
            pl.lit(VERSION).alias("data_version"),
            pl.lit(_EXTERNAL_EPOCH).alias("fetched_at"),
        )

        date_col = _DATE_COL.get(dataset)
        if date_col and date_col in schema_names:
            col = pl.col(date_col)
            if start is not None:
                out = out.filter(col >= start)
            if end is not None:
                out = out.filter(col <= end)

        if symbols and "symbol" in schema_names:
            out = out.filter(pl.col("symbol").is_in(symbols))

        return out

    def coverage_bounds(self, config: Config, dataset: str) -> tuple[date | None, date | None]:
        paths = self.files(config, dataset)
        if not paths:
            return None, None

        date_col = _DATE_COL.get(dataset)
        if not date_col:
            return None, None

        try:
            lf = scan_parquet_files(paths, missing_columns="insert", extra_columns="ignore")
            schema_names = lf.collect_schema().names()
            if date_col not in schema_names:
                renames = _RENAMES.get(dataset, {})
                reverse = {v: k for k, v in renames.items()}
                date_col = reverse.get(date_col, date_col)
                if date_col not in schema_names:
                    return None, None

            row = (
                lf.select(
                    pl.col(date_col).min().alias("first"),
                    pl.col(date_col).max().alias("last"),
                )
                .collect()
                .row(0)
            )
            first_raw, last_raw = row[0], row[1]

            def _to_date(val) -> date | None:
                if val is None:
                    return None
                if isinstance(val, date) and not isinstance(val, datetime):
                    return val
                if isinstance(val, datetime):
                    return val.date()
                if hasattr(val, "date"):
                    return val.date()
                return date.fromisoformat(str(val)[:10])

            return _to_date(first_raw), _to_date(last_raw)
        except Exception as exc:
            logger.warning("pg_parquet coverage_bounds failed for %s: %s", dataset, exc)
            return None, None


# Module-level singleton for auto-discovery.
ADAPTER = PgParquetAdapter()
