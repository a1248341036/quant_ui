"""Read-only adapter for AlphaAgent factor panel.

The AlphaAgent ``panel_1d.parquet`` (1.15 GB, 37 columns) is a daily-frequency
feature matrix combining OHLCV, valuation, and label columns for ML-based
alpha research.  It is registered as a derived-layer dataset so the dashboard
can show its coverage and row count alongside the native CNE datasets.

File → dataset
--------------
    data/alphaagent/panel_1d.parquet → ``alpha_panel_1d``  (derived, L1)

Column conventions
-------------------
The file uses ``instrument`` (e.g. ``000007.SZ``) and ``datetime``
(timestamp[ns]) as the primary key.  We rename ``instrument`` → ``symbol`` and
``datetime`` → ``date`` for consistency with CNE conventions.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from cnequity.config import Config
from cnequity.query.parquet_scan import scan_parquet_files

logger = logging.getLogger(__name__)

SOURCE = "alphaagent"
VERSION = "alphaagent-v1"
_EXTERNAL_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

DATASET = "alpha_panel_1d"
FILE_NAME = "panel_1d.parquet"
DATE_COL = "date"
SYMBOL_COL = "symbol"


class AlphaAgentAdapter:
    """Adapter for the AlphaAgent factor panel (alpha_panel_1d)."""

    is_native = False

    DATASETS = (DATASET,)

    # ── read protocol ──────────────────────────────────────────────────

    def enabled(self, config: Config, dataset: str) -> bool:
        return (
            dataset == DATASET
            and getattr(config, "external_alphaagent_enabled", False)
            and getattr(config, "external_alphaagent_root", None) is not None
        )

    def _file(self, config: Config) -> Path | None:
        root: Path | None = getattr(config, "external_alphaagent_root", None)
        if root is None:
            return None
        path = root / FILE_NAME
        return path if path.is_file() else None

    def has_data(self, config: Config, dataset: str) -> bool:
        return self._file(config) is not None

    def files(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Path]:
        path = self._file(config)
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
            raise FileNotFoundError("no alphaagent panel_1d file")

        raw = scan_parquet_files(paths, missing_columns="insert", extra_columns="ignore")
        schema_names = raw.collect_schema().names()

        rename_map: dict[str, str] = {}
        if "instrument" in schema_names:
            rename_map["instrument"] = "symbol"
        if "datetime" in schema_names:
            rename_map["datetime"] = "date"
        if rename_map:
            raw = raw.rename(rename_map)

        out = raw.with_columns(
            pl.lit(SOURCE).alias("source"),
            pl.lit(VERSION).alias("data_version"),
            pl.lit(_EXTERNAL_EPOCH).alias("fetched_at"),
        )

        if "date" in out.collect_schema().names():
            out = out.with_columns(pl.col("date").cast(pl.Date).alias("date"))

        if start is not None and "date" in out.collect_schema().names():
            out = out.filter(pl.col("date") >= start)
        if end is not None and "date" in out.collect_schema().names():
            out = out.filter(pl.col("date") <= end)
        if symbols and "symbol" in out.collect_schema().names():
            out = out.filter(pl.col("symbol").is_in(symbols))

        return out

    def coverage_bounds(self, config: Config, dataset: str) -> tuple[date | None, date | None]:
        path = self._file(config)
        if path is None:
            return None, None

        try:
            lf = scan_parquet_files([path], missing_columns="insert", extra_columns="ignore")
            if "datetime" not in lf.collect_schema().names():
                return None, None
            row = (
                lf.select(
                    pl.col("datetime").min().alias("first"),
                    pl.col("datetime").max().alias("last"),
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
            logger.warning("alphaagent coverage_bounds failed: %s", exc)
            return None, None


# Module-level singleton for auto-discovery.
ADAPTER = AlphaAgentAdapter()
