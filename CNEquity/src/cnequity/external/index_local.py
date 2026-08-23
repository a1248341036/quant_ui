"""Read-only adapter for local index CSV (data/stock/index.csv).

The quant_ui framework maintains a small index CSV with 6 major indices
used as benchmark data for backtesting. This adapter exposes that file as the
``index_bars_external`` dataset so it appears alongside other CNE datasets
in the dashboard and can be read via the unified ``cnequity.query.reader``
API.

File -> dataset
---------------
    data/stock/index.csv -> ``index_bars_external``  (L1, external)

Column conventions
-------------------
The local file uses ``code`` (e.g. ``sh000300``) and ``date`` columns.
We keep the original columns without renaming to ``symbol``/``trade_date``,
because the framework's ``load_index()`` expects ``code``/``date``/``name``
and these indices are not part of CNE's native index_bars contract.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from cnequity.config import Config

logger = logging.getLogger(__name__)

DATASET = "index_bars_external"
FILE_REL = "stock/index.csv"
SOURCE = "local_index"
VERSION = "local-v1"
_EXTERNAL_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


class IndexLocalAdapter:
    """Adapter for local index CSV (index_bars_external)."""

    is_native = False

    DATASETS = (DATASET,)

    # ── read protocol ──────────────────────────────────────────────────

    def enabled(self, config: Config, dataset: str) -> bool:
        return (
            dataset == DATASET
            and getattr(config, "external_local_assets_enabled", False)
            and getattr(config, "external_local_assets_root", None) is not None
        )

    def _resolve_path(self, config: Config) -> Path | None:
        root: Path | None = getattr(config, "external_local_assets_root", None)
        if root is None:
            return None
        path = root / FILE_REL
        return path if path.is_file() else None

    def has_data(self, config: Config, dataset: str) -> bool:
        return self._resolve_path(config) is not None

    def files(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Path]:
        path = self._resolve_path(config)
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
            raise FileNotFoundError(f"no index file for dataset {dataset!r}")

        raw = pl.scan_csv(str(paths[0]), try_parse_dates=True, ignore_errors=True)
        schema_names = raw.collect_schema().names()

        out = raw.with_columns(
            pl.lit(SOURCE).alias("source"),
            pl.lit(VERSION).alias("data_version"),
            pl.lit(_EXTERNAL_EPOCH).alias("fetched_at"),
        )

        if "date" in schema_names:
            existing_dtype = raw.collect_schema()["date"]
            if str(existing_dtype) != "Date":
                out = out.with_columns(pl.col("date").cast(pl.Date).alias("date"))
            col = pl.col("date")
            if start is not None:
                out = out.filter(col >= start)
            if end is not None:
                out = out.filter(col <= end)

        if symbols and "code" in schema_names:
            out = out.filter(pl.col("code").cast(pl.Utf8).is_in(symbols))

        return out

    def coverage_bounds(self, config: Config, dataset: str) -> tuple[date | None, date | None]:
        paths = self.files(config, dataset)
        if not paths:
            return None, None

        try:
            df = pl.read_csv(str(paths[0]), try_parse_dates=True, ignore_errors=True)
            if "date" not in df.columns:
                return None, None
            if df.schema["date"] != pl.Date:
                df = df.with_columns(pl.col("date").cast(pl.Date))
            first = df.select(pl.col("date").min()).item()
            last = df.select(pl.col("date").max()).item()

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

            return _to_date(first), _to_date(last)
        except Exception as exc:
            logger.warning("index_bars_external coverage_bounds failed: %s", exc)
            return None, None


# Module-level singleton for auto-discovery.
ADAPTER = IndexLocalAdapter()
