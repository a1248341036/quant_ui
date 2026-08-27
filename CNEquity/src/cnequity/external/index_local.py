"""Read-only + write-back adapter for the local benchmark index panel.

The quant_ui framework maintains a small index CSV with 6 major indices
used as benchmark data for backtesting. Since 2026-08-26 the file is owned
by CNE: ``step_index_bars_external`` stages a Tencent-kline snapshot and
``compact`` merges it into ``<root>/stock/index.parquet`` through this
adapter's write protocol. The legacy ``index.csv`` is only read once as a
bootstrap source when the parquet does not exist yet.

File -> dataset
---------------
    data/stock/index.parquet -> ``index_bars_external``  (L1, external)

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
from cnequity.query.parquet_scan import scan_parquet_files

logger = logging.getLogger(__name__)

DATASET = "index_bars_external"
FILE_REL = "stock/index.parquet"
LEGACY_CSV_REL = "stock/index.csv"
SOURCE = "local_index"
VERSION = "local-v2"
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
        if path.is_file():
            return path
        # One-time bootstrap source: the legacy CSV maintained by quant_ui's
        # refresh_data before CNE took ownership.
        legacy = root / LEGACY_CSV_REL
        return legacy if legacy.is_file() else None

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

        path = paths[0]
        if path.suffix == ".csv":
            raw = pl.scan_csv(str(path), try_parse_dates=True, ignore_errors=True)
            # Legacy writer doubled carriage returns on Windows, leaving the
            # last header cell as 'close\r'.
            rename = {c: c.strip() for c in raw.collect_schema().names() if c != c.strip()}
            if rename:
                raw = raw.rename(rename)
        else:
            raw = scan_parquet_files([path], missing_columns="insert", extra_columns="ignore")
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
            path = paths[0]
            if path.suffix == ".csv":
                df = pl.read_csv(str(path), try_parse_dates=True, ignore_errors=True)
                df = df.rename({c: c.strip() for c in df.columns})
            else:
                df = pl.read_parquet(str(path))
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

    # ── write protocol (compactable) ───────────────────────────────────

    def compact_layout(self) -> str:
        # Single small file; every year group resolves to the same target and
        # the sequential anti-join merges compose correctly.
        return "yearly_file"

    def compact_pk(self, dataset: str) -> list[str]:
        return ["code", "date"]

    def compact_target(self, config: Config, dataset: str, trade_date: date) -> Path:
        root: str | Path | None = getattr(config, "external_local_assets_root", None)
        if root is None:
            raise RuntimeError("external_local_assets_root not configured")
        return Path(root) / FILE_REL


# Module-level singleton for auto-discovery.
ADAPTER = IndexLocalAdapter()
