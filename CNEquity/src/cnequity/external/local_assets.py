"""Read-only adapter for raw local ETF and fund sources.

These files are managed by quant_ui today.  They live outside the curated lake
and are mounted here so catalog/dashboard/query use one CNE name without
copying them.  Downstream stock panels, predictions and universes are business
data and intentionally remain outside this bridge.

File → dataset mapping
----------------------
    data/etf/etf_panel.parquet    → ``etf_bars``  (L1)
    data/etf/etf.csv              → ``etf_list``  (L0, snapshot)
    data/fund/fund_panel.parquet  → ``fund_bars``  (L1)
    data/fund/fund_nav.parquet    → ``fund_nav``  (L1)
    data/fund/fund.csv            → ``fund_list``  (L0, snapshot)

Column conventions
-------------------
The local files use ``code`` (pure digits like ``000001``) as the identifier,
while CNE's native datasets use ``symbol`` (``000001.SZ``).  We keep the
original ``code`` column without renaming, because these datasets do not cross
the native CNE path — they are standalone research panels displayed in their
own dashboard entries.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from cnequity.config import Config
from cnequity.query.parquet_scan import scan_parquet_files

logger = logging.getLogger(__name__)

SOURCE = "local_assets"
VERSION = "local-v1"
_EXTERNAL_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# ── dataset → file config ────────────────────────────────────────────
_DATASET_FILES: dict[str, tuple[str, str | None, str, str]] = {
    "etf_bars": ("etf/etf_panel.parquet", "date", "code", "parquet"),
    "etf_list": ("etf/etf.csv", None, "code", "csv"),
    "fund_bars": ("fund/fund_panel.parquet", "date", "code", "parquet"),
    "fund_nav": ("fund/fund_nav.parquet", "date", "code", "parquet"),
    "fund_list": ("fund/fund.csv", None, "code", "csv"),
    "fund_fees": ("fund/fund_fee.parquet", None, "code", "parquet"),
}


class LocalAssetsAdapter:
    """Adapter for raw local ETF and fund parquet/CSV files."""

    is_native = False

    DATASETS = tuple(_DATASET_FILES.keys())

    # ── read protocol ──────────────────────────────────────────────────

    def enabled(self, config: Config, dataset: str) -> bool:
        included = getattr(config, "external_local_assets_include", None)
        return (
            dataset in _DATASET_FILES
            and getattr(config, "external_local_assets_enabled", False)
            and getattr(config, "external_local_assets_root", None) is not None
            and (included is None or dataset in included)
        )

    def _resolve_path(self, config: Config, dataset: str) -> Path | None:
        root: Path | None = getattr(config, "external_local_assets_root", None)
        if root is None:
            return None
        rel_path, _date_col, _code_col, _ftype = _DATASET_FILES[dataset]
        path = root / rel_path
        return path if path.exists() else None

    def has_data(self, config: Config, dataset: str) -> bool:
        return self._resolve_path(config, dataset) is not None

    def files(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Path]:
        path = self._resolve_path(config, dataset)
        return [path] if path is not None else []

    def _scan_csv(self, path: Path) -> pl.LazyFrame:
        """Scan a CSV file, handling encoding issues."""
        return pl.scan_csv(str(path), try_parse_dates=True, ignore_errors=True)

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
            raise FileNotFoundError(f"no local_assets file for dataset {dataset!r}")

        rel_path, date_col, code_col, ftype = _DATASET_FILES[dataset]
        path = paths[0]

        if ftype == "csv":
            raw = self._scan_csv(path)
        else:
            raw = scan_parquet_files([path], missing_columns="insert", extra_columns="ignore")

        schema_names = raw.collect_schema().names()

        out = raw.with_columns(
            pl.lit(SOURCE).alias("source"),
            pl.lit(VERSION).alias("data_version"),
            pl.lit(_EXTERNAL_EPOCH).alias("fetched_at"),
        )

        if date_col and date_col in schema_names:
            existing_dtype = raw.collect_schema()[date_col]
            if str(existing_dtype) != "Date":
                out = out.with_columns(pl.col(date_col).cast(pl.Date).alias(date_col))
            col = pl.col(date_col)
            if start is not None:
                out = out.filter(col >= start)
            if end is not None:
                out = out.filter(col <= end)

        if symbols and code_col and code_col in schema_names:
            out = out.filter(pl.col(code_col).cast(pl.Utf8).is_in(symbols))

        return out

    def coverage_bounds(self, config: Config, dataset: str) -> tuple[date | None, date | None]:
        paths = self.files(config, dataset)
        if not paths:
            return None, None

        _rel_path, date_col, _code_col, ftype = _DATASET_FILES[dataset]
        if not date_col:
            return None, None

        try:
            if ftype == "csv":
                df = pl.read_csv(str(paths[0]), try_parse_dates=True, ignore_errors=True)
                if date_col not in df.columns:
                    return None, None
                if df.schema[date_col] != pl.Date:
                    df = df.with_columns(pl.col(date_col).cast(pl.Date))
                first = df.select(pl.col(date_col).min()).item()
                last = df.select(pl.col(date_col).max()).item()
            else:
                lf = scan_parquet_files(paths, missing_columns="insert", extra_columns="ignore")
                if date_col not in lf.collect_schema().names():
                    return None, None
                row = (
                    lf.select(
                        pl.col(date_col).min().alias("first"),
                        pl.col(date_col).max().alias("last"),
                    )
                    .collect()
                    .row(0)
                )
                first, last = row[0], row[1]

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
            logger.warning("local_assets coverage_bounds failed for %s: %s", dataset, exc)
            return None, None


    # ── write protocol (compactable: fund_nav) ─────────────────────────

    # fund_nav is the only dataset this adapter owns writes for. The target is
    # a single parquet file: every year group from ``_compact_yearly_file``
    # resolves to the same path, and the sequential anti-join merges compose
    # correctly because each call re-reads the file the previous one wrote.
    _COMPACTABLE_DATASETS = frozenset({"fund_nav"})

    def compact_layout(self) -> str:
        return "yearly_file"

    def compact_pk(self, dataset: str) -> list[str]:
        return ["code", "date"]

    def compact_target(self, config: Config, dataset: str, trade_date: date) -> Path:
        if dataset not in self._COMPACTABLE_DATASETS:
            raise RuntimeError(
                f"local_assets adapter does not own writes for {dataset!r}"
            )
        root: str | Path | None = getattr(config, "external_local_assets_root", None)
        if root is None:
            raise RuntimeError("external_local_assets_root not configured")
        rel_path, *_rest = _DATASET_FILES[dataset]
        return Path(root) / rel_path


# Module-level singleton for auto-discovery.
ADAPTER = LocalAssetsAdapter()
