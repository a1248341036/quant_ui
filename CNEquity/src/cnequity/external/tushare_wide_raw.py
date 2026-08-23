"""Expose the full Tushare-wide daily row behind ``stock_daily_wide``.

``daily_bars`` is the canonical research contract. Some derived pipelines also
need source fields that intentionally stay out of that contract (adj_factor,
daily_basic, market cap and ST). This adapter serves those rows from the same
read-only yearly archive without copying it.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from cnequity.config import Config
from cnequity.query.parquet_scan import scan_parquet_files

logger = logging.getLogger(__name__)

SOURCE = "quant_dataset_tushare"
VERSION = "tushare-wide-raw-v1"
_EXTERNAL_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


class TushareWideRawAdapter:
    """Adapter for the raw wide-table dataset ``stock_daily_wide``."""

    is_native = False

    DATASETS = ("stock_daily_wide",)

    # ── read protocol ──────────────────────────────────────────────────

    def enabled(self, config: Config, dataset: str) -> bool:
        return (
            dataset in self.DATASETS
            and config.external_tushare_wide_enabled
            and config.external_tushare_wide_root is not None
        )

    def files(
        self, config: Config, dataset: str, *, start: date | None = None, end: date | None = None
    ) -> list[Path]:
        root = config.external_tushare_wide_root
        if root is None or dataset not in self.DATASETS:
            return []
        result: list[Path] = []
        for entry in sorted(root.iterdir() if root.exists() else [], key=lambda item: item.name):
            if not entry.is_dir() or not entry.name.isdigit() or len(entry.name) != 4:
                continue
            year = int(entry.name)
            if start is not None and year < start.year:
                continue
            if end is not None and year > end.year:
                continue
            path = entry / entry.name / "day" / "stock_daily.parquet"
            if path.is_file():
                result.append(path)
        return result

    def has_data(self, config: Config, dataset: str) -> bool:
        return bool(self.files(config, dataset))

    def scan(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
        symbols: list[str] | None = None,
    ) -> pl.LazyFrame:
        paths = self.files(config, dataset, start=start, end=end)
        if not paths:
            raise FileNotFoundError("no external Tushare-wide yearly daily files")
        raw = scan_parquet_files(paths, missing_columns="insert", extra_columns="ignore")
        out = raw.with_columns(
            pl.col("trade_date").cast(pl.Date).alias("trade_date"),
            pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
            pl.lit(SOURCE).alias("source"),
            pl.lit(VERSION).alias("data_version"),
            pl.lit(_EXTERNAL_EPOCH).alias("fetched_at"),
        )
        if start is not None:
            out = out.filter(pl.col("trade_date") >= start)
        if end is not None:
            out = out.filter(pl.col("trade_date") <= end)
        if symbols:
            out = out.filter(pl.col("ts_code").is_in(symbols))
        return out.sort(["trade_date", "ts_code"])

    def coverage_bounds(self, config: Config, dataset: str) -> tuple[date | None, date | None]:
        paths = self.files(config, dataset)
        if not paths:
            return None, None

        def _bounds(path: Path) -> tuple[date | None, date | None]:
            try:
                frame = (
                    scan_parquet_files([path])
                    .select(
                        pl.col("trade_date").min().alias("first"),
                        pl.col("trade_date").max().alias("last"),
                    )
                    .collect()
                )
                first, last = frame["first"][0], frame["last"][0]
                to_date = lambda value: value.date() if isinstance(value, datetime) else value
                return to_date(first), to_date(last)
            except Exception as exc:
                logger.warning("stock_daily_wide coverage unavailable for %s: %s", path, exc)
                return None, None

        first_date, _ = _bounds(paths[0])
        _, last_date = _bounds(paths[-1])
        return first_date, last_date


# Module-level singleton for auto-discovery.
ADAPTER = TushareWideRawAdapter()
