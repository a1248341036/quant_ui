"""Read an existing yearly Tushare-wide archive as CNE ``daily_bars``.

The source layout stays authoritative and read-only::

    <root>/<year>/<year>/day/stock_daily.parquet

Rows are mapped lazily into CNE's logical daily-bars contract. No historical
Parquet is copied into ``curated/``.

This adapter is **write-capable** (compactable=True on the DatasetSpec):
``compact_target()`` / ``compact_layout()`` / ``compact_pk()`` tell the
compaction engine where to merge staging rows back into the yearly archive.
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
VERSION = "tushare-wide-v1"
_EXTERNAL_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


class TushareWideAdapter:
    """Adapter for the yearly Tushare-wide daily archive (daily_bars).

    Read-path maps the 33-column wide table to CNE's 11-column daily-bars
    contract (OHLCV).  Write-path (compact) merges staging back into the
    same yearly files, using the wide table's own PK (ts_code, trade_date).
    """

    is_native = False

    DATASETS = ("daily_bars",)

    # ── read protocol ──────────────────────────────────────────────────

    def enabled(self, config: Config, dataset: str) -> bool:
        return (
            dataset in self.DATASETS
            and config.external_tushare_wide_enabled
            and config.external_tushare_wide_root is not None
        )

    def _root(self, config: Config) -> Path | None:
        return config.external_tushare_wide_root

    def files(
        self, config: Config, dataset: str, *, start: date | None = None, end: date | None = None
    ) -> list[Path]:
        root = config.external_tushare_wide_root
        if root is None:
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
        columns = raw.collect_schema().names()
        required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"external Tushare-wide data missing columns: {missing}")

        out = raw.with_columns(
            pl.col("trade_date").cast(pl.Date).alias("trade_date"),
            pl.col("ts_code").cast(pl.Utf8).alias("symbol"),
            (pl.col("vol").fill_null(0.0) * 100.0).round().cast(pl.Int64).alias("volume"),
            (pl.col("amount").fill_null(0.0) * 1000.0).alias("amount"),
            pl.lit(SOURCE).alias("source"),
            pl.lit(VERSION).alias("data_version"),
            pl.lit(_EXTERNAL_EPOCH).alias("fetched_at"),
        ).select(
            [
                "symbol",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "source",
                "data_version",
                "fetched_at",
            ]
        )
        if start is not None:
            out = out.filter(pl.col("trade_date") >= start)
        if end is not None:
            out = out.filter(pl.col("trade_date") <= end)
        if symbols:
            out = out.filter(pl.col("symbol").is_in(symbols))
        return out.filter(
            (pl.col("volume") > 0)
            & (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & (pl.col("close") > 0)
        )

    def coverage_bounds(self, config: Config, dataset: str) -> tuple[date | None, date | None]:
        paths = self.files(config, dataset)
        if not paths:
            return None, None

        first_path = paths[0]
        last_path = paths[-1]

        def _date_bounds(path: Path) -> tuple[date | None, date | None]:
            try:
                lf = scan_parquet_files([path], missing_columns="insert", extra_columns="ignore")
                if "trade_date" not in lf.collect_schema().names():
                    return None, None
                row = (
                    lf.select(
                        pl.col("trade_date").min().alias("first"),
                        pl.col("trade_date").max().alias("last"),
                    )
                    .collect()
                    .row(0)
                )
                first_raw, last_raw = row[0], row[1]
                first = (
                    first_raw.date()
                    if hasattr(first_raw, "date")
                    else (date.fromisoformat(str(first_raw)[:10]) if first_raw else None)
                )
                last = (
                    last_raw.date()
                    if hasattr(last_raw, "date")
                    else (date.fromisoformat(str(last_raw)[:10]) if last_raw else None)
                )
                return first, last
            except Exception as exc:
                logger.warning("coverage_bounds: failed to read %s: %s", path, exc)
                return None, None

        first_date, _ = _date_bounds(first_path)
        _, last_date = _date_bounds(last_path)

        if first_path == last_path:
            _, last_date = _date_bounds(last_path)

        return first_date, last_date

    # ── write protocol (compactable) ───────────────────────────────────

    def compact_layout(self) -> str:
        return "yearly_file"

    def compact_pk(self, dataset: str) -> list[str]:
        return ["ts_code", "trade_date"]

    def compact_target(self, config: Config, dataset: str, trade_date: date) -> Path:
        root = self._root(config)
        if root is None:
            raise RuntimeError("external_tushare_wide_root not configured")
        year = trade_date.year
        return root / str(year) / str(year) / "day" / "stock_daily.parquet"


# Module-level singleton for auto-discovery.
ADAPTER = TushareWideAdapter()
