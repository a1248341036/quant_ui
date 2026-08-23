"""Read-only adapter for local historical minute-bar parquet files.

The quant_ui framework maintains per-symbol parquet files for 1-minute,
5-minute (and 30-minute) bars spanning 2009–2026::

    <root>/<year>/<year>/1min/<symbol>.parquet
    <root>/<year>/<year>/5min/<symbol>.parquet

These files are managed outside the CNE curated lake.  This adapter bridges
them into the CNE catalog so ``minute_bars`` (1m) and ``minute_bars_5m`` (5m)
appear alongside other datasets and can be queried via the unified
``cnequity.query.reader`` API — without copying 84+ GB of data into curated.

File → dataset mapping
----------------------
    <root>/YYYY/YYYY/1min/*.parquet → ``minute_bars``   (L1, external)
    <root>/YYYY/YYYY/5min/*.parquet → ``minute_bars_5m`` (L1, external)

Column conventions
-------------------
The local files use ``ts_code`` (e.g. ``000001.SZ``) as the identifier and
``trade_date`` / ``trade_time`` for the date and bar timestamp.  We rename
``ts_code`` → ``symbol``, ``trade_time`` → ``bar_time``, cast ``vol`` →
``volume`` (Int64), and drop ``adj_factor`` (not in the CNE minute-bars
schema).  Provenance columns (``frequency``, ``source``, ``data_version``,
``fetched_at``) are added lazily.

Performance note
----------------
Each year directory holds ~5,500 per-symbol files.  Returning all of them
from ``files()`` would make the dashboard's partition-stats scan enumerate
~100 K parquet footers.  Instead, ``files()`` without a date range returns
one *representative* file per year (sufficient for coverage bounds and
catalog visibility); ``scan()`` performs its own file discovery with symbol
and date filtering, so actual queries are correct and complete.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from cnequity.config import Config
from cnequity.query.parquet_scan import scan_parquet_files

logger = logging.getLogger(__name__)

SOURCE = "local_minute_bars"
VERSION = "local-v1"
_EXTERNAL_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# ── dataset → directory / frequency value ─────────────────────────────
_DATASET_CONFIG: dict[str, tuple[str, str]] = {
    "minute_bars": ("1min", "1m"),
    "minute_bars_5m": ("5min", "5m"),
}


class MinuteBarsLocalAdapter:
    """Adapter for local historical minute-bar parquet files."""

    is_native = False

    DATASETS = tuple(_DATASET_CONFIG.keys())

    # ── helpers ─────────────────────────────────────────────────────────

    def _root(self, config: Config) -> Path | None:
        return getattr(config, "external_minute_bars_local_root", None)

    def _year_dirs(self, root: Path, freq_dir: str) -> list[tuple[int, Path]]:
        """Return ``(year, directory)`` pairs for year directories that exist."""
        if not root.exists():
            return []
        out: list[tuple[int, Path]] = []
        for entry in sorted(root.iterdir(), key=lambda e: e.name):
            if not entry.is_dir() or not entry.name.isdigit() or len(entry.name) != 4:
                continue
            year = int(entry.name)
            ydir = entry / entry.name / freq_dir
            if ydir.is_dir():
                out.append((year, ydir))
        return out

    def _all_parquet_files(self, ydir: Path) -> list[Path]:
        """Sorted parquet files in a year/frequency directory."""
        return sorted(ydir.glob("*.parquet"))

    # ── read protocol ──────────────────────────────────────────────────

    def enabled(self, config: Config, dataset: str) -> bool:
        return (
            dataset in _DATASET_CONFIG
            and getattr(config, "external_minute_bars_local_enabled", False)
            and getattr(config, "external_minute_bars_local_root", None) is not None
        )

    def has_data(self, config: Config, dataset: str) -> bool:
        root = self._root(config)
        if root is None:
            return False
        freq_dir = _DATASET_CONFIG[dataset][0]
        for _, ydir in self._year_dirs(root, freq_dir):
            if any(ydir.glob("*.parquet")):
                return True
        return False

    def files(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Path]:
        """Return parquet paths for the dataset.

        Without ``start``/``end``: one representative file per year (the first
        symbol's file), sufficient for the dashboard's partition-stats scan to
        determine coverage bounds without opening ~100 K footers.

        With ``start``/``end``: all files in matching year directories.
        """
        root = self._root(config)
        if root is None:
            return []
        freq_dir = _DATASET_CONFIG[dataset][0]

        result: list[Path] = []
        for year, ydir in self._year_dirs(root, freq_dir):
            if start is not None and year < start.year:
                continue
            if end is not None and year > end.year:
                continue

            pq_files = self._all_parquet_files(ydir)
            if not pq_files:
                continue

            if start is None and end is None:
                result.append(pq_files[0])
            else:
                result.extend(pq_files)

        return result

    def scan(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
        symbols: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Lazy scan with column mapping, date/symbol pushdown, and provenance."""
        root = self._root(config)
        if root is None:
            raise FileNotFoundError(f"no root configured for dataset {dataset!r}")

        freq_dir, freq_value = _DATASET_CONFIG[dataset]

        # ── file discovery with year + symbol filtering ───────────────────
        all_files: list[Path] = []
        for year, ydir in self._year_dirs(root, freq_dir):
            if start is not None and year < start.year:
                continue
            if end is not None and year > end.year:
                continue

            if symbols:
                for sym in symbols:
                    p = ydir / f"{sym}.parquet"
                    if p.is_file():
                        all_files.append(p)
            else:
                all_files.extend(self._all_parquet_files(ydir))

        if not all_files:
            raise FileNotFoundError(
                f"no minute-bar parquet files for dataset {dataset!r} "
                f"in {root} (freq={freq_dir})"
            )

        # ── lazy scan with schema reconciliation ──────────────────────────
        raw = scan_parquet_files(all_files, missing_columns="insert", extra_columns="ignore")
        schema_names = raw.collect_schema().names()

        # ── column mapping: ts_code → symbol, trade_time → bar_time ───────
        rename_map: dict[str, str] = {}
        if "ts_code" in schema_names:
            rename_map["ts_code"] = "symbol"
        if "trade_time" in schema_names:
            rename_map["trade_time"] = "bar_time"
        if rename_map:
            raw = raw.rename(rename_map)

        schema_names = raw.collect_schema().names()

        # ── type casts ────────────────────────────────────────────────────
        out = raw

        if "trade_date" in schema_names:
            existing_dtype = raw.collect_schema()["trade_date"]
            if str(existing_dtype) != "Date":
                out = out.with_columns(
                    pl.col("trade_date").cast(pl.Date).alias("trade_date")
                )

        if "vol" in schema_names:
            out = out.with_columns(
                pl.col("vol").fill_null(0.0).round(0).cast(pl.Int64).alias("volume")
            )
        elif "volume" not in schema_names:
            out = out.with_columns(pl.lit(0).cast(pl.Int64).alias("volume"))

        # ── provenance columns ────────────────────────────────────────────
        out = out.with_columns(
            pl.lit(freq_value).alias("frequency"),
            pl.lit(SOURCE).alias("source"),
            pl.lit(VERSION).alias("data_version"),
            pl.lit(_EXTERNAL_EPOCH).alias("fetched_at"),
        )

        # ── filters ───────────────────────────────────────────────────────
        if start is not None and "trade_date" in out.collect_schema().names():
            out = out.filter(pl.col("trade_date") >= start)
        if end is not None and "trade_date" in out.collect_schema().names():
            out = out.filter(pl.col("trade_date") <= end)
        if symbols and "symbol" in out.collect_schema().names():
            out = out.filter(pl.col("symbol").is_in(symbols))

        # ── select canonical schema ───────────────────────────────────────
        canonical = [
            "symbol",
            "trade_date",
            "bar_time",
            "frequency",
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
        available = set(out.collect_schema().names())
        select_cols = [c for c in canonical if c in available]
        return out.select(select_cols)

    def coverage_bounds(self, config: Config, dataset: str) -> tuple[date | None, date | None]:
        """Earliest and latest ``trade_date`` across all year directories.

        Uses the first and last year's representative file for a cheap bound,
        matching the approach in ``tushare_wide.coverage_bounds``.
        """
        root = self._root(config)
        if root is None:
            return None, None

        freq_dir = _DATASET_CONFIG[dataset][0]
        year_dirs = self._year_dirs(root, freq_dir)
        if not year_dirs:
            return None, None

        def _date_bounds(path: Path) -> tuple[date | None, date | None]:
            try:
                lf = scan_parquet_files(
                    [path], missing_columns="insert", extra_columns="ignore"
                )
                schema_names = lf.collect_schema().names()
                date_col = "trade_date" if "trade_date" in schema_names else None
                if date_col is None:
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
                logger.warning("minute_bars coverage_bounds: failed to read %s: %s", path, exc)
                return None, None

        first_year, first_dir = year_dirs[0]
        first_files = self._all_parquet_files(first_dir)
        first_date = None
        if first_files:
            first_date, _ = _date_bounds(first_files[0])

        last_year, last_dir = year_dirs[-1]
        last_files = self._all_parquet_files(last_dir)
        last_date = None
        if last_files:
            _, last_date = _date_bounds(last_files[0])

        if len(year_dirs) == 1 and first_files:
            _, last_date = _date_bounds(first_files[0])

        return first_date, last_date


# Module-level singleton for auto-discovery.
ADAPTER = MinuteBarsLocalAdapter()
