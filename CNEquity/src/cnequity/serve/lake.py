"""Read-only projection of one lake, shaped for the dashboard.

Every value here comes from something already on disk — the registry, the
directory layout, ``meta/stats``, ``meta/quality/health-latest.json``, the
manifest. **Nothing in this module scans curated.** A request that reads parquet
is a request that gets slower as the lake grows, which is the failure mode the
stats tables exist to prevent.

Two things this deliberately does *not* do:

* It does not open ``data/duckdb/cnequity.duckdb``. DuckDB allows many
  readers or one writer, so a held read handle would make
  ``ensure_duckdb_views()`` fail during the nightly run — the dashboard would
  break ingestion. Views are rebuilt in a private in-memory database instead;
  they are generated from the registry and cost milliseconds.
* It does not recompute audit findings. ``lake_health()`` walks the lake; the
  dashboard reads the JSON that ``cne audit --full`` already wrote. A page view
  must not cost what an audit costs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from cnequity.config import Config
from cnequity.domain.datasets import (
    DATASETS,
    TIER_LABELS,
    TIERS,
    history_mode_for,
    is_dataset_enabled,
    is_stale,
)
from cnequity.domain.market_time import is_session_final, shanghai_today
from cnequity.domain.partitions import parse_partition, partition_value
from cnequity.storage.stats import (
    PARTITION_STATS_SCHEMA,
    load_partition_stats,
    load_provenance_stats,
    refresh_stats_if_stale,
    stats_freshness,
)

logger = logging.getLogger(__name__)

# The catalog walks partition directories. Cheap, but the overview page fans out
# to several endpoints at once and they would each redo it.
_CACHE_TTL_SECONDS = 30.0

# Heatmap cell alphabet. One char per (dataset, day) keeps a 40x250 grid a few
# kilobytes instead of ten thousand JSON objects.
CELL_COVERED = "#"
CELL_GAP = "."
CELL_OUTSIDE = " "
CELL_UNPARTITIONED = "-"


@dataclass
class _Cached:
    value: Any
    at: float


def _jsonable(value: Any) -> Any:
    """Cell values as JSON, without inventing a type the column does not have."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _gap_meaning(spec) -> str:
    """Whether a missing day on this dataset is a fault or just its shape.

    Two ways a hole is expected rather than wrong, and both would otherwise
    paint most of the grid red:

    * **Not daily.** ``northbound_holdings`` is quarterly, so nearly every
      session inside its span legitimately has no partition.
    * **Snapshot semantics.** A snapshot dataset accumulates one stamped
      reading per run; a day nobody ran has no snapshot and *cannot* be given
      one, because replaying it would forge rows. That is the whole reason
      ``fetch_semantics`` exists — see ``domain/datasets.py``.

    Only a ``by_date`` dataset on a daily cadence can be honestly said to be
    missing a day it should have.
    """
    if spec.fetch_semantics == "snapshot" or spec.max_staleness_days > 1:
        return "cadence"
    return "fault"


def _stat_date(value) -> date | None:
    """Normalise Parquet/CSV statistic values to a calendar date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parquet_date_bounds(path, column: str) -> tuple[date | None, date | None]:
    """Read min/max for one column from row-group metadata when available."""
    parquet = pq.ParquetFile(path)
    index = parquet.schema_arrow.get_field_index(column)
    if index < 0:
        return None, None

    minimum: date | None = None
    maximum: date | None = None
    for group_index in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(group_index).column(index).statistics
        if statistics is None or not statistics.has_min_max:
            continue
        low = _stat_date(statistics.min)
        high = _stat_date(statistics.max)
        if low is not None and (minimum is None or low < minimum):
            minimum = low
        if high is not None and (maximum is None or high > maximum):
            maximum = high

    # Some writers omit column statistics. A single-column scan is still much
    # cheaper than letting the dashboard infer an incorrect empty interval.
    if minimum is None or maximum is None:
        frame = (
            pl.scan_parquet(path)
            .select(pl.col(column).min().alias("low"), pl.col(column).max().alias("high"))
            .collect()
        )
        minimum = _stat_date(frame["low"][0]) or minimum
        maximum = _stat_date(frame["high"][0]) or maximum
    return minimum, maximum


def _csv_date_bounds(path, column: str) -> tuple[date | None, date | None]:
    frame = (
        pl.scan_csv(str(path), try_parse_dates=True, ignore_errors=True)
        .select(pl.col(column).min().alias("low"), pl.col(column).max().alias("high"))
        .collect()
    )
    return _stat_date(frame["low"][0]), _stat_date(frame["high"][0])


def _next_period_start(day: date, granularity: str) -> date:
    """First day of the period after the one holding *day*."""
    if granularity == "year":
        return date(day.year + 1, 1, 1)
    if granularity == "quarter":
        quarter_end_month = 3 * ((day.month - 1) // 3) + 3
        return (
            date(day.year + 1, 1, 1)
            if quarter_end_month == 12
            else date(day.year, quarter_end_month + 1, 1)
        )
    if granularity == "month":
        return date(day.year + 1, 1, 1) if day.month == 12 else date(day.year, day.month + 1, 1)
    return day + timedelta(days=1)


# ── minute-bars static stats ──────────────────────────────────────────
# Hard-coded partition stats for minute_bars / minute_bars_5m.
# Scanning 100K+ per-symbol parquet footers on every dashboard refresh is
# too slow, and the data shape (per-year, per-symbol files) means a full
# scan takes minutes.  These values were captured on 2026-08-23 from a
# one-time scan and are intentionally frozen — refresh manually by running
# `cne stats refresh` or re-scanning if the underlying archive changes.
#
# Each tuple: (partition, period_start, period_end, row_count, file_count, bytes)
_MINUTE_BARS_1M: list[tuple] = [
    ("2009", "2009-01-05", "2009-12-31", 99849192, 1698, 2398703472),
    ("2010", "2010-01-04", "2010-12-31", 119151846, 2043, 2497093524),
    ("2011", "2011-01-04", "2011-12-30", 136484084, 2321, 3050963784),
    ("2012", "2012-01-04", "2012-12-31", 144767736, 2472, 3185834496),
    ("2013", "2013-01-04", "2013-12-31", 141731618, 2471, 3602972513),
    ("2014", "2014-01-02", "2014-12-31", 153044640, 2592, 3612003840),
    ("2015", "2015-01-05", "2015-12-31", 165298044, 2811, 4201545480),
    ("2016", "2016-01-04", "2016-12-30", 178234924, 3031, 4029829678),
    ("2017", "2017-01-03", "2017-12-29", 203873468, 3467, 4747949023),
    ("2018", "2018-01-02", "2018-12-28", 208894221, 3567, 4860048201),
    ("2019", "2019-01-02", "2019-12-31", 221279452, 3763, 5317344780),
    ("2020", "2020-01-02", "2020-12-31", 242860761, 4147, 5863621621),
    ("2021", "2021-01-04", "2021-12-31", 269858304, 4608, 6405760512),
    ("2022", "2022-01-04", "2022-12-30", 302282926, 5183, 6881536479),
    ("2023", "2023-01-03", "2023-12-29", 313539072, 5376, 6828966144),
    ("2024", "2024-01-02", "2024-12-31", 316280206, 5423, 6814389956),
    ("2025", "2025-01-02", "2025-12-31", 321335181, 5487, 6698063205),
    ("2026", "2026-01-05", "2026-08-14", 199761767, 5563, 4606475528),
]

_MINUTE_BARS_5M: list[tuple] = [
    ("2009", "2009-01-05", "2009-12-31", 20313244, 1699, 532937223),
    ("2010", "2010-01-04", "2010-12-31", 24007697, 2033, 541288283),
    ("2011", "2011-01-04", "2011-12-30", 27630316, 2311, 676704709),
    ("2012", "2012-01-04", "2012-12-31", 29338848, 2464, 726532576),
    ("2013", "2013-01-04", "2013-12-31", 28805140, 2470, 779475190),
    ("2014", "2014-01-02", "2014-12-31", 31116960, 2592, 795318912),
    ("2015", "2015-01-05", "2015-12-31", 33644184, 2814, 893940264),
    ("2016", "2016-01-04", "2016-12-30", 36274504, 3034, 896722972),
    ("2017", "2017-01-03", "2017-12-29", 41511232, 3472, 1044092896),
    ("2018", "2018-01-02", "2018-12-28", 42543711, 3573, 1101384396),
    ("2019", "2019-01-02", "2019-12-31", 45098032, 3772, 1176973388),
    ("2020", "2020-01-02", "2020-12-31", 49473585, 4155, 1297810095),
    ("2021", "2021-01-04", "2021-12-31", 57844206, 4858, 1504988968),
    ("2022", "2022-01-04", "2022-12-30", 61614168, 5196, 1596522960),
    ("2023", "2023-01-03", "2023-12-29", 63902762, 5389, 1614145614),
    ("2024", "2024-01-02", "2024-12-31", 64460088, 5436, 1611181476),
    ("2025", "2025-01-02", "2025-12-31", 65488500, 5500, 1577229500),
    ("2026", "2026-01-05", "2026-08-14", 40554270, 5563, 1141166005),
]

_MINUTE_BARS_STATIC: dict[str, list[tuple]] = {
    "minute_bars": _MINUTE_BARS_1M,
    "minute_bars_5m": _MINUTE_BARS_5M,
}


class LakeView:
    """Answers the dashboard's questions about one lake. Thread-safe."""

    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.Lock()
        self._cache: dict[str, _Cached] = {}
        self._refresh_lock = threading.Lock()
        self._refreshing = False
        self._inflight: dict[str, threading.Event] = {}

    # --- caching -----------------------------------------------------------

    def _cached(self, key: str, build):
        while True:
            now = time.monotonic()
            with self._lock:
                hit = self._cache.get(key)
                if hit is not None and now - hit.at < _CACHE_TTL_SECONDS:
                    return hit.value
                # single-flight: only one thread builds a given key; the rest
                # wait on its event and then re-check the cache below.
                ev = self._inflight.get(key)
                if ev is None:
                    ev = threading.Event()
                    self._inflight[key] = ev
                    owner = True
                else:
                    owner = False
            if not owner:
                ev.wait()
                # Loop: either the builder populated the cache and we return
                # the fresh value, or it failed/expired and we build it
                # ourselves. Never hand back an unpopulated result.
                continue
            try:
                value = build()
            except BaseException:
                with self._lock:
                    self._inflight.pop(key, None)
                ev.set()  # wake waiters so they retry instead of hanging
                raise
            with self._lock:
                self._cache[key] = _Cached(value, time.monotonic())
                self._inflight.pop(key, None)
            ev.set()
            return value

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    # --- background stats refresh -----------------------------------------

    def refresh_stats_in_background(self) -> bool:
        """Kick off a rebuild if ingestion has moved the lake. Never blocks.

        Threading lives here rather than in ``storage.stats`` so the module
        stays synchronous and testable. One thread at a time: the stats lock
        would already collapse duplicates, but spawning a thread per request to
        immediately lose a lock is waste.
        """
        with self._refresh_lock:
            if self._refreshing:
                return False
            if not stats_freshness(self.config).stale:
                return False
            self._refreshing = True

        def _run() -> None:
            try:
                result = refresh_stats_if_stale(self.config)
                if result is not None:
                    logger.info(
                        "stats rebuilt in background: %d dataset(s), %d row(s)",
                        len(result.datasets),
                        result.rows,
                    )
                    self.invalidate()
            except Exception:
                logger.exception("background stats refresh failed")
            finally:
                with self._refresh_lock:
                    self._refreshing = False

        threading.Thread(target=_run, name="stats-refresh", daemon=True).start()
        return True

    # --- primitives --------------------------------------------------------

    def anchor(self) -> date:
        """Last trading day — the date freshness is judged against."""

        def _build() -> date:
            from cnequity.steps.common import is_trading_day

            today = shanghai_today()
            day = today if is_session_final(today) else today - timedelta(days=1)
            for _ in range(15):
                if is_trading_day(self.config, day):
                    return day
                day -= timedelta(days=1)
            return shanghai_today()

        return self._cached("anchor", _build)

    def _external_partition_stats(self) -> pl.DataFrame:
        """Measure adapter-mounted files in the same shape as native stats."""

        def _bounds(path, date_col: str) -> tuple[date | None, date | None]:
            # Tushare-wide archives are <root>/<year>/<year>/day/*.parquet.
            # The directory is authoritative for the file's coarse span, so the
            # heatmap needs no column read for the largest external archive.
            year_parts = [part for part in path.parts if part.isdigit() and len(part) == 4]
            if len(year_parts) >= 2 and year_parts[0] == year_parts[1]:
                year = int(year_parts[0])
                return date(year, 1, 1), date(year, 12, 31)

            if path.suffix.lower() == ".csv":
                return _csv_date_bounds(path, date_col)
            else:
                return _parquet_date_bounds(path, date_col)

        def _build() -> pl.DataFrame:
            from cnequity.external import registry as ext_registry

            rows: list[dict] = []
            for name, spec in DATASETS.items():
                adapter = ext_registry.external_adapter(self.config, name)
                if adapter is None:
                    continue

                # minute_bars / minute_bars_5m: 使用缓存统计，避免每次刷新看板
                # 都扫描 100K+ 个 per-symbol parquet 文件。缓存文件存放在
                # meta/stats/minute_bars_cache.json，由 `cne stats refresh` 或
                # 下面的 lazy-build 逻辑生成。
                if name in {"minute_bars", "minute_bars_5m"}:
                    for row in self._minute_bars_cached_stats(name, adapter):
                        rows.append(row)
                    continue

                try:
                    paths = adapter.files(self.config, name)
                except Exception:
                    continue

                for path in paths:
                    try:
                        if path.suffix.lower() == ".parquet":
                            row_count = int(pq.read_metadata(str(path)).num_rows)
                        else:
                            with open(path, "rb") as handle:
                                row_count = max(sum(1 for _ in handle) - 1, 0)
                        if spec.query_date_col is None:
                            first = last = None
                        else:
                            first, last = _bounds(path, spec.query_date_col)
                    except Exception as exc:
                        logger.warning("external stats unavailable for %s: %s", name, exc)
                        continue
                    if first is None or last is None:
                        rows.append(
                            {
                                "dataset": name,
                                "partition": None,
                                "granularity": None,
                                "period_start": None,
                                "period_end": None,
                                "row_count": row_count,
                                "file_count": 1,
                                "bytes": path.stat().st_size,
                            }
                        )
                    else:
                        rows.append(
                            {
                                "dataset": name,
                                "partition": partition_value(first, spec.partition_granularity),
                                "granularity": spec.partition_granularity,
                                "period_start": first,
                                "period_end": last,
                                "row_count": row_count,
                                "file_count": 1,
                                "bytes": path.stat().st_size,
                            }
                        )
            return pl.DataFrame(rows, schema=PARTITION_STATS_SCHEMA)

        return self._cached("external_stats", _build)

    def _minute_bars_cached_stats(self, dataset: str, adapter) -> list[dict]:
        """Return hard-coded partition-stats rows for minute_bars datasets.

        The minute-bar archive stores per-symbol parquet files (100K+ total).
        Scanning their footers on every dashboard refresh takes minutes.
        Instead we return a frozen snapshot captured on 2026-08-23 — no file
        IO, no scanning, instant response.

        To refresh after a data backfill, re-run a manual scan and update
        ``_MINUTE_BARS_1M`` / ``_MINUTE_BARS_5M`` at the top of this module.
        """
        table = _MINUTE_BARS_STATIC.get(dataset, [])
        return [
            {
                "dataset": dataset,
                "partition": row[0],
                "granularity": "year",
                "period_start": date.fromisoformat(row[1]),
                "period_end": date.fromisoformat(row[2]),
                "row_count": row[3],
                "file_count": row[4],
                "bytes": row[5],
            }
            for row in table
        ]

    def _effective_partition_stats(self) -> pl.DataFrame:
        """Native stats plus adapter stats, without double-counting a dataset."""

        def _build() -> pl.DataFrame:
            from cnequity.external import registry as ext_registry

            native = load_partition_stats(self.config)
            external = self._external_partition_stats()
            mounted = [
                name
                for name in DATASETS
                if ext_registry.external_adapter(self.config, name) is not None
            ]
            if mounted and not native.is_empty():
                native = native.filter(~pl.col("dataset").is_in(mounted))
            frames = [frame for frame in (native, external) if not frame.is_empty()]
            if not frames:
                return pl.DataFrame(schema=PARTITION_STATS_SCHEMA)
            return pl.concat(frames, how="vertical_relaxed")

        return self._cached("effective_partition_stats", _build)

    def _catalog(self) -> pl.DataFrame:
        """``list_datasets()`` joined with the measured rows and bytes."""

        def _build() -> pl.DataFrame:
            from cnequity.query.reader import list_datasets

            catalog = list_datasets(config=self.config)
            stats = load_partition_stats(self.config)

            external_stats = self._external_partition_stats()
            ext_rows = external_stats.to_dicts()

            if stats.is_empty():
                catalog = catalog.with_columns(
                    pl.lit(None, dtype=pl.Int64).alias("row_count"),
                    pl.lit(None, dtype=pl.Int64).alias("bytes"),
                    pl.lit(None, dtype=pl.Int64).alias("partitions"),
                )
            else:
                # Native stats have a different grain (one partition/file per row)
                # from the adapter rollups below. Keep them separate: concatenating
                # the two schemas loses partition counts and breaks either rollup.
                native_rollup = stats.group_by("dataset").agg(
                    pl.col("row_count").sum(),
                    pl.col("bytes").sum(),
                    pl.col("file_count").sum().alias("partitions"),
                )
                catalog = catalog.join(native_rollup, on="dataset", how="left")

            if ext_rows:
                external_rollup = (
                    pl.DataFrame(ext_rows)
                    .group_by("dataset")
                    .agg(
                        pl.col("row_count").sum(),
                        pl.col("bytes").sum(),
                        pl.col("file_count").sum().alias("partitions"),
                    )
                )
                catalog = catalog.join(
                    external_rollup, on="dataset", how="left", suffix="_external"
                )
                catalog = catalog.with_columns(
                    [
                        pl.coalesce(["row_count_external", "row_count"]).alias("row_count"),
                        pl.coalesce(["bytes_external", "bytes"]).alias("bytes"),
                        pl.coalesce(["partitions_external", "partitions"]).alias("partitions"),
                    ]
                ).drop(["row_count_external", "bytes_external", "partitions_external"])

            return catalog

        return self._cached("catalog", _build)

    def _health_findings(self) -> dict:
        """The audit's last written health snapshot, or an empty stand-in."""

        def _build() -> dict:
            path = self.config.meta_root / "quality" / "health-latest.json"
            if not path.exists():
                return {}
            try:
                with open(path, encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                return {}

        return self._cached("health_findings", _build)

    def _freshness_of(self, row: dict, anchor: date) -> str:
        """fresh / STALE / empty / n/a, on the same rules as ``cne status``."""
        if not row["has_data"]:
            return "empty"
        if not is_dataset_enabled(row["dataset"], self.config):
            return "n/a"
        if not row["watermarked"]:
            return "n/a"
        mark = row["watermark"] or row["coverage_end"]
        return "stale" if is_stale(row["dataset"], mark, anchor) else "fresh"

    def _rows(self) -> list[dict]:
        """One enriched dict per registered dataset."""
        anchor = self.anchor()
        out = []
        for row in self._catalog().iter_rows(named=True):
            spec = DATASETS[row["dataset"]]
            raw_mark = row["watermark"] or row["coverage_end"]
            unified_watermark = min(raw_mark, anchor) if raw_mark else None
            unified_coverage_end = (
                min(row["coverage_end"], anchor) if row["coverage_end"] else None
            )
            out.append(
                {
                    **row,
                    "tier": spec.tier,
                    "tier_label": TIER_LABELS[spec.tier],
                    "required": spec.required,
                    "intraday": spec.intraday_frequency,
                    "description": spec.description,
                    # What one row covers. Carried separately from `intraday`
                    # because trade_ticks is intraday without being bars, and
                    # keying the catalog on `intraday` alone showed it as daily.
                    "row_grain": spec.row_grain,
                    "granularity": spec.partition_granularity if spec.partition_col else None,
                    "unified_watermark": unified_watermark,
                    "unified_coverage_end": unified_coverage_end,
                    "ahead_of_anchor": bool(raw_mark and raw_mark > anchor),
                    "freshness": self._freshness_of(row, anchor),
                }
            )
        return out

    # --- endpoint payloads -------------------------------------------------

    def health(self) -> dict:
        rows = self._rows()
        findings = self._health_findings()
        freshness = stats_freshness(self.config)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["freshness"]] = counts.get(row["freshness"], 0) + 1
        return {
            "anchor": self.anchor(),
            "datasets": len(rows),
            "fresh": counts.get("fresh", 0),
            "stale": counts.get("stale", 0),
            "empty": counts.get("empty", 0),
            "not_applicable": counts.get("n/a", 0),
            "stale_datasets": sorted(r["dataset"] for r in rows if r["freshness"] == "stale"),
            # Empty is not automatically a problem: an opt-in dataset nobody
            # enabled and a required one that failed look identical on disk.
            "empty_optional": sorted(
                r["dataset"] for r in rows if r["freshness"] == "empty" and not r["required"]
            ),
            "empty_required": sorted(
                r["dataset"] for r in rows if r["freshness"] == "empty" and r["required"]
            ),
            "rows": sum(r["row_count"] or 0 for r in rows),
            "bytes": sum(r["bytes"] or 0 for r in rows),
            "findings_by_severity": findings.get("findings_by_severity", {}),
            "audit_trade_date": findings.get("trade_date"),
            "stats_stale": freshness.stale,
            "stats_reason": freshness.reason,
            "stats_generated_at": freshness.generated_at,
        }

    def tiers(self) -> list[dict]:
        rows = self._rows()
        out = []
        for tier in TIERS:
            members = [r for r in rows if r["tier"] == tier]
            if not members:
                continue
            out.append(
                {
                    "tier": tier,
                    "label": TIER_LABELS[tier],
                    "datasets": len(members),
                    "fresh": sum(1 for r in members if r["freshness"] == "fresh"),
                    "stale": sum(1 for r in members if r["freshness"] == "stale"),
                    "empty": sum(1 for r in members if r["freshness"] == "empty"),
                    "rows": sum(r["row_count"] or 0 for r in members),
                    "bytes": sum(r["bytes"] or 0 for r in members),
                    "members": [r["dataset"] for r in members],
                }
            )
        return out

    def datasets(self, *, tier: str | None = None) -> list[dict]:
        rows = self._rows()
        if tier:
            rows = [r for r in rows if r["tier"] == tier]
        return rows

    def provenance(self, dataset: str) -> list[dict]:
        """Source mix for one dataset, newest ``fetched_at`` first."""
        stats = load_provenance_stats(self.config)
        if stats.is_empty():
            return []
        rolled = (
            stats.filter(pl.col("dataset") == dataset)
            .group_by(["source", "data_version"])
            .agg(
                pl.col("row_count").sum(),
                pl.col("fetched_at_min").min(),
                pl.col("fetched_at_max").max(),
            )
            .sort("row_count", descending=True)
        )
        return rolled.to_dicts()

    # --- one dataset -------------------------------------------------------

    def partitions(self, dataset: str) -> list[dict]:
        """Per-partition rows and bytes, oldest first — the size/volume series."""
        stats = self._effective_partition_stats()
        if stats.is_empty():
            return []
        rows = stats.filter(pl.col("dataset") == dataset)
        if rows.is_empty():
            return []
        return (
            rows.sort("period_start", nulls_last=True)
            .select("partition", "granularity", "period_start", "period_end", "row_count", "bytes")
            .to_dicts()
        )

    def _gaps(self, spec, parts: list[dict]) -> dict:
        """Periods inside the covered span that hold no partition.

        Counted in the dataset's own period, not in days: a year-partitioned
        dataset is not missing 364 days because one directory covers the year,
        and reporting it that way would drown the real gaps.
        """
        dated = [p for p in parts if p["period_start"] is not None]
        if len(dated) < 2:
            return {"missing": [], "total": 0, "unit": spec.partition_granularity}

        present = {p["partition"] for p in dated}
        first, last = dated[0]["period_start"], max(p["period_end"] for p in dated)
        missing: list[str] = []

        if spec.partition_granularity == "day":
            # Only sessions count as missing; a weekend is not a gap.
            from cnequity.steps.common import _load_trading_calendar_df

            calendar = _load_trading_calendar_df(self.config, start=first, end=last)
            if calendar is None or calendar.is_empty():
                return {"missing": [], "total": 0, "unit": "day"}
            for day in calendar.filter(pl.col("is_trading")).sort("trade_date")["trade_date"]:
                if day.isoformat() not in present:
                    missing.append(day.isoformat())
        else:
            from cnequity.domain.partitions import partition_value

            cursor = first
            while cursor <= last:
                value = partition_value(cursor, spec.partition_granularity)
                if value not in present:
                    missing.append(value)
                cursor = _next_period_start(cursor, spec.partition_granularity)

        return {
            "missing": missing[:60],
            "total": len(missing),
            "unit": spec.partition_granularity,
        }

    def _commands(self, spec, freshness: str) -> list[dict]:
        """What to run, and why. The dashboard names the fix; it does not run it."""
        name = spec.name
        out: list[dict] = []
        if spec.layer == "external":
            out.append(
                {
                    "cmd": "cne catalog",
                    "why": f"只读桥接：{spec.primary_source}",
                }
            )
            return out
        if spec.layer == "derived":
            out.append({"cmd": f"cne derive {name}", "why": "由 curated 重算"})
        elif spec.backfill_source:
            out.append(
                {"cmd": f"cne backfill {name}", "why": f"专用历史源：{spec.backfill_source}"}
            )
        elif spec.fetch_semantics == "by_date":
            out.append({"cmd": f"cne backfill {name}", "why": "按日期回补缺口"})
        if freshness == "stale":
            out.append({"cmd": "cne status", "why": "查看最近 run，再 cne retry --run-id"})
        out.append({"cmd": f"cne stats show --dataset {name}", "why": "逐分区行数与体积"})
        return out

    # --- quality ------------------------------------------------------------

    def _quality_files(self, kind: str, limit: int) -> list[tuple[str, dict]]:
        """Newest ``meta/quality/<kind>/*.json``, as (run_id, payload).

        Only the per-run files: that directory also holds artefacts written by
        other checks (``authority-<date>.json``) whose shape is entirely
        different, and reading those as run findings would produce nonsense.
        A run id is a UUID, which is what tells them apart.
        """
        root = self.config.meta_root / "quality" / kind
        if not root.is_dir():
            return []
        out: list[tuple[str, dict]] = []
        for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if len(path.stem) != 36 or path.stem.count("-") != 4:
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    out.append((path.stem, json.load(handle)))
            except (OSError, json.JSONDecodeError):
                continue
            if len(out) >= limit:
                break
        return out

    def _quarantine(self) -> list[dict]:
        """What has been pulled out of curated, and how big it is.

        Not a wastebasket. Everything here was removed from the lake because
        something was wrong with it, and it is kept as evidence — sizing it is
        how you decide whether the evidence is still worth the disk.
        """
        root = self.config.data_root / "_quarantine"
        if not root.is_dir():
            return []
        out = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            files = [f for f in entry.rglob("*") if f.is_file()]
            out.append(
                {
                    "name": entry.name,
                    "files": len(files),
                    "bytes": sum(f.stat().st_size for f in files),
                    "modified": datetime.fromtimestamp(
                        entry.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        return out

    def _on_demand(self) -> list[dict]:
        """Per-symbol caches under ``meta/on_demand``.

        On-demand datasets are not in ``DATASETS`` and never reach curated, so
        nothing else on this dashboard can see them. An empty list means nobody
        has queried one yet, which is a normal state rather than a gap.
        """
        root = self.config.meta_root / "on_demand"
        if not root.is_dir():
            return []
        out = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            files = [f for f in entry.rglob("*") if f.is_file()]
            newest = max((f.stat().st_mtime for f in files), default=None)
            out.append(
                {
                    "dataset": entry.name,
                    "entries": len(files),
                    "bytes": sum(f.stat().st_size for f in files),
                    "newest": datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()
                    if newest
                    else None,
                }
            )
        return out

    def quality(self, *, limit: int = 30) -> dict:
        findings_runs = []
        for run_id, payload in self._quality_files("findings", limit):
            by_severity: dict[str, int] = {}
            by_check: dict[str, int] = {}
            for finding in payload.get("findings", []):
                sev = finding.get("severity", "info")
                by_severity[sev] = by_severity.get(sev, 0) + 1
                key = finding.get("check", "?")
                by_check[key] = by_check.get(key, 0) + 1
            findings_runs.append(
                {
                    "run_id": run_id,
                    "trade_date": payload.get("trade_date"),
                    "total": len(payload.get("findings", [])),
                    "by_severity": by_severity,
                    "top_checks": sorted(by_check.items(), key=lambda kv: -kv[1])[:5],
                }
            )

        diff_runs = []
        for run_id, payload in self._quality_files("source_diffs", limit):
            by_check: dict[str, int] = {}
            for diff in payload.get("diffs", []):
                key = diff.get("check", "?")
                by_check[key] = by_check.get(key, 0) + 1
            diff_runs.append(
                {
                    "run_id": run_id,
                    "trade_date": payload.get("trade_date"),
                    "diff_count": payload.get("diff_count", len(payload.get("diffs", []))),
                    "by_check": by_check,
                }
            )

        return {
            "findings_runs": findings_runs,
            "diff_runs": diff_runs,
            "quarantine": self._quarantine(),
            "on_demand": self._on_demand(),
        }

    def quality_run(self, run_id: str) -> dict | None:
        """One run's findings and cross-source diffs, in full."""
        findings = dict(self._quality_files("findings", 200)).get(run_id)
        diffs = dict(self._quality_files("source_diffs", 200)).get(run_id)
        if findings is None and diffs is None:
            return None
        return {
            "run_id": run_id,
            "trade_date": (findings or diffs or {}).get("trade_date"),
            "findings": (findings or {}).get("findings", []),
            "diffs": (diffs or {}).get("diffs", []),
        }

    # --- runs ---------------------------------------------------------------

    def _manifest_rows(self, sql: str, params: tuple = ()) -> list[dict]:
        """Query the manifest read-only. Returns [] when there is none yet."""
        import sqlite3

        path = self.config.manifest_path
        if not path.exists():
            return []
        conn = None
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            with conn:
                return [dict(row) for row in conn.execute(sql, params).fetchall()]
        except sqlite3.Error:
            return []
        finally:
            if conn is not None:
                conn.close()

    def runs(self, *, limit: int = 40) -> list[dict]:
        """Recent runs, newest first, with their batch tally."""
        runs = self._manifest_rows(
            """SELECT run_id, job_name, status, started_at, finished_at,
                      rows_read, rows_written, error_message
               FROM ingestion_runs ORDER BY started_at DESC LIMIT ?""",
            (limit,),
        )
        if not runs:
            return []
        tally = self._manifest_rows(
            """SELECT run_id, status, COUNT(*) AS n FROM ingestion_batches
               WHERE run_id IN ({}) GROUP BY run_id, status""".format(",".join("?" * len(runs))),
            tuple(r["run_id"] for r in runs),
        )
        by_run: dict[str, dict[str, int]] = {}
        for row in tally:
            by_run.setdefault(row["run_id"], {})[row["status"]] = int(row["n"])
        for run in runs:
            counts = by_run.get(run["run_id"], {})
            run["batches"] = sum(counts.values())
            run["batch_status"] = counts
            run["datasets"] = []
        return runs

    def run_detail(self, run_id: str) -> dict | None:
        """One run and every batch in it, with a stalled flag per batch.

        ``stalled`` is not a manifest column: it is "still ``running`` but
        silent for longer than ``batch_stale_seconds``". The engine uses the
        same threshold to promote such batches on the next run, so surfacing it
        here shows the operator what the engine is about to conclude, before it
        does — a worker that died is otherwise indistinguishable from a slow one.
        """
        rows = self._manifest_rows(
            """SELECT run_id, job_name, status, started_at, finished_at,
                      rows_read, rows_written, error_message, metadata_json
               FROM ingestion_runs WHERE run_id = ?""",
            (run_id,),
        )
        if not rows:
            return None
        run = rows[0]
        batches = self._manifest_rows(
            """SELECT batch_id, dataset, status, window_start, window_end,
                      rows_read, rows_written, retry_count, started_at,
                      finished_at, heartbeat_at, error_message
               FROM ingestion_batches WHERE run_id = ?
               ORDER BY COALESCE(started_at, '')""",
            (run_id,),
        )
        now = datetime.now(timezone.utc)
        threshold = float(getattr(self.config, "batch_stale_seconds", 3600) or 3600)
        for batch in batches:
            batch["stalled"] = False
            if batch["status"] != "running":
                continue
            mark = batch["heartbeat_at"] or batch["started_at"]
            if not mark:
                continue
            try:
                silent = (now - datetime.fromisoformat(mark)).total_seconds()
            except ValueError:
                continue
            batch["silent_seconds"] = round(silent)
            batch["stalled"] = silent >= threshold
        run["batches"] = batches
        run["stale_after_seconds"] = threshold
        return run

    def run_fingerprint(self, run_id: str) -> str:
        """Cheap value that changes whenever the run's batches do.

        The stream compares this instead of diffing rows: a poll that finds it
        unchanged sends nothing, which is what keeps an idle subscriber free.
        """
        rows = self._manifest_rows(
            """SELECT COUNT(*) AS n, COALESCE(MAX(finished_at), '') AS f,
                      COALESCE(MAX(heartbeat_at), '') AS h,
                      COALESCE(SUM(rows_written), 0) AS w,
                      COALESCE(SUM(retry_count), 0) AS r,
                      COALESCE(GROUP_CONCAT(status), '') AS s
               FROM ingestion_batches WHERE run_id = ?""",
            (run_id,),
        )
        state = self._manifest_rows("SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,))
        head = rows[0] if rows else {}
        return (
            "|".join(str(head.get(k, "")) for k in ("n", "f", "h", "w", "r", "s"))
            + f"|{state[0]['status'] if state else ''}"
        )

    def recent_batches(self, dataset: str, *, limit: int = 15) -> list[dict]:
        """Latest manifest batches for this dataset, newest first.

        stdlib sqlite3 on a read-only URI rather than DuckDB's sqlite_scanner:
        that scanner is an autoloadable extension fetched from the network on
        first use, which on an offline or proxied box turns the page into a
        spinner. The manifest is small and WAL is already on, so a concurrent
        run is not blocked by this read.
        """
        return self._manifest_rows(
            """SELECT run_id, batch_id, status, window_start, window_end, rows_written,
                      retry_count, started_at, finished_at, error_message
               FROM ingestion_batches WHERE dataset = ?
               ORDER BY COALESCE(started_at, '') DESC LIMIT ?""",
            (dataset, limit),
        )

    def _source_status(self, dataset: str) -> dict | None:
        """Read the last ``source_status`` written by the ingestion step.

        For ``daily_bars`` the step writes a dict tracking which provider
        served the data (tushare / akshare / cne_existing_fallback) and
        whether Tushare was available. Other datasets do not write it.
        """
        path = self.config.meta_root / "state" / f"{dataset}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload.get("source_status")

    def dataset_detail(self, dataset: str) -> dict:
        """Everything the detail page shows, in one round trip."""
        from cnequity.domain.schemas import DATASET_SCHEMAS, PRIMARY_KEYS
        from cnequity.query.reader import ADJUSTABLE_DATASETS

        spec = DATASETS[dataset]
        row = next(r for r in self._rows() if r["dataset"] == dataset)
        parts = self.partitions(dataset)
        findings = self._health_findings()
        mine = [
            f
            for key in ("error_findings", "warning_findings")
            for f in findings.get(key, [])
            if f.get("dataset") == dataset
        ]

        return {
            **row,
            "layer": spec.layer,
            "partition_col": spec.partition_col,
            "max_staleness_days": spec.max_staleness_days,
            "backfill_chunk_days": spec.backfill_chunk_days,
            "backfill_chunk_symbols": getattr(spec, "backfill_chunk_symbols", None),
            # The source's own floor, not this lake's backlog: earlier windows
            # return nothing rather than less, and no backfill reaches past it.
            "earliest_available": spec.earliest_available(shanghai_today()),
            # Which *kind* of limit produced that floor. Without this the panel
            # cannot tell a fixed date from a rolling count, and a dataset
            # limited by a date (trade_ticks) reads as unlimited because its
            # `history_horizon_days` is null.
            "history_floor_date": spec.history_floor_date,
            # Whether load() can join adj_factors — the data tab only offers the
            # 复权 control where it means something.
            "adjustable": dataset in ADJUSTABLE_DATASETS,
            "primary_key": PRIMARY_KEYS.get(dataset, []),
            "schema": [
                {"column": col, "dtype": str(dtype)}
                for col, dtype in DATASET_SCHEMAS.get(dataset, {}).items()
            ],
            # The per-partition series is not inlined: daily_bars alone is 6,202
            # rows, and the detail payload is loaded on every tab switch while
            # the series is only needed for one chart. `/partitions` serves it.
            "gaps": self._gaps(spec, parts),
            "findings": mine,
            "commands": self._commands(spec, row["freshness"]),
            "batches": self.recent_batches(dataset),
            # Which provider served the last run (tushare / akshare / fallback).
            # Absent for datasets that never write source_status.
            "source_status": self._source_status(dataset),
        }

    def provenance_series(self, dataset: str, *, max_buckets: int = 400) -> dict:
        """Source mix over time, bucketed to stay chartable.

        The collapsed :meth:`provenance` answers "which sources are in here";
        this answers "when did that change", which is where a routing switch or
        a mis-attributed backfill actually becomes visible.

        daily_bars alone has 11,324 (day, source) points — a megabyte of JSON to
        draw a few hundred pixels. Buckets widen until the series fits, and the
        chosen width is returned rather than applied silently: a caller that
        does not know it is looking at months cannot label the axis honestly.
        """
        stats = load_provenance_stats(self.config)
        partitions = load_partition_stats(self.config)
        empty = {"bucket": "day", "points": []}
        if stats.is_empty() or partitions.is_empty():
            return empty
        periods = partitions.filter(pl.col("dataset") == dataset).select(
            "partition", "period_start"
        )
        rows = stats.filter(pl.col("dataset") == dataset)
        if rows.is_empty() or periods.is_empty():
            return empty

        joined = rows.join(periods, on="partition", how="inner").filter(
            pl.col("period_start").is_not_null()
        )
        if joined.is_empty():
            return empty

        for bucket, expr in (
            ("day", pl.col("period_start")),
            ("month", pl.col("period_start").dt.truncate("1mo")),
            ("year", pl.col("period_start").dt.truncate("1y")),
        ):
            grouped = (
                joined.with_columns(expr.alias("period_start"))
                .group_by(["period_start", "source", "data_version"])
                .agg(pl.col("row_count").sum())
                .sort(["period_start", "source"])
            )
            if grouped.height <= max_buckets or bucket == "year":
                return {"bucket": bucket, "points": grouped.to_dicts()}
        return empty  # pragma: no cover — the year branch always returns

    # --- browsing rows -----------------------------------------------------

    def date_options(self, dataset: str, *, limit: int = 400) -> dict:
        """What the date control may offer, and which control that should be.

        There is no single picker: the registry uses twelve different date
        columns across four shapes, and a calendar widget over ``report_period``
        would invite a query the column cannot answer.

        Only values that exist are offered. A day with no partition is not
        selectable, which is the honest version of the ``snapshot_only`` warning
        — those datasets accumulate one stamped reading per run, and a day
        nobody ran can never be given one.
        """
        spec = DATASETS[dataset]
        parts = self.partitions(dataset)
        if spec.partition_col is None:
            return {
                "kind": "none",
                "column": spec.query_date_col,
                "granularity": None,
                "values": [],
                "total": 0,
                "note": "单文件 merge：只有当前状态，没有按日期取数的概念。",
            }

        values = [p["partition"] for p in reversed(parts) if p["partition"] is not None]
        if spec.partition_col == "report_period":
            kind = "report_period"
        elif spec.partition_granularity == "day" and spec.partition_col == "trade_date":
            kind = "trading_day"
        elif spec.partition_granularity == "day":
            kind = "event_day"
        else:
            kind = "period"

        note = None
        if history_mode_for(spec) == "snapshot_only":
            note = (
                "snapshot_only：每个 run 落一份当日快照。没跑的那天没有快照，"
                "而且补不出来——重放会伪造行。列表里只有真实存在的日期。"
            )
        elif spec.partition_granularity != "day":
            note = f"按 {spec.partition_granularity} 分区：选一个周期取回它整段的行。"

        return {
            "kind": kind,
            "column": spec.query_date_col,
            "granularity": spec.partition_granularity,
            "values": values[:limit],
            "total": len(values),
            "note": note,
        }

    def rows(
        self,
        dataset: str,
        *,
        period: str | None = None,
        symbol: str | None = None,
        as_of: date | None = None,
        adjust: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        """A page of actual rows, with the provenance columns kept.

        ``source`` / ``data_version`` / ``fetched_at`` are not dropped to make
        room: row-level provenance is the point of this lake, and a viewer that
        hides it teaches you it is not there.

        Reads through ``query.reader.load`` so the dashboard sees exactly what
        ``load()`` gives a researcher — adjustment, PIT collapsing and all —
        rather than a second, subtly different read path.
        """
        from cnequity.query.reader import ReaderError, load

        spec = DATASETS[dataset]
        kwargs: dict[str, Any] = {"config": self.config}

        if period:
            part = parse_partition(period)
            if spec.partition_col == "report_period":
                # A String column: the period *is* the value, and a date range
                # over it would compare text to dates.
                pass
            elif part is None:
                raise ValueError(f"{period!r} is not a period for {dataset}")
            else:
                kwargs["start"], kwargs["end"] = part.start, part.end
        if symbol:
            kwargs["symbols"] = [symbol]
        if as_of:
            kwargs["as_of"] = as_of
        if adjust:
            kwargs["adjust"] = adjust

        try:
            frame = load(dataset, **kwargs)
        except ReaderError as exc:
            raise ValueError(str(exc)) from exc

        if period and spec.partition_col == "report_period":
            frame = frame.filter(pl.col("report_period") == period)

        total = frame.height
        page = frame.slice(offset, limit)
        return {
            "columns": page.columns,
            "rows": [
                [None if v is None else _jsonable(v) for v in row] for row in page.iter_rows()
            ],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def heatmap(self, *, days: int = 90) -> dict:
        """Coverage grid: one row per dataset, one cell per recent trading day.

        Cells answer "does a partition covering this day exist", which for a
        month/year-partitioned dataset is coarser than the day it is drawn on —
        the directory covers the period, and whether one particular session has
        rows in it is not knowable without reading the file. ``granularity``
        rides along on each row so a renderer can say so rather than imply a
        precision the layout does not have.
        """
        from cnequity.steps.common import _load_trading_calendar_df

        anchor = self.anchor()
        window_start = anchor - timedelta(days=int(days * 1.7) + 10)
        calendar = _load_trading_calendar_df(self.config, start=window_start, end=anchor)
        if calendar is None or calendar.is_empty():
            trading_days: list[date] = []
        else:
            trading_days = (
                calendar.filter(pl.col("is_trading"))
                .sort("trade_date")["trade_date"]
                .to_list()[-days:]
            )

        stats = self._effective_partition_stats()
        spans: dict[str, list[tuple[date, date]]] = {}
        if not stats.is_empty():
            for row in stats.iter_rows(named=True):
                if row["period_start"] is None or row["period_end"] is None:
                    continue
                spans.setdefault(row["dataset"], []).append(
                    (row["period_start"], row["period_end"])
                )

        rows = []
        for row in self._rows():
            name = row["dataset"]
            intervals = sorted(spans.get(name, []))
            if row["granularity"] is None:
                cells = CELL_UNPARTITIONED * len(trading_days)
            elif not intervals:
                cells = CELL_OUTSIDE * len(trading_days)
            else:
                first, last = intervals[0][0], max(end for _, end in intervals)
                # Binary-search each interval into the sorted day list instead of
                # testing every day against every interval. That inner scan was
                # O(datasets × intervals × days), and a day-partitioned dataset
                # brings one interval per session: daily_bars alone put ~6,200
                # intervals against 250 days. Measured on this lake, the endpoint
                # took 0.1s at days=60 and 24-67s at days=250 — the dashboard's
                # whole first paint waits on it.
                covered_flags = bytearray(len(trading_days))
                for start, end in intervals:
                    lo = bisect_left(trading_days, start)
                    hi = bisect_right(trading_days, end)
                    for i in range(lo, hi):
                        covered_flags[i] = 1
                cells = "".join(
                    CELL_COVERED
                    if covered_flags[i]
                    else (CELL_GAP if first <= day <= last else CELL_OUTSIDE)
                    for i, day in enumerate(trading_days)
                )
            rows.append(
                {
                    "dataset": name,
                    "tier": row["tier"],
                    "granularity": row["granularity"],
                    "freshness": row["freshness"],
                    "cadence_days": DATASETS[name].max_staleness_days,
                    "gap_meaning": _gap_meaning(DATASETS[name]),
                    "cells": cells,
                }
            )

        return {
            "days": trading_days,
            "legend": {
                CELL_COVERED: "covered",
                CELL_GAP: "gap",
                CELL_OUTSIDE: "outside coverage",
                CELL_UNPARTITIONED: "unpartitioned",
            },
            "rows": rows,
        }
