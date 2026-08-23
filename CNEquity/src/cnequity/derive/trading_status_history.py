"""Reconstruct historical suspension status from daily_bars trading gaps.

For feeds that omit suspended rows, a listed symbol with no bar on a trading
day was suspended that day. Some lake sources retain an OHLC placeholder on a
suspended day, however; those rows carry ``volume=0``. A listed symbol with no
*traded* bar is therefore the actual evidence used here. This is authoritative
and covers the whole bar history — filling the trading_status gap that free ST
feeds (EastMoney's current-snapshot ST board) cannot reach.

Only sparse ``suspended`` rows are written; real daily rows (EastMoney) win on
any primary-key overlap.

Optional ``start`` / ``end`` bound the calendar cross-join so a 2001→today
rebuild can run year-by-year without OOM.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from cnequity.config import Config
from cnequity.domain.datasets import DATASETS
from cnequity.domain.schemas import validate_dataframe, with_provenance
from cnequity.query.canonical import dedupe_by_primary_key
from cnequity.query.parquet_scan import dataset_has_parquet, scan_parquet_root
from cnequity.storage.parquet import CuratedWriter

logger = logging.getLogger(__name__)

_DERIVED_SOURCE = "derived_bar_gap"
_STATUS_SPEC = DATASETS["trading_status"]
_CURRENT_SNAPSHOT_SOURCES = frozenset({"eastmoney", "tdx_protocol"})
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SESSION_FINAL = time(15, 0)


def _as_shanghai_datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo("UTC"))
    return value.astimezone(_SHANGHAI)


def status_evidence_rank(row: Mapping[str, Any]) -> int:
    """Precedence for a collision with a derived bar-gap suspension.

    Historical Baostock evidence and a finalized same-session EastMoney
    snapshot are point-in-time facts. A later current-state snapshot stamped
    onto an older date is not. Unknown sources win conservatively so a newly
    introduced authority is never overwritten without an explicit policy.
    """
    source = str(row.get("source") or "")
    if source == "baostock":
        return 0
    if source in _CURRENT_SNAPSHOT_SOURCES:
        fetched = _as_shanghai_datetime(row.get("fetched_at"))
        trade_date = row.get("trade_date")
        if (
            fetched is not None
            and isinstance(trade_date, date)
            and fetched.date() == trade_date
            and fetched.time() >= _SESSION_FINAL
        ):
            return 0
        return 2
    if source == _DERIVED_SOURCE:
        return 1
    return 0


def _suspended_pairs(
    config: Config,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """(symbol, trade_date) that were trading days in a symbol's active range but have no bar."""
    bars_root = config.curated_root / "daily_bars"
    cal_root = config.curated_root / "trading_calendar"
    inst_root = config.curated_root / "instruments"
    if not (
        dataset_has_parquet(bars_root)
        and dataset_has_parquet(cal_root)
        and dataset_has_parquet(inst_root)
    ):
        return pl.DataFrame(schema={"symbol": pl.Utf8, "trade_date": pl.Date})

    # Lifetime bounds may include zero-volume suspension placeholders, but a
    # symbol represented only by placeholders must not get a synthetic active
    # range and be reported as suspended on every calendar day. Keep the
    # traded-only scan per file so legacy files without volume retain row-based
    # semantics when they coexist with current files.
    bars_all_lf = scan_parquet_root(bars_root, partition_col="trade_date")
    traded_bars_lf = scan_parquet_root(bars_root, partition_col="trade_date", traded_only=True)
    traded_symbols = traded_bars_lf.select("symbol").unique()
    bars_lf = bars_all_lf.join(traded_symbols, on="symbol", how="semi")
    sym_range = (
        bars_lf.group_by("symbol")
        .agg(
            pl.col("trade_date").min().alias("bmin"),
            pl.col("trade_date").max().alias("bmax"),
        )
        .collect()
    )
    if sym_range.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "trade_date": pl.Date})

    # Anti-join only needs bars inside the derive window.
    # Suspended days may survive as an OHLC placeholder with volume=0. The
    # traded-only scan already removed those rows while retaining legacy rows
    # from files without volume.
    bars_lf = traded_bars_lf.select(["symbol", "trade_date"])
    if start is not None:
        bars_lf = bars_lf.filter(pl.col("trade_date") >= start)
    if end is not None:
        bars_lf = bars_lf.filter(pl.col("trade_date") <= end)
    bars = bars_lf.unique().collect()

    cal_lf = (
        scan_parquet_root(cal_root, partition_col="trade_date")
        .filter(pl.col("is_trading"))
        .select("trade_date")
    )
    if start is not None:
        cal_lf = cal_lf.filter(pl.col("trade_date") >= start)
    if end is not None:
        cal_lf = cal_lf.filter(pl.col("trade_date") <= end)
    cal = cal_lf.collect()
    if cal.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "trade_date": pl.Date})

    inst = dedupe_by_primary_key(
        scan_parquet_root(inst_root, hive=False)
        .select(["symbol", "list_date", "delist_date"])
        .collect(),
        "instruments",
    )

    active = inst.join(sym_range, on="symbol", how="inner").with_columns(
        pl.max_horizontal(pl.col("list_date").fill_null(pl.col("bmin")), pl.col("bmin")).alias(
            "astart"
        ),
        pl.min_horizontal(pl.col("delist_date").fill_null(pl.col("bmax")), pl.col("bmax")).alias(
            "aend"
        ),
    )
    if start is not None:
        active = active.with_columns(
            pl.max_horizontal(pl.col("astart"), pl.lit(start)).alias("astart")
        )
    if end is not None:
        active = active.with_columns(pl.min_horizontal(pl.col("aend"), pl.lit(end)).alias("aend"))
    active = active.filter(pl.col("astart") <= pl.col("aend"))
    if active.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "trade_date": pl.Date})

    expected = (
        active.select(["symbol", "astart", "aend"])
        .join(cal, how="cross")
        .filter(
            (pl.col("trade_date") >= pl.col("astart")) & (pl.col("trade_date") <= pl.col("aend"))
        )
        .select(["symbol", "trade_date"])
    )
    return expected.join(bars, on=["symbol", "trade_date"], how="anti")


def derive_suspension_history(
    config: Config,
    *,
    start: date | None = None,
    end: date | None = None,
) -> int:
    """Write derived ``suspended`` rows into curated trading_status. Returns row count.

    When *start* / *end* are set, only that calendar window is considered — use
    yearly chunks for a full-history rebuild to keep the cross-join bounded.
    """
    pairs = _suspended_pairs(config, start=start, end=end)
    if pairs.is_empty():
        return 0

    rows = pairs.with_columns(
        pl.lit(False).alias("is_trading"),
        pl.lit("suspended").alias("status"),
    )
    rows = with_provenance(rows, source=_DERIVED_SOURCE, data_version="v1")
    rows = validate_dataframe(rows, "trading_status")

    # trading_status is month-partitioned — never write day dirs that fight the
    # registry (audit: mixed_partition_granularity) and republish PKs.
    writer = CuratedWriter(config.curated_root)
    pcol = _STATUS_SPEC.partition_col or "trade_date"
    part_vals = [
        _STATUS_SPEC.partition_for(td) if isinstance(td, date) else str(td)
        for td in rows["trade_date"].to_list()
    ]
    rows = rows.with_columns(pl.Series("_part", part_vals))
    total = 0
    for key, group in rows.partition_by("_part", as_dict=True).items():
        val = str(key[0] if isinstance(key, tuple) else key)
        group = group.drop("_part")
        existing_dir = writer.partition_path("trading_status", pcol, val)
        frames = [group]
        stray_parts: list = []
        if existing_dir.exists():
            for f in existing_dir.rglob("*.parquet"):
                frames.append(validate_dataframe(pl.read_parquet(f), "trading_status"))
                if f.name != "part-merged.parquet":
                    stray_parts.append(f)
        merged = pl.concat(frames, how="diagonal_relaxed")
        if "fetched_at" not in merged.columns:
            merged = merged.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("fetched_at"))
        merged = merged.with_columns(
            pl.struct(["source", "trade_date", "fetched_at"])
            .map_elements(status_evidence_rank, return_dtype=pl.Int64)
            .alias("_evidence_rank")
        ).sort("_evidence_rank")
        merged = merged.unique(subset=["symbol", "trade_date"], keep="first").drop("_evidence_rank")
        # Reuse compact's filename so we overwrite the single canonical part
        # rather than adding a second file (which would double-count on read).
        writer.write_partition("trading_status", pcol, val, merged, "part-merged.parquet")
        for stray in stray_parts:
            stray.unlink(missing_ok=True)
        total += group.height
    logger.info("derived %d historical suspension rows into trading_status", total)
    return total
