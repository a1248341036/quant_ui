"""Universe filtering for the query reader."""

from __future__ import annotations

from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.domain.symbols import (
    CDR_PREFIXES,
    ETF_PREFIXES,
    EXCLUDED_PREFIXES,
    PREFIX_WHITELIST,
)
from cnequity.query.canonical import dedupe_by_primary_key, dedupe_lazy_by_primary_key
from cnequity.query.parquet_scan import (
    collect_parquet_root,
    coverage_start_from_partitions,
    list_partitions,
    scan_parquet_root,
)

EXCLUDED_STATUSES = frozenset({"st", "*st", "suspended"})


class UniverseCoverageError(ValueError):
    """Raised when a strict universe query lacks required coverage."""


def _all_a_symbol_expr(symbol_col: str = "symbol") -> pl.Expr:
    code = pl.col(symbol_col).str.split(".").list.first()
    exchange = pl.col(symbol_col).str.split(".").list.last()
    excluded = pl.lit(False)
    # 81–89 is a SH/SZ reservation; do not apply it to BJ (legacy 83xxxx).
    for prefix in EXCLUDED_PREFIXES:
        excluded = excluded | (exchange.is_in(["SH", "SZ"]) & code.str.starts_with(prefix))
    # CDRs (SH 689xxx) are depositary receipts, not common stock; they stay in
    # the lake but are not part of the all_a selection universe.
    for prefix in CDR_PREFIXES:
        excluded = excluded | ((exchange == "SH") & code.str.starts_with(prefix))
    # ETFs/LOFs stay in instruments + daily_bars for UI/quotes, but never enter
    # the all_a research universe (PREFIX_WHITELIST also omits them).
    for exch, prefixes in ETF_PREFIXES.items():
        for prefix in prefixes:
            excluded = excluded | ((exchange == exch) & code.str.starts_with(prefix))
    allowed = pl.lit(False)
    for exch, prefixes in PREFIX_WHITELIST.items():
        for prefix in prefixes:
            allowed = allowed | ((exchange == exch) & code.str.starts_with(prefix))
    return (~excluded) & allowed


def _load_instruments(config: Config) -> pl.DataFrame:
    root = config.curated_root / "instruments"
    if not root.exists():
        return pl.DataFrame()
    try:
        return dedupe_by_primary_key(
            collect_parquet_root(root, hive=False),
            "instruments",
        )
    except FileNotFoundError:
        return pl.DataFrame()


def _load_trading_status(
    config: Config,
    *,
    trade_date: date | None = None,
) -> pl.DataFrame:
    root = config.curated_root / "trading_status"
    if not root.exists():
        return pl.DataFrame()
    try:
        return dedupe_by_primary_key(
            collect_parquet_root(
                root,
                partition_col="trade_date",
                start=trade_date,
                end=trade_date,
            ),
            "trading_status",
        )
    except FileNotFoundError:
        return pl.DataFrame()


def coverage_start_date(
    config: Config,
    dataset: str,
    *,
    date_col: str = "trade_date",
) -> date | None:
    """Earliest *date_col* present in curated *dataset*, if any."""
    root = config.curated_root / dataset
    if not root.exists():
        return None

    # A day partition's directory value is the exact date it can contain. A
    # month/year/quarter directory is only a container, though: using its
    # theoretical start would claim coverage before the first real row. Mixed
    # layouts and loose root-level files need the same exact scan.
    parts = list_partitions(root, date_col)
    if (
        dataset != "daily_bars"
        and parts
        and all(part.start == part.end for part in parts)
        and not list(root.glob("*.parquet"))
    ):
        return coverage_start_from_partitions(root, date_col)
    try:
        scan = scan_parquet_root(
            root,
            partition_col=date_col,
            traded_only=dataset == "daily_bars" and date_col == "trade_date",
        )
        return scan.select(pl.col(date_col).min()).collect().item()
    except FileNotFoundError:
        return None


def coverage_end_date(
    config: Config,
    dataset: str,
    *,
    date_col: str = "trade_date",
) -> date | None:
    """Latest *date_col* present in curated *dataset*, if any.

    Partition directory bounds are exact only for day partitions. Coarser or
    mixed layouts must use the date stored in the row; otherwise a partial
    current month/year can make an incomplete research window look complete.
    """
    root = config.curated_root / dataset
    if not root.exists():
        return None
    parts = list_partitions(root, date_col)
    if (
        dataset != "daily_bars"
        and parts
        and all(part.start == part.end for part in parts)
        and not list(root.glob("*.parquet"))
    ):
        return parts[-1].end
    try:
        scan = scan_parquet_root(
            root,
            partition_col=date_col,
            traded_only=dataset == "daily_bars" and date_col == "trade_date",
        )
        return scan.select(pl.col(date_col).max()).collect().item()
    except FileNotFoundError:
        return None


def trading_status_coverage_start(config: Config) -> date | None:
    """First trade_date with curated trading_status rows."""
    return coverage_start_date(config, "trading_status")


def st_coverage_start(config: Config) -> date | None:
    """First trade_date with an ``st``/``*st`` label in trading_status.

    Suspension is reconstructed from bar gaps (whole history), but ST labels
    come only from live ST snapshots, so this is the real ST-history boundary.
    """
    root = config.curated_root / "trading_status"
    if not root.exists():
        return None
    try:
        return (
            scan_parquet_root(root, partition_col="trade_date")
            .filter(pl.col("status").is_in(["st", "*st"]))
            .select(pl.col("trade_date").min())
            .collect()
            .item()
        )
    except FileNotFoundError:
        return None


def tradable_symbols_on_date(
    config: Config,
    trade_date: date,
    *,
    universe: str = "all_a",
    strict: bool = False,
) -> pl.DataFrame | None:
    """Return ``symbol`` rows tradable on *trade_date* for the given universe rule.

    Applies list/delist dates from ``instruments``. ST/suspended filtering uses
    ``trading_status`` only when rows exist for *trade_date*; dates before
    :func:`trading_status_coverage_start` are not ST-filtered.
    """
    if universe != "all_a":
        raise ValueError(f"unsupported universe: {universe!r} (supported: 'all_a')")

    instruments = _load_instruments(config)
    if instruments.is_empty():
        if strict:
            raise UniverseCoverageError("all_a universe requires curated instruments")
        return None

    out = (
        instruments.filter(_all_a_symbol_expr())
        .filter(pl.col("list_date").is_null() | (pl.col("list_date") <= trade_date))
        .filter(pl.col("delist_date").is_null() | (pl.col("delist_date") >= trade_date))
        .select("symbol")
    )
    if out.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8})

    status = _load_trading_status(config, trade_date=trade_date)
    if status.is_empty():
        if strict:
            raise UniverseCoverageError(
                f"all_a universe has no trading_status coverage for {trade_date.isoformat()}"
            )
        return out

    if strict:
        covered_symbols = set(status.get_column("symbol").drop_nulls().to_list())
        missing_symbols = sorted(set(out.get_column("symbol").to_list()) - covered_symbols)
        if missing_symbols:
            raise UniverseCoverageError(
                f"all_a universe trading_status is missing {len(missing_symbols)} "
                f"symbol(s) for {trade_date.isoformat()}, first={missing_symbols[0]}"
            )

    bad = status.filter((~pl.col("is_trading")) | pl.col("status").is_in(list(EXCLUDED_STATUSES)))[
        "symbol"
    ]
    if not bad.is_empty():
        out = out.filter(~pl.col("symbol").is_in(bad))
    return out


def apply_universe_filter(
    df: pl.DataFrame,
    config: Config,
    *,
    universe: str,
    date_col: str = "trade_date",
    strict: bool = False,
) -> pl.DataFrame:
    """Filter bar-like frames to tradable universe rows per *date_col*.

    ``instruments`` list/delist rules always apply. ST/suspended removal via
    ``trading_status`` only affects dates with status rows; earlier history
    passes through unchanged (see :func:`trading_status_coverage_start`).
    """
    if df.is_empty() or universe != "all_a":
        return df
    if date_col not in df.columns:
        raise ValueError(
            f"apply_universe_filter requires date column {date_col!r} for universe filtering"
        )

    instruments = _load_instruments(config)
    if instruments.is_empty():
        if strict:
            raise UniverseCoverageError("all_a universe requires curated instruments")
        return df

    valid_symbols = instruments.filter(_all_a_symbol_expr())["symbol"]
    inst = instruments.select(["symbol", "list_date", "delist_date"])
    df = df.join(inst, on="symbol", how="left")
    df = df.filter(
        pl.col("list_date").is_null() | (pl.col("list_date") <= pl.col(date_col))
    ).filter(pl.col("delist_date").is_null() | (pl.col("delist_date") >= pl.col(date_col)))
    # An existing instruments catalog with zero valid all-A symbols is a real
    # empty universe (for example an ETF/CDR-only fixture), not a reason to
    # bypass the filter and leak every input row through.
    df = df.filter(pl.col("symbol").is_in(valid_symbols.to_list()))

    try:
        status = dedupe_lazy_by_primary_key(
            scan_parquet_root(
                config.curated_root / "trading_status",
                partition_col="trade_date",
            ),
            "trading_status",
        )
    except FileNotFoundError:
        if strict:
            raise UniverseCoverageError("all_a universe requires curated trading_status") from None
        return df.drop(["list_date", "delist_date"], strict=False)

    if strict and not df.is_empty():
        requested = df.select(["symbol", date_col]).unique()
        covered = status.select(["symbol", pl.col("trade_date").alias(date_col)]).unique().collect()
        missing = requested.join(covered, on=["symbol", date_col], how="anti")
        if not missing.is_empty():
            missing_dates = sorted(missing[date_col].unique().to_list())
            missing_symbols = sorted(missing["symbol"].unique().to_list())
            raise UniverseCoverageError(
                f"all_a universe trading_status missing {missing.height} "
                f"symbol-date row(s) across {len(missing_dates)} date(s), "
                f"first={missing_symbols[0]}@{missing_dates[0].isoformat()}"
            )

    bad = (
        status.filter((~pl.col("is_trading")) | pl.col("status").is_in(list(EXCLUDED_STATUSES)))
        .select(["symbol", pl.col("trade_date").alias(date_col)])
        .collect()
    )
    if bad.is_empty():
        return df.drop(["list_date", "delist_date"], strict=False)

    df = df.join(bad, on=["symbol", date_col], how="anti")
    return df.drop(["list_date", "delist_date"], strict=False)
