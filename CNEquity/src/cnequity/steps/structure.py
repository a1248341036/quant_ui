"""L5 structure steps: sector members, index constituents, industry members."""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from cnequity.adapters.cni.index_constituents_history import (
    CNI_BACKFILL_INDICES,
    expand_cni_constituents_as_of,
    fetch_cni_index_adjustments,
)
from cnequity.adapters.eastmoney.index_constituents import fetch_index_constituents
from cnequity.adapters.eastmoney.industry import fetch_industry_members
from cnequity.adapters.eastmoney.sectors import fetch_sector_members
from cnequity.adapters.sw.industry_history import (
    expand_sw_industry_as_of,
    fetch_sw_industry_intervals,
)
from cnequity.config import Config
from cnequity.orchestrator.registry import register_step
from cnequity.query.canonical import dedupe_lazy_by_primary_key
from cnequity.steps.common import BACKFILL_START, list_trading_dates
from cnequity.steps.http_common import run_incremental_fetched, write_fetched

logger = logging.getLogger(__name__)

_INDUSTRY_HISTORY_START = date(2020, 1, 1)
_MIN_CNI_MEMBERS_PER_INDEX = 50
_MIN_DAILY_INDEX_MEMBERS_PER_INDEX = 50
_MIN_DAILY_INDUSTRY_MEMBER_SYMBOLS = 1000
_MIN_DAILY_SECTOR_MEMBER_ROWS = 10_000
_MIN_DAILY_SECTOR_CODES = 50
_MIN_DAILY_INDUSTRY_CODES = 50


def _month_end_trading_days(config: Config, start: date, end: date) -> list[date]:
    """Last trading day of each calendar month in [start, end]."""
    days = list_trading_dates(config, start, end)
    if not days:
        return []
    by_month: dict[tuple[int, int], date] = {}
    for d in days:
        by_month[(d.year, d.month)] = d
    return [by_month[k] for k in sorted(by_month)]


def _existing_as_of_dates(
    config: Config,
    dataset: str,
    *,
    required_index_symbols: tuple[str, ...] | None = None,
    min_members_per_index: int | None = None,
) -> set[date]:
    root = config.curated_root / dataset
    files = list(root.glob("**/*.parquet")) if root.exists() else []
    if not files:
        return set()
    if required_index_symbols is not None:
        grouped = (
            pl.scan_parquet(files)
            .filter(pl.col("index_symbol").is_in(list(required_index_symbols)))
            .group_by("as_of_date", "index_symbol")
            .agg(pl.col("symbol").n_unique().alias("_member_count"))
            .group_by("as_of_date")
            .agg(
                pl.col("index_symbol").n_unique().alias("_index_count"),
                pl.col("_member_count").min().alias("_min_member_count"),
            )
            .filter(pl.col("_index_count") >= len(required_index_symbols))
        )
        if min_members_per_index is not None:
            grouped = grouped.filter(pl.col("_min_member_count") >= min_members_per_index)
        return set(grouped.select("as_of_date").collect().get_column("as_of_date").to_list())
    return set(
        pl.scan_parquet(files).select("as_of_date").unique().collect()["as_of_date"].to_list()
    )


def _validate_daily_membership_snapshot(
    frame: pl.DataFrame,
    dataset: str,
    *,
    min_count: int,
    count_columns: list[str],
    min_unique_counts: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Reject a non-empty but obviously truncated live membership snapshot.

    Both EastMoney membership endpoints are point-in-time snapshots.  A
    successful HTTP response is not evidence that pagination returned the full
    report, so letting any non-empty frame advance the snapshot watermark can
    permanently hide a partial universe.  The floors are deliberately applied
    after adapter de-duplication and count the dataset's natural membership
    key, not raw HTTP rows.
    """
    if frame.is_empty():
        return frame
    min_unique_counts = min_unique_counts or {}
    required_columns = {*count_columns, *min_unique_counts}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{dataset}: membership snapshot is missing required column(s): " + ", ".join(missing)
        )
    member_count = frame.unique(subset=count_columns).height
    thin_dimensions = {
        column: int(frame.get_column(column).drop_nulls().n_unique())
        for column in min_unique_counts
        if frame.get_column(column).drop_nulls().n_unique() < min_unique_counts[column]
    }
    if member_count < min_count or thin_dimensions:
        details = [f"{member_count} unique membership row(s)"]
        details.extend(
            f"{column}={count} (minimum {min_unique_counts[column]})"
            for column, count in sorted(thin_dimensions.items())
        )
        raise RuntimeError(
            f"{dataset}: incomplete daily snapshot; expected at least {min_count} "
            + ", ".join(details)
        )
    return frame


@register_step("sector_members", group="capital", depends_on=["instruments"])
def step_sector_members(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("sector_members: eastmoney source disabled in config")

    def _fetch(d: date) -> pl.DataFrame:
        return _validate_daily_membership_snapshot(
            fetch_sector_members(d, config=config),
            "sector_members",
            min_count=_MIN_DAILY_SECTOR_MEMBER_ROWS,
            count_columns=["symbol", "sector_code", "as_of_date"],
            min_unique_counts={"sector_code": _MIN_DAILY_SECTOR_CODES},
        )

    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "sector_members",
        _fetch,
        source="eastmoney",
        date_col="as_of_date",
    )


@register_step("index_constituents", group="fundamentals", depends_on=["instruments"])
def step_index_constituents(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if getattr(config, "_backfill", False):
        return _backfill_index_constituents(config, trade_date, run_id)
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("index_constituents: eastmoney source disabled in config")

    def _fetch(d: date) -> pl.DataFrame:
        frame = fetch_index_constituents(d, config=config)
        if frame.is_empty():
            return frame
        counts = (
            frame.group_by("index_symbol")
            .agg(pl.col("symbol").n_unique().alias("_member_count"))
            .filter(pl.col("_member_count") < _MIN_DAILY_INDEX_MEMBERS_PER_INDEX)
        )
        if not counts.is_empty():
            # EastMoney's RPT_INDEX_CONSTITUENT returns only *changes* at each
            # rebalance, not the full constituent list.  A low member count is
            # therefore expected, not a data-quality failure.  Log a warning
            # and proceed rather than failing the step.
            details = ", ".join(
                f"{row['index_symbol']}={row['_member_count']}"
                for row in counts.iter_rows(named=True)
            )
            logger.warning(
                "index_constituents: partial snapshot for %s "
                "(source returns rebalance changes, not full list); "
                "expected ≥%d members per index. Details: %s",
                d.isoformat(),
                _MIN_DAILY_INDEX_MEMBERS_PER_INDEX,
                details,
            )
        return frame

    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "index_constituents",
        _fetch,
        source="eastmoney",
        date_col="as_of_date",
    )


def _backfill_index_constituents(config: Config, trade_date: date, run_id: str) -> dict:
    """CNI adjustment history → as_of snapshots for 399001/399006 (C2)."""
    start = getattr(config, "_backfill_start", None) or date(2021, 12, 1)
    end = getattr(config, "_backfill_end", None) or trade_date
    have = _existing_as_of_dates(
        config,
        "index_constituents",
        required_index_symbols=CNI_BACKFILL_INDICES,
        min_members_per_index=_MIN_CNI_MEMBERS_PER_INDEX,
    )
    # Prefer rebalance-month ends so as_of aligns with CNI spell boundaries.
    todo = [
        d for d in _month_end_trading_days(config, start, min(end, trade_date)) if d not in have
    ]
    if not todo:
        return {"rows_read": 0, "rows_written": 0, "note": "all CNI as_of months already present"}

    frames: list[pl.DataFrame] = []
    failed_indices: list[str] = []
    for index_symbol in CNI_BACKFILL_INDICES:
        adj = fetch_cni_index_adjustments(index_symbol)
        if adj.is_empty():
            failed_indices.append(index_symbol)
            continue
        frames.append(expand_cni_constituents_as_of(adj, todo))

    if not frames:
        raise RuntimeError(
            "index_constituents backfill: no CNI adjustment rows for "
            + ", ".join(CNI_BACKFILL_INDICES)
        )
    df = pl.concat([f for f in frames if not f.is_empty()])
    if df.is_empty():
        raise RuntimeError("index_constituents backfill: expansion produced 0 rows")

    # An adjustment workbook can parse successfully yet cover only a thin
    # slice of an index.  Writing that slice as a complete as-of snapshot is
    # worse than an explicit retry: ``_existing_as_of_dates`` would otherwise
    # see the date on disk and a later backfill could treat the partial index
    # as historical coverage.  Remove only incomplete index/date pairs so a
    # healthy index can still make progress while the manifest records the
    # affected scope for retry.
    unique = df.unique(subset=["index_symbol", "symbol", "as_of_date"])
    counts = unique.group_by("as_of_date", "index_symbol").len().rename({"len": "_member_count"})
    thin = counts.filter(pl.col("_member_count") < _MIN_CNI_MEMBERS_PER_INDEX)
    if not thin.is_empty():
        df = df.join(
            thin.select("as_of_date", "index_symbol"),
            on=["as_of_date", "index_symbol"],
            how="anti",
        )
    thin_details = [
        f"{row['index_symbol']}@{row['as_of_date']}={row['_member_count']}"
        for row in thin.sort(["as_of_date", "index_symbol"]).iter_rows(named=True)
    ]
    if df.is_empty():
        raise RuntimeError(
            "index_constituents backfill: all CNI as-of snapshots were below the "
            f"minimum {_MIN_CNI_MEMBERS_PER_INDEX} unique members"
        )

    result = write_fetched(config, run_id, "index_constituents", df, source="cni")
    if failed_indices or thin_details:
        result["status"] = "warning"
        result.setdefault("context_updates", {})["audit_findings"] = [
            {
                "dataset": "index_constituents",
                "severity": "warning",
                "code": "cni_index_backfill_incomplete",
                "message": (
                    (
                        "CNI returned empty adjustment history for: "
                        + ", ".join(failed_indices)
                        + ". "
                        if failed_indices
                        else ""
                    )
                    + (
                        "CNI returned thin as-of snapshots: "
                        + ", ".join(thin_details[:8])
                        + (" (more omitted)" if len(thin_details) > 8 else "")
                        + ". "
                        if thin_details
                        else ""
                    )
                    + " (CSI 000300/000905 still EM-daily only)"
                ),
                "failed_indices": failed_indices,
                "thin_snapshots": thin_details,
            }
        ]
    result["as_of_dates"] = len(todo)
    return result


@register_step("industry_members", group="fundamentals", depends_on=["instruments"])
def step_industry_members(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if getattr(config, "_backfill", False):
        return _backfill_industry_members(config, trade_date, run_id)
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("industry_members: eastmoney source disabled in config")

    def _fetch(d: date) -> pl.DataFrame:
        return _validate_daily_membership_snapshot(
            fetch_industry_members(d, config=config),
            "industry_members",
            min_count=_MIN_DAILY_INDUSTRY_MEMBER_SYMBOLS,
            count_columns=["symbol", "classification_system", "as_of_date"],
            min_unique_counts={"industry_code": _MIN_DAILY_INDUSTRY_CODES},
        )

    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "industry_members",
        _fetch,
        source="eastmoney",
        date_col="as_of_date",
    )


def _backfill_industry_members(config: Config, trade_date: date, run_id: str) -> dict:
    """Shenwan classification intervals → monthly as_of snapshots from 2020 (C2)."""
    start = getattr(config, "_backfill_start", None) or max(BACKFILL_START, _INDUSTRY_HISTORY_START)
    end = getattr(config, "_backfill_end", None) or trade_date
    have = _existing_as_of_dates(config, "industry_members")
    # Skip eastmoney daily snapshots already in lake when choosing SW months —
    # SW rows use classification_system=sw and share as_of_date partitions, so
    # only skip dates that already contain sw rows.
    # A thin snapshot is not safe to expose as a complete as-of universe. Drop
    # it before staging so a query between retries cannot silently consume a
    # partial month; the next backfill will retry it because it is absent from
    # ``_existing_sw_as_of_dates(min_rows=1000)``.
    sw_have = _existing_sw_as_of_dates(config, min_rows=1000)
    todo = [
        d for d in _month_end_trading_days(config, start, min(end, trade_date)) if d not in sw_have
    ]
    if not todo:
        return {
            "rows_read": 0,
            "rows_written": 0,
            "note": "all Shenwan monthly as_of dates already present",
            "eastmoney_as_of_dates": len(have),
        }

    intervals = fetch_sw_industry_intervals()
    df = expand_sw_industry_as_of(intervals, todo)
    if df.is_empty():
        raise RuntimeError("industry_members backfill: Shenwan expansion produced 0 rows")
    # A month with far fewer names than typical means the XLS window does not
    # reach that as_of. Treat both that case and a completely missing requested
    # month as incomplete rather than letting a non-empty response look like a
    # successful full history sweep.
    unique = df.unique(subset=["symbol", "classification_system", "as_of_date"], keep="last")
    counts = unique.group_by("as_of_date").len().sort("as_of_date")
    thin = counts.filter(pl.col("len") < 1000)
    observed_dates = set(counts["as_of_date"].to_list())
    missing_dates = sorted(set(todo) - observed_dates)
    thin_dates = set(thin["as_of_date"].to_list())
    incomplete_dates = set(missing_dates) | thin_dates
    if incomplete_dates:
        df = df.filter(~pl.col("as_of_date").is_in(sorted(incomplete_dates)))
    if df.is_empty():
        raise RuntimeError(
            "industry_members backfill: all requested Shenwan as-of snapshots were "
            f"missing or below the minimum {_MIN_DAILY_INDUSTRY_MEMBER_SYMBOLS} "
            "unique members"
        )
    result = write_fetched(config, run_id, "industry_members", df, source="sw")
    result["as_of_dates"] = df["as_of_date"].n_unique()
    if missing_dates or thin.height:
        result["status"] = "warning"
        result.setdefault("context_updates", {})["audit_findings"] = [
            {
                "dataset": "industry_members",
                "severity": "warning",
                "code": "sw_industry_thin_months",
                "message": (
                    (
                        f"{len(missing_dates)} requested month(s) returned no Shenwan members "
                        f"(sample {missing_dates[:3]})"
                        if missing_dates
                        else ""
                    )
                    + (
                        ("; " if missing_dates else "")
                        + f"{thin.height} month(s) have <{_MIN_DAILY_INDUSTRY_MEMBER_SYMBOLS} "
                        f"Shenwan members (sample {thin['as_of_date'].head(3).to_list()})"
                        if thin.height
                        else ""
                    )
                ),
                "missing_as_of_dates": [d.isoformat() for d in missing_dates],
                "thin_as_of_dates": [d.isoformat() for d in sorted(thin_dates)],
            }
        ]
    return result


def _existing_sw_as_of_dates(config: Config, *, min_rows: int | None = None) -> set[date]:
    root = config.curated_root / "industry_members"
    files = list(root.glob("**/*.parquet")) if root.exists() else []
    if not files:
        return set()
    scan = dedupe_lazy_by_primary_key(
        pl.scan_parquet(files).filter(pl.col("classification_system") == "sw"),
        "industry_members",
    )
    if min_rows is not None:
        return set(
            scan.group_by("as_of_date")
            .len()
            .filter(pl.col("len") >= min_rows)
            .select("as_of_date")
            .collect()
            .get_column("as_of_date")
            .to_list()
        )
    return set(scan.select("as_of_date").unique().collect().get_column("as_of_date").to_list())
