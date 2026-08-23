"""Integrity checks for datasets written by local derivation steps."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from cnequity.config import Config
from cnequity.derive.market_breadth import MARKET_BREADTH_METRICS
from cnequity.query.parquet_scan import collect_parquet_root, dataset_has_parquet

_LOOKBACK_DAYS = 30
_SAMPLE_DATES = 8
_METRIC_SET = frozenset(MARKET_BREADTH_METRICS)
_INDUSTRY_WEIGHTINGS = frozenset({"equal", "amount"})


def _date_sample(values: list[date]) -> list[str]:
    return [value.isoformat() for value in sorted(values)[:_SAMPLE_DATES]]


def _industry_group_coverage_findings(config: Config, frame: pl.DataFrame) -> list[dict]:
    """Check that every PIT membership group has an index observation.

    Pairing both weighting rows is not enough: a truncated derive can drop an
    entire industry while leaving all remaining groups structurally valid.
    Compare each index day with the latest 申万 snapshot known on that day so
    membership changes do not look like missing data.
    """
    from cnequity.derive.industry_index import LEVELS, _membership

    members = _membership(config)
    if members.is_empty() or "trade_date" not in frame.columns:
        return []
    members = members.select("industry_code", "as_of_date").drop_nulls().unique()
    snapshots = members.select("as_of_date").unique().sort("as_of_date")
    days = frame.select("trade_date").drop_nulls().unique().sort("trade_date")
    mapping = days.join_asof(
        snapshots,
        left_on="trade_date",
        right_on="as_of_date",
        strategy="backward",
    ).drop_nulls("as_of_date")
    if mapping.is_empty():
        return []

    snapshot_members = mapping.join(members, on="as_of_date", how="inner")
    expected_parts = []
    for level, width in LEVELS.items():
        expected_parts.append(
            snapshot_members.with_columns(
                pl.col("industry_code").str.slice(0, width).alias("industry_code")
            ).select("trade_date", pl.lit(level).alias("level"), "industry_code")
        )
    expected = pl.concat(expected_parts).unique()
    actual = frame.select("trade_date", "level", "industry_code").drop_nulls().unique()
    missing = expected.join(
        actual,
        on=["trade_date", "level", "industry_code"],
        how="anti",
    )
    if missing.is_empty():
        return []

    dates = missing["trade_date"].unique().to_list()
    sample = missing.sort("trade_date", "level", "industry_code").head(_SAMPLE_DATES)
    return [
        {
            "dataset": "industry_index",
            "severity": "error",
            "check": "industry_index_missing_groups",
            "message": (
                f"{missing.height} PIT membership group(s) have no industry-index row; "
                f"sample dates: {', '.join(_date_sample(dates))}"
            ),
            "missing_groups": missing.height,
            "date_sample": _date_sample(dates),
            "group_sample": sample.to_dicts(),
        }
    ]


def _finding(
    *,
    severity: str,
    check: str,
    message: str,
    **extra: object,
) -> dict:
    return {
        "dataset": "market_breadth",
        "severity": severity,
        "check": check,
        "message": message,
        **extra,
    }


def market_breadth_findings(
    config: Config,
    trade_date: date,
    *,
    full: bool = False,
    lookback_days: int = _LOOKBACK_DAYS,
) -> list[dict]:
    """Verify that each stored breadth session is a complete valid metric set.

    A non-empty year partition is enough for the generic dense-coverage check,
    but it is not enough for this dataset: seven rows make one observation and
    a partial write can otherwise look like a covered trading day. Normal run
    audits inspect a recent window; explicit lake health scans all history.
    """
    root = config.curated_root / "market_breadth"
    if not dataset_has_parquet(root):
        return []

    start = None if full else trade_date - timedelta(days=lookback_days)
    try:
        frame = collect_parquet_root(
            root,
            partition_col="trade_date",
            start=start,
            end=trade_date,
        )
    except (FileNotFoundError, OSError, pl.exceptions.PolarsError, ValueError) as exc:
        files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.parquet"))
        return [
            _finding(
                severity="error",
                check="derived_unreadable",
                message=(
                    "market_breadth cannot be read for integrity checks; repair the "
                    f"derived Parquet files before using its coverage: {exc}"
                ),
                files=files[:_SAMPLE_DATES],
            )
        ]

    if frame.is_empty():
        return []
    required = {"trade_date", "metric_id", "value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return [
            _finding(
                severity="error",
                check="derived_schema_contract",
                message=f"market_breadth is missing required column(s): {', '.join(missing)}",
                missing_columns=missing,
            )
        ]

    findings: list[dict] = []
    known = frame.filter(pl.col("metric_id").is_in(list(_METRIC_SET)))
    unknown = sorted(set(frame["metric_id"].drop_nulls().to_list()) - _METRIC_SET)
    if unknown:
        findings.append(
            _finding(
                severity="error",
                check="market_breadth_unknown_metric",
                message=(
                    f"market_breadth contains {len(unknown)} unknown metric id(s): "
                    f"{', '.join(str(value) for value in unknown[:_SAMPLE_DATES])}"
                ),
                unknown_metrics=[str(value) for value in unknown[:_SAMPLE_DATES]],
            )
        )

    non_null = known.filter(pl.col("value").is_not_null())
    observed = (
        frame.group_by("trade_date")
        .agg(
            pl.col("metric_id")
            .filter(pl.col("metric_id").is_in(list(_METRIC_SET)) & pl.col("value").is_not_null())
            .n_unique()
            .alias("metric_count")
        )
        .sort("trade_date")
    )
    incomplete = observed.filter(pl.col("metric_count") != len(MARKET_BREADTH_METRICS))
    if not incomplete.is_empty():
        incomplete_dates = [row["trade_date"] for row in incomplete.iter_rows(named=True)]
        seen = non_null.group_by("trade_date").agg(pl.col("metric_id").unique().alias("metrics"))
        seen_by_date = {row["trade_date"]: row["metrics"] for row in seen.iter_rows(named=True)}
        missing_by_date = {
            day.isoformat(): sorted(_METRIC_SET - set(seen_by_date.get(day, [])))
            for day in incomplete_dates
        }
        findings.append(
            _finding(
                severity="error",
                check="market_breadth_incomplete_day",
                message=(
                    f"{len(incomplete_dates)} session(s) have fewer than the required "
                    f"{len(MARKET_BREADTH_METRICS)} market-breadth metrics; sample: "
                    f"{', '.join(_date_sample(incomplete_dates))}"
                ),
                incomplete_days=len(incomplete_dates),
                incomplete_date_sample=_date_sample(incomplete_dates),
                missing_metrics_sample={
                    key: value for key, value in list(missing_by_date.items())[:_SAMPLE_DATES]
                },
            )
        )

    duplicate_keys = (
        known.group_by("trade_date", "metric_id")
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )
    if not duplicate_keys.is_empty():
        findings.append(
            _finding(
                severity="error",
                check="market_breadth_duplicate_metric",
                message=(
                    f"{duplicate_keys.height} trade_date/metric_id key(s) are duplicated; "
                    "deduplicate before consuming breadth signals"
                ),
                duplicate_keys=duplicate_keys.height,
            )
        )

    # Aggregate the first value per metric so a duplicate row cannot hide a
    # malformed observation. Duplicate keys are reported separately above.
    wide = non_null.group_by("trade_date").agg(
        [
            pl.col("value").filter(pl.col("metric_id") == metric).first().alias(metric)
            for metric in MARKET_BREADTH_METRICS
        ]
    )
    complete = wide.filter(
        pl.all_horizontal(pl.col(metric).is_not_null() for metric in MARKET_BREADTH_METRICS)
    )
    if not complete.is_empty():
        invalid = complete.filter(
            ~pl.all_horizontal(pl.col(metric).is_finite() for metric in MARKET_BREADTH_METRICS)
            | (pl.col("total_count") <= 0)
            | (pl.col("advance_count") < 0)
            | (pl.col("decline_count") < 0)
            | (pl.col("flat_count") < 0)
            | (pl.col("limit_up_count") < 0)
            | (pl.col("limit_down_count") < 0)
            | (pl.col("advance_ratio") < 0)
            | (pl.col("advance_ratio") > 1)
            | (
                pl.col("advance_count") + pl.col("decline_count") + pl.col("flat_count")
                != pl.col("total_count")
            )
            | (pl.col("limit_up_count") > pl.col("advance_count"))
            | (pl.col("limit_down_count") > pl.col("decline_count"))
            | (
                (pl.col("advance_ratio") - pl.col("advance_count") / pl.col("total_count")).abs()
                > 1e-6
            )
        )
        if not invalid.is_empty():
            bad_dates = invalid["trade_date"].to_list()
            findings.append(
                _finding(
                    severity="error",
                    check="market_breadth_inconsistent_metrics",
                    message=(
                        f"{invalid.height} market-breadth session(s) violate count/ratio "
                        f"invariants; sample: {', '.join(_date_sample(bad_dates))}"
                    ),
                    invalid_days=invalid.height,
                    invalid_date_sample=_date_sample(bad_dates),
                )
            )
    return findings


def industry_index_findings(
    config: Config,
    trade_date: date,
    *,
    full: bool = False,
    lookback_days: int = _LOOKBACK_DAYS,
) -> list[dict]:
    """Verify industry-index weighting pairs and row-level accounting fields."""
    root = config.derived_root / "industry_index"
    if not dataset_has_parquet(root):
        return []

    start = None if full else trade_date - timedelta(days=lookback_days)
    try:
        frame = collect_parquet_root(
            root,
            partition_col="trade_date",
            start=start,
            end=trade_date,
        )
    except (FileNotFoundError, OSError, pl.exceptions.PolarsError, ValueError) as exc:
        files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.parquet"))
        return [
            {
                "dataset": "industry_index",
                "severity": "error",
                "check": "derived_unreadable",
                "message": (
                    "industry_index cannot be read for integrity checks; repair the "
                    f"derived Parquet files before using its coverage: {exc}"
                ),
                "files": files[:_SAMPLE_DATES],
            }
        ]

    if frame.is_empty():
        return []
    required = {
        "trade_date",
        "industry_code",
        "level",
        "weighting",
        "ret",
        "n_members",
        "n_priced",
        "n_excluded",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return [
            {
                "dataset": "industry_index",
                "severity": "error",
                "check": "derived_schema_contract",
                "message": f"industry_index is missing required column(s): {', '.join(missing)}",
                "missing_columns": missing,
            }
        ]

    findings: list[dict] = []
    unknown_weightings = sorted(
        set(frame["weighting"].drop_nulls().to_list()) - _INDUSTRY_WEIGHTINGS
    )
    if unknown_weightings:
        findings.append(
            {
                "dataset": "industry_index",
                "severity": "error",
                "check": "industry_index_unknown_weighting",
                "message": (
                    f"industry_index contains unknown weighting value(s): "
                    f"{', '.join(str(value) for value in unknown_weightings[:_SAMPLE_DATES])}"
                ),
                "unknown_weightings": [str(value) for value in unknown_weightings[:_SAMPLE_DATES]],
            }
        )

    groups = (
        frame.group_by("trade_date", "industry_code", "level")
        .agg(pl.col("weighting").n_unique().alias("weighting_count"))
        .filter(pl.col("weighting_count") != len(_INDUSTRY_WEIGHTINGS))
    )
    if not groups.is_empty():
        dates = groups["trade_date"].unique().to_list()
        findings.append(
            {
                "dataset": "industry_index",
                "severity": "error",
                "check": "industry_index_incomplete_weightings",
                "message": (
                    f"{groups.height} date/industry/level group(s) do not contain both "
                    f"weightings {sorted(_INDUSTRY_WEIGHTINGS)}; sample dates: "
                    f"{', '.join(_date_sample(dates))}"
                ),
                "incomplete_groups": groups.height,
                "date_sample": _date_sample(dates),
            }
        )

    findings.extend(_industry_group_coverage_findings(config, frame))

    duplicate_keys = (
        frame.group_by("trade_date", "industry_code", "level", "weighting")
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )
    if not duplicate_keys.is_empty():
        findings.append(
            {
                "dataset": "industry_index",
                "severity": "error",
                "check": "industry_index_duplicate_key",
                "message": (
                    f"{duplicate_keys.height} industry-index primary key(s) are duplicated"
                ),
                "duplicate_keys": duplicate_keys.height,
            }
        )

    invalid = frame.filter(
        pl.col("trade_date").is_null()
        | pl.col("industry_code").is_null()
        | pl.col("level").is_null()
        | pl.col("weighting").is_null()
        | pl.col("n_members").is_null()
        | pl.col("n_priced").is_null()
        | pl.col("n_excluded").is_null()
        | (pl.col("n_members") < 0)
        | (pl.col("n_priced") < 0)
        | (pl.col("n_excluded") < 0)
        | (pl.col("n_priced") > pl.col("n_members"))
        | (pl.col("n_excluded") != pl.col("n_members") - pl.col("n_priced"))
        | (pl.col("ret").is_not_null() & ~pl.col("ret").is_finite())
    )
    if not invalid.is_empty():
        dates = invalid["trade_date"].drop_nulls().unique().to_list()
        findings.append(
            {
                "dataset": "industry_index",
                "severity": "error",
                "check": "industry_index_invalid_accounting",
                "message": (
                    f"{invalid.height} industry-index row(s) violate membership/pricing "
                    f"accounting or contain non-finite returns; sample dates: "
                    f"{', '.join(_date_sample(dates))}"
                ),
                "invalid_rows": invalid.height,
                "date_sample": _date_sample(dates),
            }
        )
    return findings
