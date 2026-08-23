"""Machine-readable validity contract for historical all-A universes."""

from __future__ import annotations

from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.quality.st_coverage import st_evidence_coverage_report
from cnequity.query.parquet_scan import dataset_has_parquet, scan_parquet_root
from cnequity.query.universe import coverage_end_date, coverage_start_date, st_coverage_start
from cnequity.steps.delisted import delisted_coverage_report


def _daily_bar_missing_sessions(
    config: Config,
    start: date,
    end: date,
) -> list[date]:
    """Return whole-market trading sessions absent from daily_bars.

    Individual symbols can legitimately be suspended, so this is deliberately
    a dataset-level check: a session is missing only when no daily bar landed
    for any symbol.  The boundary check above cannot see this interior hole,
    yet a backtest spanning it would silently bridge the missing session.
    """
    if start > end or not dataset_has_parquet(config.curated_root / "daily_bars"):
        return []

    from cnequity.steps.common import list_trading_dates

    expected = list_trading_dates(config, start, end)
    if not expected:
        return []
    bars = scan_parquet_root(
        config.curated_root / "daily_bars",
        partition_col="trade_date",
        start=start,
        end=end,
        traded_only=True,
    )
    actual = bars.select("trade_date").unique().collect()
    present = set(actual["trade_date"].drop_nulls().to_list())
    return [session for session in expected if session not in present]


def _bars_end(config: Config) -> date | None:
    return coverage_end_date(config, "daily_bars")


def historical_universe_validity(
    config: Config,
    start: date | None = None,
    end: date | None = None,
    *,
    sample: int = 15,
) -> dict:
    """Return a strict, read-only all-A universe validity manifest.

    The contract covers the price-window boundary, historical ST filtering and
    catalogued delistings. Adjustment-factor exactness, PIT fundamentals and
    strategy-specific feature coverage remain downstream responsibilities.
    """
    bars_read_error: str | None = None
    try:
        observed_start = coverage_start_date(config, "daily_bars")
        observed_end = _bars_end(config)
    except (OSError, pl.exceptions.PolarsError, ValueError) as exc:
        # The structural lake audit reports the exact bad file. Keep this
        # machine-readable research gate usable as a standalone API too:
        # unreadable bars mean the historical claim is not ready, not that the
        # health command itself should crash.
        observed_start = observed_end = None
        bars_read_error = str(exc)
    requested_start = start or observed_start
    requested_end = end or observed_end

    blockers: list[dict] = []
    if bars_read_error:
        blockers.append(
            {
                "check": "daily_bars_readable",
                "code": "daily_bars_unreadable",
                "message": f"daily_bars could not be read: {bars_read_error}",
                "remediation": (
                    "Repair or remove the unreadable Parquet file reported by the lake "
                    "quality audit, then rerun this validity check."
                ),
            }
        )
    window_valid = (
        requested_start is not None
        and requested_end is not None
        and requested_start <= requested_end
        and observed_start is not None
        and observed_end is not None
        and observed_start <= requested_start
        and observed_end >= requested_end
    )
    if requested_start is None or requested_end is None:
        blockers.append(
            {
                "check": "daily_bars_window",
                "code": "daily_bars_window_unknown",
                "message": "daily_bars has no observable research window",
                "remediation": "Backfill and compact daily_bars before validating research history.",
            }
        )
    elif requested_start > requested_end:
        blockers.append(
            {
                "check": "daily_bars_window",
                "code": "invalid_requested_window",
                "message": "requested start is after requested end",
                "remediation": "Choose an inclusive window with start on or before end.",
            }
        )
    elif not window_valid:
        blockers.append(
            {
                "check": "daily_bars_window",
                "code": "daily_bars_window_incomplete",
                "message": (
                    f"requested {requested_start.isoformat()}..{requested_end.isoformat()} "
                    f"is not contained by daily_bars "
                    f"{observed_start.isoformat() if observed_start else 'unknown'}.."
                    f"{observed_end.isoformat() if observed_end else 'unknown'}"
                ),
                "remediation": "Backfill and compact daily_bars for the full requested window.",
            }
        )

    observed_positive_st_start = st_coverage_start(config)
    st_evidence = st_evidence_coverage_report(config, requested_start, requested_end)
    st_valid = bool(st_evidence["verified"])
    if not st_valid:
        blockers.append(
            {
                "check": "historical_st_labels",
                "code": "historical_st_labels_incomplete",
                "message": "historical ST evidence has no complete, current scope receipt "
                f"for the requested window ({st_evidence['reason']})",
                "remediation": (
                    "Run a full `cne backfill trading_status` for this window and current "
                    "all-A symbol scope; resolve every failed symbol."
                ),
            }
        )

    daily_bar_missing: list[date] = []
    if (
        not bars_read_error
        and window_valid
        and requested_start is not None
        and requested_end is not None
    ):
        daily_bar_missing = _daily_bar_missing_sessions(config, requested_start, requested_end)
        if daily_bar_missing:
            blockers.append(
                {
                    "check": "daily_bars_interior_coverage",
                    "code": "daily_bars_interior_gap",
                    "message": (
                        f"daily_bars has no rows for {len(daily_bar_missing)} trading "
                        f"session(s) inside the requested window (e.g. "
                        f"{daily_bar_missing[0].isoformat()})"
                    ),
                    "missing_sessions": len(daily_bar_missing),
                    "sample_sessions": [d.isoformat() for d in daily_bar_missing[:15]],
                    "remediation": (
                        "Backfill and compact daily_bars for the missing sessions before "
                        "using this research window."
                    ),
                }
            )

    survivorship: dict | None = None
    survivorship_valid = False
    if (
        not bars_read_error
        and requested_start is not None
        and requested_end is not None
        and requested_start <= requested_end
    ):
        try:
            survivorship = delisted_coverage_report(
                config, requested_start, requested_end, sample=sample
            )
        except (OSError, pl.exceptions.PolarsError, ValueError) as exc:
            survivorship = {
                "verified": False,
                "counts": {
                    "pending_probe": 0,
                    "missing_bars": 0,
                    "unknown_overlap": 0,
                    "terminal_mismatch": 0,
                    "missing_instrument": 0,
                    "invalid_delist_date": 0,
                },
                "error": str(exc),
            }
            blockers.append(
                {
                    "check": "delisted_universe",
                    "code": "delisted_universe_unreadable",
                    "message": f"delisted coverage could not be read: {exc}",
                    "remediation": (
                        "Repair the unreadable curated Parquet file, then rerun delisted "
                        "coverage and this validity check."
                    ),
                }
            )
        survivorship_valid = bool(survivorship["verified"])
        if not survivorship_valid and "error" not in survivorship:
            counts = survivorship["counts"]
            blockers.append(
                {
                    "check": "delisted_universe",
                    "code": "delisted_universe_unverified",
                    "message": (
                        "delisted coverage is unverified: "
                        f"{counts.get('pending_probe', 0)} pending probes, "
                        f"{counts.get('recent_quarantined', 0)} recent names without current evidence, "
                        f"{counts.get('formal_unresolved', 0)} formal names without overlap evidence, "
                        f"{counts.get('missing_bars', 0)} missing bars, "
                        f"{counts.get('unknown_overlap', 0)} unknown overlaps, "
                        f"{counts.get('terminal_mismatch', 0)} terminal mismatches, "
                        f"{counts.get('missing_instrument', 0)} missing instruments, "
                        f"{counts.get('invalid_delist_date', 0)} invalid delist dates"
                    ),
                    "remediation": (
                        "Run `cne delisted coverage` for samples, then complete discovery and "
                        "repair the reported catalogue, bars, or instruments gaps."
                    ),
                }
            )

    universe_ready = window_valid and not daily_bar_missing and st_valid and survivorship_valid
    return {
        "schema_version": 1,
        "claim": "historical_all_a_universe_validity",
        "window": {
            "start": requested_start.isoformat() if requested_start else None,
            "end": requested_end.isoformat() if requested_end else None,
        },
        "universe_ready": universe_ready,
        "checks": {
            "daily_bars_window": {
                "passed": window_valid,
                "observed_start": observed_start.isoformat() if observed_start else None,
                "observed_end": observed_end.isoformat() if observed_end else None,
            },
            "daily_bars_interior_coverage": {
                "passed": not daily_bar_missing,
                "missing_sessions": len(daily_bar_missing),
                "sample_sessions": [d.isoformat() for d in daily_bar_missing[:15]],
            },
            "historical_st_labels": {
                "passed": st_valid,
                "coverage_start": st_evidence.get("coverage_start"),
                "coverage_end": st_evidence.get("coverage_end"),
                "observed_positive_st_start": (
                    observed_positive_st_start.isoformat() if observed_positive_st_start else None
                ),
                "evidence": st_evidence,
            },
            "delisted_universe": {
                "passed": survivorship_valid,
                "report": survivorship,
            },
        },
        "blockers": blockers,
        "limitations": [
            "Does not verify adjustment-factor exactness or strategy feature coverage.",
            "Does not verify point-in-time semantics of fundamentals used by a strategy.",
        ],
    }
