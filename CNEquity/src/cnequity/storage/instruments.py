"""Merge-style compact for instruments (preserve delisted symbols)."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import polars as pl

from cnequity.domain.schemas import INSTRUMENTS_SCHEMA, validate_dataframe
from cnequity.storage.atomic import write_parquet_atomic
from cnequity.storage.parquet import StagingWriter

# Refuse delist inference when too many symbols vanish from a snapshot — usually
# a partial TDX fetch, not a mass delisting event.
ABSENT_DELIST_THRESHOLD = 0.05
# A live-universe snapshot is a point-in-time observation. Require repeated
# absences before assigning a sticky delist_date; one partial fetch must not
# permanently remove a still-trading name from all_a.
ABSENT_DELIST_CONFIRMATIONS = 2


def _absence_state_path(curated_root: Path) -> Path:
    return curated_root.parent / "meta" / "instruments_absence_streak.json"


def _load_absence_state(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(symbol): value
        for symbol, value in payload.items()
        if isinstance(value, dict) and isinstance(value.get("count"), int)
    }


def _save_absence_state(path: Path, state: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def compact_instruments(
    staging_root: Path,
    curated_root: Path,
    run_id: str,
    trade_date: date,
) -> tuple[int, list[dict]]:
    """Merge staging instruments into curated, retaining symbols missing from TDX."""
    staging = StagingWriter(staging_root)
    files = staging.list_run_files("instruments", run_id)
    if not files:
        return 0, []

    incoming = pl.concat(
        [validate_dataframe(pl.read_parquet(f), "instruments") for f in files],
        how="diagonal_relaxed",
    )
    incoming = incoming.sort("fetched_at").unique(subset=["symbol"], keep="last")

    out_path = curated_root / "instruments" / "part-merged.parquet"
    curated_files = sorted(out_path.parent.rglob("*.parquet")) if out_path.parent.exists() else []
    if curated_files:
        existing = pl.concat(
            [validate_dataframe(pl.read_parquet(path), "instruments") for path in curated_files],
            how="diagonal_relaxed",
        )
        existing = existing.sort("fetched_at").unique(subset=["symbol"], keep="last")
    else:
        existing = pl.DataFrame(schema=INSTRUMENTS_SCHEMA)

    incoming_symbols = incoming["symbol"].to_list()
    findings: list[dict] = []
    absence_path = _absence_state_path(curated_root)
    absence_state = _load_absence_state(absence_path)
    if not existing.is_empty():
        preserved = existing.filter(~pl.col("symbol").is_in(incoming_symbols))
        # Symbols with a known delist date before this snapshot are expected
        # to be absent.  Counting them in the circuit breaker made a mature
        # catalog look like a partial fetch on every run, eventually masking
        # genuine omissions from the still-live universe.
        expected_live = existing.filter(
            pl.col("delist_date").is_null() | (pl.col("delist_date") >= trade_date)
        )
        absent_live = expected_live.filter(~pl.col("symbol").is_in(incoming_symbols))
        absent_count = absent_live.height
        expected_live_count = expected_live.height
        absent_ratio = absent_count / expected_live_count if expected_live_count else 0.0
        if absent_count and absent_ratio > ABSENT_DELIST_THRESHOLD:
            findings.append(
                {
                    "dataset": "instruments",
                    "severity": "error",
                    "check": "instruments_delist_suppressed",
                    "message": (
                        f"Refused to infer delist_date: {absent_count}/{expected_live_count} symbols "
                        f"({absent_ratio:.1%}) absent from snapshot (>{ABSENT_DELIST_THRESHOLD:.0%} "
                        "threshold); likely partial fetch"
                    ),
                    "absent_count": absent_count,
                    "existing_count": expected_live_count,
                    "absent_ratio": absent_ratio,
                }
            )
        else:
            inferred: dict[str, date] = {}
            pending = 0
            for row in absent_live.select("symbol", "delist_date").iter_rows(named=True):
                symbol = row["symbol"]
                if row["delist_date"] is not None:
                    absence_state.pop(symbol, None)
                    continue
                prior = absence_state.get(symbol, {})
                count = int(prior.get("count", 0)) + 1
                absence_state[symbol] = {
                    "count": count,
                    "last_missing": trade_date.isoformat(),
                }
                if count >= ABSENT_DELIST_CONFIRMATIONS:
                    inferred[symbol] = trade_date
                    absence_state.pop(symbol, None)
                else:
                    pending += 1
            if inferred:
                delist_expr = pl.col("delist_date")
                for symbol, inferred_date in inferred.items():
                    delist_expr = (
                        pl.when(pl.col("symbol") == symbol)
                        .then(pl.lit(inferred_date))
                        .otherwise(delist_expr)
                    )
                preserved = preserved.with_columns(delist_expr.alias("delist_date"))
            if pending:
                findings.append(
                    {
                        "dataset": "instruments",
                        "severity": "warning",
                        "check": "instruments_delist_pending",
                        "message": (
                            f"{pending}/{absent_count} absent symbol(s) need another "
                            f"consecutive snapshot before delist inference "
                            f"({ABSENT_DELIST_CONFIRMATIONS} confirmations required)"
                        ),
                        "pending_count": pending,
                        "absent_count": absent_count,
                        "confirmations_required": ABSENT_DELIST_CONFIRMATIONS,
                    }
                )
        prior_dates = existing.select(
            [
                "symbol",
                pl.col("list_date").alias("_prior_list_date"),
                pl.col("delist_date").alias("_prior_delist_date"),
            ]
        )
        incoming = incoming.join(prior_dates, on="symbol", how="left")
        # Both dates are sticky: a live snapshot carries neither (TDX has no such
        # field), so coalescing is what keeps a delist_date — inferred from an
        # earlier absence or fetched from baostock — from being erased the next
        # day. Never resurrect a name a prior run buried.
        incoming = incoming.with_columns(
            pl.coalesce(pl.col("list_date"), pl.col("_prior_list_date")).alias("list_date"),
            pl.coalesce(pl.col("delist_date"), pl.col("_prior_delist_date")).alias("delist_date"),
        ).drop("_prior_list_date", "_prior_delist_date")
        for symbol in incoming_symbols:
            absence_state.pop(symbol, None)
    else:
        preserved = pl.DataFrame(schema=INSTRUMENTS_SCHEMA)

    merged = pl.concat([incoming, preserved], how="diagonal_relaxed")
    merged = merged.sort("fetched_at").unique(subset=["symbol"], keep="last")

    write_parquet_atomic(out_path, merged, compression="zstd")
    # Instruments is merge-style, so there is no partition writer to clean up
    # stale fragments. Keep one canonical file; readers and whole-lake audits
    # must not see an old ``part-*.parquet`` beside it.
    for stale in out_path.parent.rglob("*.parquet"):
        if stale != out_path:
            stale.unlink()
    _save_absence_state(absence_path, absence_state)
    return merged.height, findings
