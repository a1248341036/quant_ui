"""Versioned completion evidence for historical ST backfills.

Rows in ``trading_status`` answer a market question (ST, normal, suspended).
They cannot, by themselves, prove that every symbol in a requested scope was
checked.  This module keeps that operational proof separate from the facts:
an exact, versioned checkpoint while work is in progress and an immutable
coverage receipt once the whole scope is resolved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from cnequity.config import Config
from cnequity.domain.symbols import is_all_a_symbol, is_cdr_symbol, parse_symbol
from cnequity.query.canonical import dedupe_by_primary_key, dedupe_lazy_by_primary_key
from cnequity.query.parquet_scan import collect_parquet_root, scan_parquet_files

ST_EVIDENCE_VERSION = 2
ST_COVERAGE_CLAIM = "historical_st_evidence"

logger = logging.getLogger(__name__)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def symbol_scope_hash(symbols: list[str]) -> str:
    return _canonical_hash(sorted(set(symbols)))


def current_st_universe(config: Config) -> list[str]:
    """All-A instruments whose price history makes ST evidence relevant.

    This is deliberately disk-only.  A coverage check must never make a
    provider call merely to decide whether an existing receipt is trustworthy.
    """
    root = config.curated_root / "instruments"
    if not root.exists():
        return []
    try:
        frame = dedupe_by_primary_key(
            collect_parquet_root(root, hive=False),
            "instruments",
        )
    except (FileNotFoundError, OSError, pl.exceptions.PolarsError, ValueError) as exc:
        # A partial universe is unsafe here: it can make an old receipt appear
        # to cover every currently visible name. Keep the claim unverified
        # until the structural audit repairs the unreadable input.
        logger.warning("ST coverage: instruments are not readable: %s", exc)
        return []
    if "symbol" not in frame.columns:
        return []
    symbols: list[str] = []
    for raw in frame["symbol"].to_list():
        try:
            parsed = parse_symbol(str(raw))
        except ValueError:
            continue
        # ``is_all_a_symbol`` is intentionally a broad code-space predicate
        # used by source discovery and therefore still admits SH 689xxx CDRs.
        # The research ``all_a`` universe excludes CDRs; ST evidence must use
        # the same universe or it creates an unnecessary scope member that can
        # never contribute to stock-universe queries.
        if is_all_a_symbol(parsed.code, parsed.exchange) and not is_cdr_symbol(
            parsed.code, parsed.exchange
        ):
            symbols.append(str(raw))
    bars_root = config.curated_root / "daily_bars"
    bar_files = list(bars_root.rglob("*.parquet")) if bars_root.exists() else []
    if bar_files:
        bar_scans: list[pl.LazyFrame] = []
        for path in bar_files:
            try:
                bar_scan = scan_parquet_files([path])
                # Suspended symbols may have carried-forward OHLC rows with no
                # actual print. Keep old/minimal files without volume readable.
                if "volume" in bar_scan.collect_schema().names():
                    bar_scan = bar_scan.filter(pl.col("volume") > 0)
                bar_scans.append(bar_scan)
            except (FileNotFoundError, OSError, pl.exceptions.PolarsError, ValueError) as exc:
                logger.warning("ST coverage: daily_bars file is not readable: %s: %s", path, exc)
                return []
        bar_scan = pl.concat(bar_scans, how="diagonal_relaxed")
        try:
            bars = set(bar_scan.select("symbol").unique().collect()["symbol"].to_list())
        except (FileNotFoundError, OSError, pl.exceptions.PolarsError, ValueError) as exc:
            logger.warning("ST coverage: daily_bars scan is not readable: %s", exc)
            return []
        symbols = [symbol for symbol in symbols if symbol in bars]
    return sorted(set(symbols))


def build_st_scope(
    symbols: list[str],
    start: date,
    end: date,
    *,
    universe: str,
) -> dict[str, Any]:
    if start > end:
        raise ValueError(f"ST evidence window is inverted: {start} > {end}")
    resolved = sorted(set(symbols))
    identity = {
        "evidence_version": ST_EVIDENCE_VERSION,
        "source": "baostock",
        "universe": universe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols_sha256": symbol_scope_hash(resolved),
    }
    return {
        **identity,
        "scope_id": _canonical_hash(identity),
        "expected_symbols": resolved,
        "expected_symbols_count": len(resolved),
    }


def st_checkpoint_path(config: Config, scope_id: str) -> Path:
    return (
        config.meta_root
        / "state"
        / ST_COVERAGE_CLAIM
        / f"v{ST_EVIDENCE_VERSION}"
        / f"{scope_id}.json"
    )


def load_st_checkpoint(config: Config, scope: dict[str, Any]) -> dict[str, Any]:
    """Load only an exact v2 scope; legacy sparse-ST markers are invalid.

    Version 1 marked never-ST symbols complete without persisting their normal
    rows. Reusing it would claim evidence that does not exist, so migration is
    intentionally a clean resweep rather than a metadata rewrite.
    """
    path = st_checkpoint_path(config, str(scope["scope_id"]))
    if not path.exists():
        # The all-A universe can grow after a delisted/instrument recovery.
        # In that case the exact scope hash changes even though the requested
        # date window and evidence contract are unchanged. Reuse the most
        # advanced compatible checkpoint as a seed; reusable_st_checkpoint_symbols
        # will still verify that its positive facts survived compaction before
        # the caller skips any symbol.
        root = path.parent
        current_symbols = set(scope.get("expected_symbols", []))
        candidates: list[tuple[int, float, dict[str, Any]]] = []
        for candidate_path in root.glob("*.json"):
            if candidate_path == path:
                continue
            try:
                candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            previous_scope = candidate.get("scope") or {}
            previous_symbols = set(previous_scope.get("expected_symbols", []))
            if not previous_symbols or not previous_symbols <= current_symbols:
                continue
            if any(
                previous_scope.get(key) != scope.get(key)
                for key in ("evidence_version", "source", "start", "end")
            ):
                continue
            completed = set(candidate.get("completed_symbols", [])) & current_symbols
            candidates.append((len(completed), candidate_path.stat().st_mtime, candidate))
        if candidates:
            _, _, previous = max(candidates, key=lambda item: (item[0], item[1]))
            completed = sorted(set(previous.get("completed_symbols", [])) & current_symbols)
            evidence_rows = {
                symbol: int(count)
                for symbol, count in (previous.get("evidence_rows_by_symbol") or {}).items()
                if symbol in completed
            }
            unresolved = sorted(set(previous.get("unresolved_symbols", [])) & current_symbols)
            logger.info(
                "ST checkpoint scope grew from %d to %d symbols; inheriting %d completed symbols",
                len(previous.get("scope", {}).get("expected_symbols", [])),
                len(current_symbols),
                len(completed),
            )
            return {
                "schema_version": 1,
                "claim": ST_COVERAGE_CLAIM,
                "scope": scope,
                "status": "incomplete" if unresolved else "pending",
                "completed_symbols": completed,
                "evidence_rows_by_symbol": evidence_rows,
                "unresolved_symbols": unresolved,
                "inherited_from_scope_id": previous.get("scope", {}).get("scope_id"),
            }
        return {
            "schema_version": 1,
            "claim": ST_COVERAGE_CLAIM,
            "scope": scope,
            "status": "pending",
            "completed_symbols": [],
            "unresolved_symbols": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scope") != scope or payload.get("claim") != ST_COVERAGE_CLAIM:
        raise RuntimeError(f"ST checkpoint identity mismatch: {path}")
    return payload


def write_st_checkpoint(config: Config, payload: dict[str, Any]) -> Path:
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = st_checkpoint_path(config, str(payload["scope"]["scope_id"]))
    _atomic_json(path, payload)
    return path


def _st_row_counts(
    config: Config,
    scope: dict[str, Any],
    symbols: set[str],
    *,
    staging_run_id: str | None = None,
) -> dict[str, int]:
    """Count persisted Baostock facts in curated plus one resumable run."""
    if not symbols:
        return {}
    files = list((config.curated_root / "trading_status").rglob("*.parquet"))
    if staging_run_id:
        from cnequity.storage import StagingWriter

        files.extend(
            StagingWriter(config.staging_root).list_run_files("trading_status", staging_run_id)
        )
    if not files:
        return {}
    frame = scan_parquet_files(files, missing_columns="insert", extra_columns="ignore")
    schema = frame.collect_schema().names()
    frame = frame.filter(
        pl.col("symbol").is_in(sorted(symbols)),
        pl.col("trade_date").is_between(
            date.fromisoformat(scope["start"]),
            date.fromisoformat(scope["end"]),
        ),
    )
    if "source" in schema:
        frame = frame.filter(pl.col("source") == "baostock")
    # Count only distinct Baostock facts. Filter the evidence owner first: a
    # same-PK EastMoney snapshot must not erase the Baostock row from this
    # source-specific coverage claim before canonicalization.
    frame = dedupe_lazy_by_primary_key(frame, "trading_status")
    counts = frame.group_by("symbol").len().collect()
    return {row["symbol"]: int(row["len"]) for row in counts.iter_rows(named=True)}


def reusable_st_checkpoint_symbols(
    config: Config,
    checkpoint: dict[str, Any],
    run_id: str,
) -> set[str]:
    """Symbols whose checkpoint facts still exist after a crash/restart.

    Zero-row source responses are durable operational evidence on their own.
    Positive row counts must still be visible either in curated storage or in
    staging for the same run being resumed.
    """
    completed = set(checkpoint.get("completed_symbols", []))
    expected_counts = checkpoint.get("evidence_rows_by_symbol") or {}
    persisted = _st_row_counts(
        config,
        checkpoint["scope"],
        completed,
        staging_run_id=run_id,
    )
    return {
        symbol
        for symbol in completed
        if symbol in expected_counts
        and (
            int(expected_counts[symbol]) == 0
            or persisted.get(symbol, 0) >= int(expected_counts[symbol])
        )
    }


def _receipt_path(config: Config, scope_id: str) -> Path:
    return config.meta_root / "quality" / "coverage" / ST_COVERAGE_CLAIM / f"{scope_id}.json"


def publish_st_coverage_receipt(config: Config, checkpoint: dict[str, Any]) -> Path:
    scope = checkpoint["scope"]
    expected = set(scope["expected_symbols"])
    completed = set(checkpoint.get("completed_symbols", []))
    unresolved = set(checkpoint.get("unresolved_symbols", []))
    if completed != expected or unresolved:
        raise ValueError("cannot publish incomplete ST coverage evidence")
    expected_counts = checkpoint.get("evidence_rows_by_symbol") or {}
    if set(expected_counts) != expected:
        raise ValueError("cannot publish ST coverage without per-symbol row evidence")
    persisted = _st_row_counts(config, scope, expected)
    missing = [
        symbol
        for symbol in expected
        if int(expected_counts[symbol]) > 0
        and persisted.get(symbol, 0) < int(expected_counts[symbol])
    ]
    if missing:
        raise ValueError(
            f"cannot publish ST coverage before {len(missing)} symbol(s) reach curated storage"
        )
    receipt = {
        "schema_version": 1,
        "claim": ST_COVERAGE_CLAIM,
        "status": "complete",
        "scope": {key: value for key, value in scope.items() if key != "expected_symbols"},
        "completed_symbols": sorted(completed),
        "completed_symbols_count": len(completed),
        "completed_symbols_sha256": symbol_scope_hash(sorted(completed)),
        "evidence_rows": sum(int(value) for value in expected_counts.values()),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _receipt_path(config, str(scope["scope_id"]))
    _atomic_json(path, receipt)
    return path


def publish_st_receipts_for_compacted_run(config: Config, run_id: str) -> list[Path]:
    """Publish completed scopes only after this run's staging was compacted."""
    root = config.meta_root / "state" / ST_COVERAGE_CLAIM / f"v{ST_EVIDENCE_VERSION}"
    if not root.exists():
        return []
    published: list[Path] = []
    for path in root.glob("*.json"):
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if checkpoint.get("status") != "complete":
            continue
        if checkpoint.get("completion_run_id") != run_id:
            continue
        published.append(publish_st_coverage_receipt(config, checkpoint))
    return published


def _receipts(config: Config) -> list[dict[str, Any]]:
    root = config.meta_root / "quality" / "coverage" / ST_COVERAGE_CLAIM
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("claim") == ST_COVERAGE_CLAIM:
            out.append(payload)
    return out


def _valid_receipt_scope(receipt: dict[str, Any]) -> tuple[dict[str, Any], date, date] | None:
    """Return a receipt scope only when its integrity fields are consistent."""
    scope = receipt.get("scope")
    completed = receipt.get("completed_symbols")
    if not isinstance(scope, dict) or not isinstance(completed, list):
        return None
    if receipt.get("schema_version") != 1 or receipt.get("status") != "complete":
        return None
    if not all(isinstance(symbol, str) for symbol in completed):
        return None
    if receipt.get("completed_symbols_count") != len(completed):
        return None
    if scope.get("expected_symbols_count") != len(completed):
        return None
    if receipt.get("completed_symbols_sha256") != symbol_scope_hash(completed):
        return None
    if scope.get("symbols_sha256") != symbol_scope_hash(completed):
        return None
    scope_id = scope.get("scope_id")
    identity = {
        key: value
        for key, value in scope.items()
        if key not in {"scope_id", "expected_symbols_count"}
    }
    if not isinstance(scope_id, str) or scope_id != _canonical_hash(identity):
        return None
    try:
        scope_start = date.fromisoformat(str(scope["start"]))
        scope_end = date.fromisoformat(str(scope["end"]))
    except (KeyError, TypeError, ValueError):
        return None
    if scope_start > scope_end:
        return None
    return scope, scope_start, scope_end


def st_evidence_coverage_report(
    config: Config,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    """Return whether a complete, current all-A receipt covers the window."""
    symbols = current_st_universe(config)
    candidates: list[dict[str, Any]] = []
    for receipt in _receipts(config):
        parsed = _valid_receipt_scope(receipt)
        if parsed is None:
            continue
        scope, scope_start, scope_end = parsed
        if scope.get("evidence_version") != ST_EVIDENCE_VERSION:
            continue
        if scope.get("source") != "baostock" or scope.get("universe") != "all_a":
            continue
        covered_symbols = set(receipt.get("completed_symbols", []))
        if not symbols or not set(symbols) <= covered_symbols:
            continue
        if start is not None and scope_start > start:
            continue
        if end is not None and scope_end < end:
            continue
        candidates.append(receipt)

    best = min(
        candidates,
        key=lambda item: (
            date.fromisoformat(item["scope"]["start"]),
            -date.fromisoformat(item["scope"]["end"]).toordinal(),
        ),
        default=None,
    )
    if best is None:
        return {
            "verified": False,
            "claim": ST_COVERAGE_CLAIM,
            "evidence_version": ST_EVIDENCE_VERSION,
            "requested_start": start.isoformat() if start else None,
            "requested_end": end.isoformat() if end else None,
            "current_symbols": len(symbols),
            "reason": "no_current_universe" if not symbols else "no_matching_complete_receipt",
        }
    scope = best["scope"]
    return {
        "verified": True,
        "claim": ST_COVERAGE_CLAIM,
        "evidence_version": ST_EVIDENCE_VERSION,
        "requested_start": start.isoformat() if start else None,
        "requested_end": end.isoformat() if end else None,
        "coverage_start": scope["start"],
        "coverage_end": scope["end"],
        "current_symbols": len(symbols),
        "scope_id": scope["scope_id"],
    }
