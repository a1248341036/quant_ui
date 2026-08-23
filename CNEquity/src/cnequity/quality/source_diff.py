"""Cross-source diff engine — primary curated vs backup snapshots (ADR-0003)."""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from cnequity.config import Config, FailoverDatasetSpec
from cnequity.domain.schemas import PRIMARY_KEYS
from cnequity.query.canonical import dedupe_by_primary_key
from cnequity.storage.atomic import write_json_atomic
from cnequity.storage.source_snapshots import SnapshotStore

logger = logging.getLogger(__name__)

DEFAULT_PRICE_TOLERANCE_BPS = 10.0


def _relative_bps(left: float, right: float) -> float:
    if right == 0:
        return 0.0 if left == 0 else float("inf")
    return abs(left - right) / abs(right) * 10_000.0


def _compare_numeric_fields(
    joined: pl.DataFrame,
    fields: list[str],
    *,
    pk: list[str],
    tolerance_bps: float,
    dataset: str,
    primary_source: str,
    backup_source: str,
) -> list[dict]:
    diffs: list[dict] = []
    for field in fields:
        if field not in joined.columns or f"{field}_backup" not in joined.columns:
            continue
        for row in joined.iter_rows(named=True):
            left = row.get(field)
            right = row.get(f"{field}_backup")
            if left is None or right is None:
                if left is None and right is None:
                    continue
                diffs.append(
                    {
                        "dataset": dataset,
                        "check": "field_null_mismatch",
                        "severity": "warning",
                        "field": field,
                        "primary_source": primary_source,
                        "backup_source": backup_source,
                        "primary_value": left,
                        "backup_value": right,
                        **{k: row.get(k) for k in pk if k in row},
                    }
                )
                continue
            # Relative bps tolerance applies to every configured numeric field
            # (volume differs by a few lots between sources routinely; exact
            # equality checks just spam findings).
            bps = _relative_bps(float(left), float(right))
            if bps <= tolerance_bps:
                continue
            is_price = field in ("open", "high", "low", "close")
            diffs.append(
                {
                    "dataset": dataset,
                    "check": "price_drift" if is_price else "field_drift",
                    "severity": "warning" if is_price else "info",
                    "field": field,
                    "bps": round(bps, 2),
                    "tolerance_bps": tolerance_bps,
                    "primary_source": primary_source,
                    "backup_source": backup_source,
                    "primary_value": float(left),
                    "backup_value": float(right),
                    **{k: row.get(k) for k in pk if k in row},
                }
            )
    return diffs


def _coverage_finding(
    *,
    dataset: str,
    primary_source: str,
    backup_source: str,
    primary: pl.DataFrame,
    backup: pl.DataFrame,
    join_keys: list[str],
) -> dict:
    """Explain whether a source diff had a comparable key universe.

    An empty inner join is not evidence of agreement. It can mean the primary
    source returned nothing, the snapshot belongs to another date, or the two
    feeds covered disjoint symbols. Keep these states explicit so an operator
    cannot read a non-comparison as a clean comparison.
    """
    primary_keys = primary.select(join_keys).unique()
    backup_keys = backup.select(join_keys).unique()
    primary_count = primary_keys.height
    backup_count = backup_keys.height
    overlap = primary_keys.join(backup_keys, on=join_keys, how="inner").height
    missing_backup = primary_keys.join(backup_keys, on=join_keys, how="anti").height
    missing_primary = backup_keys.join(primary_keys, on=join_keys, how="anti").height

    if not primary_count and not backup_count:
        check = "no_comparable_rows"
        severity = "info"
        message = "Primary and backup returned no rows for the comparison window"
    elif not primary_count:
        check = "primary_missing_for_date"
        severity = "warning"
        message = (
            f"backup returned {backup_count} unique key(s), but primary "
            "returned none for the comparison window"
        )
    elif not backup_count:
        check = "backup_missing_for_date"
        severity = "warning"
        message = (
            f"primary returned {primary_count} unique key(s), but backup "
            "returned none for the comparison window"
        )
    elif not overlap:
        check = "no_pk_overlap"
        severity = "warning"
        message = (
            f"primary ({primary_count}) and backup ({backup_count}) returned "
            "disjoint primary-key sets; no fields were comparable"
        )
    elif missing_backup:
        check = "backup_coverage_gap"
        severity = "warning"
        message = (
            f"backup is missing {missing_backup} of {primary_count} primary key(s); "
            f"{overlap} key(s) are comparable"
            + (
                f"; primary is also missing {missing_primary} backup key(s)"
                if missing_primary
                else ""
            )
        )
    elif missing_primary:
        check = "primary_coverage_gap"
        severity = "warning"
        message = (
            f"primary is missing {missing_primary} of {backup_count} backup key(s); "
            f"{overlap} key(s) are comparable"
        )
    else:
        return {}

    return {
        "dataset": dataset,
        "check": check,
        "severity": severity,
        "message": message,
        "primary_source": primary_source,
        "backup_source": backup_source,
        "primary_unique_keys": primary_count,
        "backup_unique_keys": backup_count,
        "overlap_unique_keys": overlap,
        "missing_backup_keys": missing_backup,
        "missing_primary_keys": missing_primary,
    }


def diff_dataset(
    config: Config,
    spec: FailoverDatasetSpec,
    *,
    trade_date: date | None = None,
    sample_limit: int = 500,
) -> list[dict]:
    curated_root = config.curated_root / spec.name
    from cnequity.query.parquet_scan import dataset_has_parquet, scan_parquet_root

    curated_has_data = dataset_has_parquet(curated_root)

    date_col = _date_column(spec.name)
    if curated_has_data:
        lf = scan_parquet_root(
            curated_root,
            partition_col=date_col,
            start=trade_date,
            end=trade_date,
        )
        if "source" in lf.collect_schema().names():
            lf = lf.filter(pl.col("source") == spec.primary)
        primary = dedupe_by_primary_key(lf.collect(), spec.name)
    else:
        primary = pl.DataFrame()

    backup = SnapshotStore(config.meta_root).read_latest(spec.name, source=spec.backup)
    if backup.is_empty():
        if not curated_has_data:
            return [
                {
                    "dataset": spec.name,
                    "check": "no_curated",
                    "severity": "info",
                    "message": f"No curated data for {spec.name}",
                }
            ]
        return [
            {
                "dataset": spec.name,
                "check": "no_snapshot",
                "severity": "info",
                "message": (
                    f"No backup snapshot for {spec.name} source={spec.backup}; "
                    f"primary rows available for comparison: {primary.height}"
                ),
                "primary_rows": primary.height,
            }
        ]

    if trade_date is not None:
        date_col = _date_column(spec.name)
        if date_col and date_col in backup.columns:
            backup = backup.filter(pl.col(date_col) == trade_date)

    pk = PRIMARY_KEYS.get(spec.name, [])
    if not pk:
        return []

    join_keys = [
        k for k in pk if k in backup.columns and (k in primary.columns or primary.is_empty())
    ]
    if not join_keys:
        return []

    # A missing curated root has no schema to carry the PK columns. Add typed
    # empty columns so the coverage report can still say "primary missing".
    if primary.is_empty():
        primary = primary.with_columns(
            [pl.Series(key, [], dtype=backup.schema[key]) for key in join_keys]
        )

    findings: list[dict] = []
    coverage = _coverage_finding(
        dataset=spec.name,
        primary_source=spec.primary,
        backup_source=spec.backup,
        primary=primary,
        backup=backup,
        join_keys=join_keys,
    )
    if coverage:
        findings.append(coverage)

    # Deterministic spread sample: hashing the join keys picks rows across the
    # whole universe instead of always the first N symbols in file order.
    if primary.height > sample_limit:
        primary = (
            primary.with_columns(
                pl.concat_str([pl.col(k).cast(pl.Utf8) for k in join_keys])
                .hash(seed=0)
                .alias("_sample_key")
            )
            .sort("_sample_key")
            .head(sample_limit)
            .drop("_sample_key")
        )
    joined = primary.join(
        backup.select([*join_keys, *[c for c in backup.columns if c not in join_keys]]),
        on=join_keys,
        how="inner",
        suffix="_backup",
    )
    if joined.is_empty():
        return findings

    findings.extend(
        _compare_numeric_fields(
            joined,
            spec.compare_fields,
            pk=pk,
            tolerance_bps=spec.price_tolerance_bps,
            dataset=spec.name,
            primary_source=spec.primary,
            backup_source=spec.backup,
        )
    )
    return findings


def _date_column(dataset: str) -> str | None:
    from cnequity.domain.datasets import DATASETS

    spec = DATASETS.get(dataset)
    return spec.query_date_col if spec else None


def run_source_diffs(
    config: Config,
    run_id: str,
    trade_date: date,
) -> list[dict]:
    if not config.failover_enabled or not config.failover_datasets:
        return []

    all_diffs: list[dict] = []
    for spec in config.failover_datasets:
        try:
            all_diffs.extend(diff_dataset(config, spec, trade_date=trade_date))
        except Exception as exc:
            logger.warning("source_diff failed for %s: %s", spec.name, exc)
            all_diffs.append(
                {
                    "dataset": spec.name,
                    "check": "diff_error",
                    "severity": "error",
                    "message": str(exc),
                }
            )

    out_dir = config.meta_root / "quality" / "source_diffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "trade_date": trade_date.isoformat(),
        "diff_count": len(all_diffs),
        "diffs": all_diffs,
    }
    write_json_atomic(
        out_dir / f"{run_id}.json",
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return all_diffs
