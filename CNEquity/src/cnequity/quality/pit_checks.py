"""Semantic checks for point-in-time datasets."""

from __future__ import annotations

from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.domain.datasets import DATASETS
from cnequity.query.parquet_scan import dataset_has_parquet, scan_parquet_root

_MIN_FINANCIAL_ANNOUNCE_DATE = date(2001, 1, 1)


def pit_announce_date_findings(config: Config) -> list[dict]:
    """Reject missing or sentinel announcement dates in every PIT dataset."""
    findings: list[dict] = []
    for spec in DATASETS.values():
        if not spec.pit:
            continue
        root = config.curated_root / spec.name
        if not dataset_has_parquet(root):
            continue
        scan = scan_parquet_root(root, partition_col=spec.partition_col, hive=False)
        columns = set(scan.collect_schema().names())
        if "announce_date" not in columns:
            continue
        invalid = pl.col("announce_date").is_null()
        if spec.name == "financial_statement_items":
            invalid = invalid | (pl.col("announce_date") < _MIN_FINANCIAL_ANNOUNCE_DATE)
        sample_columns = [
            column
            for column in (
                "symbol",
                "report_period",
                "count_date",
                "record_date",
                "item_code",
                "announce_date",
            )
            if column in columns
        ]
        bad = scan.filter(invalid).select(sample_columns).collect()
        if bad.is_empty():
            continue
        if spec.name == "financial_statement_items":
            check = "pit_invalid_announce_date"
            message = (
                f"{bad.height} {spec.name} row(s) have missing or pre-floor announce_date; "
                f"the supported PIT floor is {_MIN_FINANCIAL_ANNOUNCE_DATE.isoformat()}"
            )
        else:
            check = "pit_missing_announce_date"
            message = (
                f"{bad.height} {spec.name} row(s) have no announce_date and cannot be queried PIT"
            )
        findings.append(
            {
                "dataset": spec.name,
                "severity": "error",
                "check": check,
                "message": message,
                "invalid_rows": bad.height,
                "sample": bad.head(8).to_dicts(),
            }
        )
    return findings
