"""A trading day no run ever covered must not read as *fresh*.

2026-10-23 was lost end to end: the 21:30 job never fired, 10-24/25 were weekend
skips, and the 23:00 guard printed "nothing stale as of 2026-10-23". ``is_stale``
tolerates one day of publication lag (margin trading is T+1, disclosure datasets
are quarterly), so a one-day hole sits inside that tolerance forever.

These tests pin the second signal that closes the gap: the watermark is behind
the anchor **and** no successful batch was recorded for that session.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from cnequity.cli.main import missed_day_datasets, stale_fetch_steps
from cnequity.domain.datasets import is_stale
from cnequity.orchestrator.manifest import Manifest

ANCHOR = date(2026, 10, 23)
BEHIND = date(2026, 10, 22)
FETCHED = datetime(2026, 10, 23, 13, 30, tzinfo=timezone.utc)


def _seed(config, dataset: str, day: date, *, partition_col: str = "trade_date") -> None:
    part = config.curated_root / dataset / f"{partition_col}={day}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            partition_col: [day],
            "source": ["tdx_protocol"],
            "data_version": ["v1"],
            "fetched_at": [FETCHED],
        }
    ).write_parquet(part / "part-0.parquet")
    from cnequity.storage.state import StateStore

    StateStore(config.meta_root).set_date(dataset, day)


def _record(config, dataset: str, trade_date: date, status: str = "success") -> None:
    """Write one run + one batch, as a real pipeline run would."""
    manifest = Manifest(config.manifest_path)
    run_id = manifest.start_run("daily:core", {"trade_date": trade_date.isoformat()})
    manifest.start_batch(run_id, "b1", "task-1", dataset)
    manifest.finish_batch(run_id, "b1", status, rows_read=10, rows_written=10)
    manifest.finish_run(run_id, status)


@pytest.fixture
def lake(config):
    _seed(config, "etf_bars", BEHIND)
    return config


def test_a_one_day_hole_is_invisible_to_is_stale(lake):
    """The regression that hid 2026-10-23: tolerance 1 swallows a one-day gap."""
    assert is_stale("etf_bars", BEHIND, ANCHOR) is False


def test_missed_day_is_reported_when_no_batch_landed(lake):
    assert "etf_bars" in missed_day_datasets(lake, ANCHOR)


def test_missed_day_joins_the_stale_fetch_steps(lake):
    """`run daily --stale-only` must retry it, not report "nothing stale"."""
    assert "etf_bars" in stale_fetch_steps(lake, ANCHOR)


def test_landed_batch_keeps_a_publication_lag_fresh(lake):
    """A successful batch on the anchor means the vendor is merely late."""
    _record(lake, "etf_bars", ANCHOR)
    assert "etf_bars" not in missed_day_datasets(lake, ANCHOR)
    assert "etf_bars" not in stale_fetch_steps(lake, ANCHOR)


def test_failed_batch_does_not_count_as_landed(lake):
    """Attempted but failed == the lake still does not have it."""
    _record(lake, "etf_bars", ANCHOR, status="failed")
    assert "etf_bars" in missed_day_datasets(lake, ANCHOR)


def test_batch_for_another_session_does_not_cover_the_anchor(lake):
    """A batch stamped 10-22 says nothing about the 10-23 session."""
    _record(lake, "etf_bars", BEHIND)
    assert "etf_bars" in missed_day_datasets(lake, ANCHOR)


def test_dataset_already_at_the_anchor_is_not_missed(lake):
    _seed(lake, "etf_bars", ANCHOR)
    assert "etf_bars" not in missed_day_datasets(lake, ANCHOR)


def test_non_daily_cadence_is_not_judged(lake):
    """Quarterly disclosures are not on a daily cadence; day-based holes are n/a."""
    _seed(lake, "balancesheet", BEHIND, partition_col="end_date")
    assert "balancesheet" not in missed_day_datasets(lake, ANCHOR)
