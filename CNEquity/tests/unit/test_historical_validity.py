"""Historical-universe validity stays strict and machine-readable."""

from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.quality.historical_validity import historical_universe_validity
from cnequity.quality.st_coverage import (
    build_st_scope,
    publish_st_coverage_receipt,
    write_st_checkpoint,
)
from cnequity.query.universe import coverage_end_date, coverage_start_date
from cnequity.steps.common import list_trading_dates


def _partition(cfg: Config, dataset: str, day: date, frame: pl.DataFrame) -> None:
    root = cfg.curated_root / dataset / f"trade_date={day.isoformat()}"
    root.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(root / "part-0.parquet")


def _lake(tmp_path) -> Config:
    cfg = Config(data_root=tmp_path / "lake")
    instruments = cfg.curated_root / "instruments"
    instruments.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH"]}).write_parquet(instruments / "part-merged.parquet")
    sessions = list_trading_dates(cfg, date(2020, 1, 2), date(2024, 12, 31))
    bars_root = cfg.curated_root / "daily_bars"
    bars_root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"] * len(sessions),
            "trade_date": sessions,
            "volume": [100] * len(sessions),
        }
    ).write_parquet(bars_root / "part-0.parquet")
    _partition(
        cfg,
        "trading_status",
        date(2019, 12, 31),
        pl.DataFrame(
            {
                "symbol": ["600001.SH"],
                "trade_date": [date(2019, 12, 31)],
                "status": ["st"],
            }
        ),
    )
    scope = build_st_scope(
        ["600519.SH"],
        date(2020, 1, 2),
        date(2024, 12, 31),
        universe="all_a",
    )
    checkpoint = {
        "schema_version": 1,
        "claim": "historical_st_evidence",
        "scope": scope,
        "status": "complete",
        "completed_symbols": ["600519.SH"],
        "evidence_rows_by_symbol": {"600519.SH": 0},
        "unresolved_symbols": [],
    }
    write_st_checkpoint(cfg, checkpoint)
    publish_st_coverage_receipt(cfg, checkpoint)
    return cfg


def _survivorship(*, verified: bool) -> dict:
    return {
        "verified": verified,
        "counts": {
            "pending_probe": 0 if verified else 2,
            "missing_bars": 0,
            "unknown_overlap": 0,
            "terminal_mismatch": 0,
            "missing_instrument": 0,
            "invalid_delist_date": 0,
        },
    }


def test_manifest_is_ready_only_when_all_universe_checks_pass(tmp_path, monkeypatch):
    cfg = _lake(tmp_path)
    monkeypatch.setattr(
        "cnequity.quality.historical_validity.delisted_coverage_report",
        lambda *args, **kwargs: _survivorship(verified=True),
    )

    report = historical_universe_validity(cfg, date(2020, 1, 2), date(2024, 12, 31))

    assert report["universe_ready"] is True
    assert report["blockers"] == []
    assert all(check["passed"] for check in report["checks"].values())


def test_manifest_explains_each_blocking_boundary(tmp_path, monkeypatch):
    cfg = _lake(tmp_path)
    monkeypatch.setattr(
        "cnequity.quality.historical_validity.delisted_coverage_report",
        lambda *args, **kwargs: _survivorship(verified=False),
    )

    report = historical_universe_validity(cfg, date(2019, 1, 1), date(2024, 12, 31))

    assert report["universe_ready"] is False
    assert {blocker["code"] for blocker in report["blockers"]} == {
        "daily_bars_window_incomplete",
        "historical_st_labels_incomplete",
        "delisted_universe_unverified",
    }
    assert all(blocker["remediation"] for blocker in report["blockers"])


def test_manifest_blocks_interior_daily_bar_gap(tmp_path, monkeypatch):
    cfg = _lake(tmp_path)
    bars_path = cfg.curated_root / "daily_bars" / "part-0.parquet"
    bars = pl.read_parquet(bars_path)
    bars.filter(pl.col("trade_date") != date(2020, 1, 3)).write_parquet(bars_path)
    monkeypatch.setattr(
        "cnequity.quality.historical_validity.delisted_coverage_report",
        lambda *args, **kwargs: _survivorship(verified=True),
    )

    report = historical_universe_validity(cfg, date(2020, 1, 2), date(2024, 12, 31))

    assert report["universe_ready"] is False
    blocker = next(b for b in report["blockers"] if b["code"] == "daily_bars_interior_gap")
    assert blocker["missing_sessions"] == 1
    assert blocker["sample_sessions"] == ["2020-01-03"]


def test_manifest_blocks_interior_placeholder_only_day(tmp_path, monkeypatch):
    cfg = _lake(tmp_path)
    bars_path = cfg.curated_root / "daily_bars" / "part-0.parquet"
    bars = pl.read_parquet(bars_path).with_columns(
        pl.when(pl.col("trade_date") == date(2020, 1, 3))
        .then(0)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    bars.write_parquet(bars_path)
    monkeypatch.setattr(
        "cnequity.quality.historical_validity.delisted_coverage_report",
        lambda *args, **kwargs: _survivorship(verified=True),
    )

    report = historical_universe_validity(cfg, date(2020, 1, 2), date(2024, 12, 31))

    blocker = next(b for b in report["blockers"] if b["code"] == "daily_bars_interior_gap")
    assert blocker["missing_sessions"] == 1
    assert blocker["sample_sessions"] == ["2020-01-03"]


def test_coarse_partition_bounds_use_real_rows_for_history_window(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "lake")
    instruments = cfg.curated_root / "instruments"
    instruments.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH"]}).write_parquet(instruments / "part-merged.parquet")

    bars = cfg.curated_root / "daily_bars" / "trade_date=2024-01"
    bars.mkdir(parents=True)
    pl.DataFrame(
        {"symbol": ["600519.SH", "600519.SH"], "trade_date": [date(2024, 1, 15), date(2024, 1, 20)]}
    ).write_parquet(bars / "part-0.parquet")

    scope = build_st_scope(["600519.SH"], date(2024, 1, 1), date(2024, 1, 31), universe="all_a")
    checkpoint = {
        "schema_version": 1,
        "claim": "historical_st_evidence",
        "scope": scope,
        "status": "complete",
        "completed_symbols": ["600519.SH"],
        "evidence_rows_by_symbol": {"600519.SH": 0},
        "unresolved_symbols": [],
    }
    write_st_checkpoint(cfg, checkpoint)
    publish_st_coverage_receipt(cfg, checkpoint)
    monkeypatch.setattr(
        "cnequity.quality.historical_validity.delisted_coverage_report",
        lambda *args, **kwargs: _survivorship(verified=True),
    )

    assert coverage_start_date(cfg, "daily_bars") == date(2024, 1, 15)
    assert coverage_end_date(cfg, "daily_bars") == date(2024, 1, 20)
    report = historical_universe_validity(cfg, date(2024, 1, 1), date(2024, 1, 31))

    assert report["universe_ready"] is False
    assert {blocker["code"] for blocker in report["blockers"]} == {
        "daily_bars_window_incomplete",
    }


def test_daily_bar_coverage_ignores_placeholder_only_boundary_partitions(tmp_path):
    cfg = Config(data_root=tmp_path / "lake")
    for day, volume in (
        (date(2024, 1, 10), 0),
        (date(2024, 1, 15), 100),
        (date(2024, 1, 20), 0),
    ):
        _partition(
            cfg,
            "daily_bars",
            day,
            pl.DataFrame(
                {
                    "symbol": ["600519.SH"],
                    "trade_date": [day],
                    "volume": [volume],
                }
            ),
        )

    assert coverage_start_date(cfg, "daily_bars") == date(2024, 1, 15)
    assert coverage_end_date(cfg, "daily_bars") == date(2024, 1, 15)
