from datetime import date

import polars as pl
import pytest

from cnequity.config import Config
from cnequity.quality.audit import run_audit
from cnequity.quality.st_coverage import (
    build_st_scope,
    current_st_universe,
    publish_st_coverage_receipt,
    st_evidence_coverage_report,
    write_st_checkpoint,
)
from cnequity.query.universe import coverage_start_date, trading_status_coverage_start
from cnequity.steps.finalize import step_audit


def _write_status_partition(
    cfg: Config,
    trade_date: date,
    status: str = "normal",
    *,
    source: str = "eastmoney",
) -> None:
    path = cfg.curated_root / "trading_status" / f"trade_date={trade_date.isoformat()}"
    path.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [trade_date],
            "is_trading": [status != "suspended"],
            "status": [status],
            "source": [source],
            "data_version": ["v1"],
            "fetched_at": [f"{trade_date.isoformat()}T00:00:00+00:00"],
        }
    ).write_parquet(path / "part-0.parquet")


def _write_bars_partition(cfg: Config, trade_date: date) -> None:
    path = cfg.curated_root / "daily_bars" / f"trade_date={trade_date.isoformat()}"
    path.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [trade_date],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
            "amount": [10_500.0],
            "source": ["tdx_protocol"],
            "data_version": ["v1"],
            "fetched_at": [f"{trade_date.isoformat()}T00:00:00+00:00"],
        }
    ).write_parquet(path / "part-0.parquet")


def _write_st_receipt(cfg: Config, start: date, end: date) -> None:
    root = cfg.curated_root / "instruments"
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": ["600519.SH"]}).write_parquet(root / "part-merged.parquet")
    scope = build_st_scope(["600519.SH"], start, end, universe="all_a")
    checkpoint = {
        "schema_version": 1,
        "claim": "historical_st_evidence",
        "scope": scope,
        "status": "complete",
        "completed_symbols": ["600519.SH"],
        "evidence_rows_by_symbol": {"600519.SH": 1},
        "unresolved_symbols": [],
    }
    write_st_checkpoint(cfg, checkpoint)
    publish_st_coverage_receipt(cfg, checkpoint)


def test_trading_status_coverage_start_from_partitions(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    _write_status_partition(cfg, date(2024, 6, 27))
    _write_status_partition(cfg, date(2024, 6, 28))
    assert trading_status_coverage_start(cfg) == date(2024, 6, 27)
    assert coverage_start_date(cfg, "daily_bars") is None


def test_current_st_universe_reads_nested_instrument_fragments(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    root = cfg.curated_root / "instruments"
    root.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH"]}).write_parquet(root / "part-merged.parquet")
    nested = root / ".old-fragments"
    nested.mkdir()
    pl.DataFrame({"symbol": ["000001.SZ"]}).write_parquet(nested / "part-old.parquet")

    assert current_st_universe(cfg) == ["000001.SZ", "600519.SH"]


def test_current_st_universe_excludes_cdrs(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    instruments = cfg.curated_root / "instruments"
    instruments.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH", "689009.SH"]}).write_parquet(
        instruments / "part-merged.parquet"
    )

    assert current_st_universe(cfg) == ["600519.SH"]


def test_current_st_universe_ignores_zero_volume_placeholders(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    instruments = cfg.curated_root / "instruments"
    instruments.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH", "000001.SZ"]}).write_parquet(
        instruments / "part-merged.parquet"
    )
    bars = cfg.curated_root / "daily_bars" / "trade_date=2024-06-28"
    bars.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "trade_date": [date(2024, 6, 28)] * 2,
            "volume": [0, 100],
        }
    ).write_parquet(bars / "part-0.parquet")

    assert current_st_universe(cfg) == ["000001.SZ"]


def test_current_st_universe_keeps_legacy_rows_in_mixed_daily_schema(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    instruments = cfg.curated_root / "instruments"
    instruments.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH", "000001.SZ"]}).write_parquet(
        instruments / "part-merged.parquet"
    )
    bars = cfg.curated_root / "daily_bars"
    bars.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH"], "trade_date": [date(2024, 6, 27)]}).write_parquet(
        bars / "legacy.parquet"
    )
    pl.DataFrame(
        {"symbol": ["000001.SZ"], "trade_date": [date(2024, 6, 28)], "volume": [0]}
    ).write_parquet(bars / "current.parquet")

    assert current_st_universe(cfg) == ["600519.SH"]


def test_current_st_universe_does_not_shrink_scope_around_corrupt_input(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    instruments = cfg.curated_root / "instruments"
    instruments.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH"]}).write_parquet(instruments / "valid.parquet")
    (instruments / "broken.parquet").write_bytes(b"not a parquet file")

    assert current_st_universe(cfg) == []


def test_current_st_universe_refuses_partial_bar_scope_when_a_file_is_corrupt(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    instruments = cfg.curated_root / "instruments"
    instruments.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH"]}).write_parquet(instruments / "part-merged.parquet")
    bars = cfg.curated_root / "daily_bars"
    bars.mkdir(parents=True)
    pl.DataFrame(
        {"symbol": ["600519.SH"], "trade_date": [date(2024, 6, 27)], "volume": [100]}
    ).write_parquet(bars / "valid.parquet")
    (bars / "broken.parquet").write_bytes(b"not a parquet file")

    assert current_st_universe(cfg) == []


def test_st_coverage_ignores_tampered_receipt(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    start = end = date(2024, 6, 27)
    _write_status_partition(cfg, start, source="baostock")
    _write_st_receipt(cfg, start, end)

    import json

    receipt_path = next(
        (cfg.meta_root / "quality" / "coverage" / "historical_st_evidence").glob("*.json")
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["completed_symbols_sha256"] = "tampered"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    report = st_evidence_coverage_report(cfg, start, end)
    assert report["verified"] is False
    assert report["reason"] == "no_matching_complete_receipt"


def test_st_receipt_rejects_duplicate_primary_rows(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    day = date(2024, 6, 27)
    path = cfg.curated_root / "trading_status" / f"trade_date={day.isoformat()}"
    path.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "600519.SH"],
            "trade_date": [day, day],
            "is_trading": [True, True],
            "status": ["st", "st"],
            "source": ["baostock", "baostock"],
            "data_version": ["v1", "v1"],
            "fetched_at": [
                "2024-06-27T00:00:00+00:00",
                "2024-06-27T01:00:00+00:00",
            ],
        }
    ).write_parquet(path / "part-0.parquet")
    scope = build_st_scope(["600519.SH"], day, day, universe="all_a")
    checkpoint = {
        "schema_version": 1,
        "claim": "historical_st_evidence",
        "scope": scope,
        "status": "complete",
        "completed_symbols": ["600519.SH"],
        "evidence_rows_by_symbol": {"600519.SH": 2},
        "unresolved_symbols": [],
    }

    with pytest.raises(ValueError, match="reach curated storage"):
        publish_st_coverage_receipt(cfg, checkpoint)


def test_st_receipt_counts_baostock_before_cross_source_dedupe(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    day = date(2024, 6, 27)
    path = cfg.curated_root / "trading_status" / f"trade_date={day.isoformat()}"
    path.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "600519.SH"],
            "trade_date": [day, day],
            "is_trading": [True, True],
            "status": ["st", "normal"],
            "source": ["baostock", "eastmoney"],
            "data_version": ["v1", "v1"],
            "fetched_at": [
                "2024-06-27T00:00:00+00:00",
                "2024-06-27T01:00:00+00:00",
            ],
        }
    ).write_parquet(path / "part-0.parquet")
    scope = build_st_scope(["600519.SH"], day, day, universe="all_a")
    checkpoint = {
        "schema_version": 1,
        "claim": "historical_st_evidence",
        "scope": scope,
        "status": "complete",
        "completed_symbols": ["600519.SH"],
        "evidence_rows_by_symbol": {"600519.SH": 1},
        "unresolved_symbols": [],
    }

    assert publish_st_coverage_receipt(cfg, checkpoint).exists()


def test_audit_warns_when_st_labels_lag_bars(tmp_path):
    """Suspension covered historically, but ST labels only start late → warning."""
    cfg = Config(data_root=tmp_path / "data")
    _write_bars_partition(cfg, date(2016, 1, 4))
    _write_status_partition(cfg, date(2016, 1, 4), status="suspended")  # historical suspension
    _write_status_partition(cfg, date(2024, 6, 28), status="st")  # ST only recent
    run_id = "run-ts-coverage"
    trade_date = date(2024, 6, 28)

    run_audit(cfg, run_id, trade_date, {})

    import json

    findings = json.loads(
        (cfg.meta_root / "quality" / "findings" / f"{run_id}.json").read_text(encoding="utf-8")
    )["findings"]
    coverage = [f for f in findings if f.get("check") == "trading_status_coverage_start"]
    assert len(coverage) == 1
    assert coverage[0]["severity"] == "warning"
    assert coverage[0]["coverage_start"] == "2016-01-04"
    assert coverage[0]["st_coverage_start"] == "2024-06-28"
    assert "ST evidence" in coverage[0]["message"]


def test_audit_coverage_info_when_st_and_suspension_aligned(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    _write_bars_partition(cfg, date(2024, 6, 28))
    _write_status_partition(cfg, date(2024, 6, 28), status="st", source="baostock")
    _write_st_receipt(cfg, date(2024, 6, 28), date(2024, 6, 28))
    run_id = "run-aligned"
    trade_date = date(2024, 6, 28)

    step_audit(cfg, trade_date, run_id, {})

    import json

    payload = json.loads(
        (cfg.meta_root / "quality" / "findings" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    coverage = [f for f in payload["findings"] if f.get("check") == "trading_status_coverage_start"]
    assert len(coverage) == 1
    assert coverage[0]["severity"] == "info"
    assert coverage[0]["st_evidence_verified"] is True
