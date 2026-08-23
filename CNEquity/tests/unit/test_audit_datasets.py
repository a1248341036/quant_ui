from datetime import date

import polars as pl
import pytest

from cnequity.config import Config
from cnequity.domain.datasets import PARTITION_COLS
from cnequity.domain.schemas import MOCK_SOURCE
from cnequity.quality.audit import run_audit
from cnequity.quality.dataset_checks import (
    audit_curated_dataset,
    check_partition_row_mutation,
)


def _bar_row(symbol: str, trade_date: date) -> dict:
    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000,
        "amount": 10_500.0,
        "source": "tdx_protocol",
        "data_version": "v1",
        "fetched_at": f"{trade_date.isoformat()}T00:00:00+00:00",
    }


def _write_daily_bars_partition(cfg: Config, trade_date: date, symbols: list[str]) -> None:
    path = cfg.curated_root / "daily_bars" / f"trade_date={trade_date.isoformat()}"
    path.mkdir(parents=True)
    pl.DataFrame([_bar_row(sym, trade_date) for sym in symbols]).write_parquet(
        path / "part-merged.parquet"
    )


def test_audit_checks_all_partition_col_datasets(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    run_id = "run-all-datasets"
    trade_date = date(2024, 6, 28)

    run_audit(cfg, run_id, trade_date, {})

    import json

    payload = json.loads(
        (cfg.meta_root / "quality" / "findings" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    exists_checks = {f["dataset"] for f in payload["findings"] if f.get("check") == "exists"}
    assert exists_checks == set(PARTITION_COLS.keys())
    # Optional datasets (source not yet wired) must not fail lake health alone.
    optional_exists = [
        f
        for f in payload["findings"]
        if f.get("check") == "exists" and f.get("dataset") == "economic_calendar"
    ]
    assert optional_exists and optional_exists[0]["severity"] == "warning"
    required_exists = [
        f
        for f in payload["findings"]
        if f.get("check") == "exists" and f.get("dataset") == "daily_bars"
    ]
    assert required_exists and required_exists[0]["severity"] == "error"


def test_row_count_mutation_warns_on_partial_market_drop():
    finding = check_partition_row_mutation(
        "daily_bars",
        "trade_date",
        current_value=date(2024, 6, 28),
        previous_value=date(2024, 6, 27),
        current_stats={"rows": 2400, "symbols": 2400},
        previous_stats={"rows": 5000, "symbols": 5000},
    )
    assert finding is not None
    assert finding["check"] == "row_count_mutation"
    assert finding["severity"] == "warning"
    assert finding["row_ratio"] == pytest.approx(0.48)


def test_row_count_mutation_ignores_small_baselines():
    assert (
        check_partition_row_mutation(
            "trading_calendar",
            "trade_date",
            current_value=date(2024, 6, 28),
            previous_value=date(2024, 6, 27),
            current_stats={"rows": 1, "symbols": None},
            previous_stats={"rows": 1, "symbols": None},
        )
        is None
    )


def test_audit_row_count_mutation_detects_daily_bars_drop(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    trade_date = date(2024, 6, 28)
    prev_date = date(2024, 6, 27)

    prev_symbols = [f"600{i:03d}.SH" for i in range(100)]
    cur_symbols = [f"600{i:03d}.SH" for i in range(40)]
    _write_daily_bars_partition(cfg, prev_date, prev_symbols)
    _write_daily_bars_partition(cfg, trade_date, cur_symbols)

    findings = audit_curated_dataset(
        "daily_bars",
        "trade_date",
        cfg.curated_root / "daily_bars",
        trade_date,
    )
    mutation = [f for f in findings if f.get("check") == "row_count_mutation"]
    assert len(mutation) == 1
    assert mutation[0]["current_rows"] == 40
    assert mutation[0]["previous_rows"] == 100


def test_audit_flags_mock_rows_in_trade_date_partition(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    trade_date = date(2024, 6, 28)
    path = cfg.curated_root / "daily_bars" / f"trade_date={trade_date.isoformat()}"
    path.mkdir(parents=True)
    row = _bar_row("600519.SH", trade_date)
    row["source"] = MOCK_SOURCE
    pl.DataFrame([row]).write_parquet(path / "part-0.parquet")

    findings = audit_curated_dataset(
        "daily_bars",
        "trade_date",
        cfg.curated_root / "daily_bars",
        trade_date,
    )
    mock = [f for f in findings if f.get("check") == "mock_source"]
    assert len(mock) == 1
    assert mock[0]["severity"] == "error"


def test_audit_checks_pk_duplicates_beyond_sample_files(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    trade_date = date(2024, 6, 28)
    path = cfg.curated_root / "daily_bars" / f"trade_date={trade_date.isoformat()}"
    path.mkdir(parents=True)
    for index in range(20):
        pl.DataFrame([_bar_row(f"600{index:03d}.SH", trade_date)]).write_parquet(
            path / f"part-{index:02d}.parquet"
        )
    # The old audit sampled the first 20 files, so this duplicate was invisible.
    pl.DataFrame([_bar_row("600000.SH", trade_date)]).write_parquet(path / "part-20.parquet")

    findings = audit_curated_dataset(
        "daily_bars", "trade_date", cfg.curated_root / "daily_bars", trade_date
    )
    duplicate = [f for f in findings if f.get("check") == "pk_unique"]
    assert duplicate and duplicate[0]["message"].startswith("1 duplicate PK rows")


def test_audit_checks_required_nulls_across_partition(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    trade_date = date(2024, 6, 28)
    path = cfg.curated_root / "daily_bars" / f"trade_date={trade_date.isoformat()}"
    path.mkdir(parents=True)
    row = _bar_row("600519.SH", trade_date)
    row["close"] = None
    pl.DataFrame([row]).write_parquet(path / "part-0.parquet")

    findings = audit_curated_dataset(
        "daily_bars", "trade_date", cfg.curated_root / "daily_bars", trade_date
    )
    required = [f for f in findings if f.get("check") == "required_non_null"]
    assert required and required[0]["null_counts"] == {"close": 1}


def test_full_audit_scans_historical_schema_contract(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    old_date = date(2024, 6, 27)
    current_date = date(2024, 6, 28)

    old = _bar_row("600519.SH", old_date)
    old["close"] = -1.0
    _write_daily_bars_partition(cfg, old_date, ["600519.SH"])
    old_path = (
        cfg.curated_root
        / "daily_bars"
        / f"trade_date={old_date.isoformat()}"
        / "part-merged.parquet"
    )
    pl.DataFrame([old]).write_parquet(old_path)
    _write_daily_bars_partition(cfg, current_date, ["600519.SH"])

    light = audit_curated_dataset(
        "daily_bars",
        "trade_date",
        cfg.curated_root / "daily_bars",
        current_date,
    )
    assert not [f for f in light if f.get("check") == "schema_contract"]

    full = audit_curated_dataset(
        "daily_bars",
        "trade_date",
        cfg.curated_root / "daily_bars",
        current_date,
        full=True,
    )
    contract = [f for f in full if f.get("check") == "schema_contract"]
    assert len(contract) == 1
    assert contract[0]["invalid_files"] == 1
    assert contract[0]["sample"][0]["file"].endswith(
        f"trade_date={old_date.isoformat()}/part-merged.parquet"
    )


def test_full_audit_isolates_an_unreadable_historical_parquet(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    old_date = date(2024, 6, 27)
    current_date = date(2024, 6, 28)
    _write_daily_bars_partition(cfg, current_date, ["600519.SH"])
    broken = cfg.curated_root / "daily_bars" / f"trade_date={old_date.isoformat()}"
    broken.mkdir(parents=True)
    (broken / "part-broken.parquet").write_bytes(b"not a parquet file")

    findings = audit_curated_dataset(
        "daily_bars",
        "trade_date",
        cfg.curated_root / "daily_bars",
        current_date,
        full=True,
    )

    contract = [f for f in findings if f.get("check") == "schema_contract"]
    assert contract and contract[0]["invalid_files"] == 1
    assert contract[0]["sample"][0]["file"].endswith(
        f"trade_date={old_date.isoformat()}/part-broken.parquet"
    )
    assert any(
        f.get("check") == "row_count" and f["message"].startswith("1 rows") for f in findings
    )


def test_lake_health_reports_unreadable_parquet_without_aborting(tmp_path):
    from cnequity.quality.audit import lake_health

    cfg = Config(data_root=tmp_path / "data")
    broken = cfg.curated_root / "daily_bars" / "trade_date=2024-06-27"
    broken.mkdir(parents=True)
    (broken / "part-broken.parquet").write_bytes(b"not a parquet file")

    health = lake_health(cfg, date(2024, 6, 28))

    contract = [
        f
        for f in health["error_findings"]
        if f.get("dataset") == "daily_bars" and f.get("check") == "schema_contract"
    ]
    assert contract and contract[0]["invalid_files"] == 1
    skipped = [f for f in health["warning_findings"] if f.get("check") == "quality_checks_skipped"]
    assert skipped and skipped[0]["datasets"] == ["daily_bars"]
    assert any(
        blocker["code"] == "daily_bars_unreadable"
        for blocker in health["historical_universe_validity"]["blockers"]
    )


def test_run_audit_reports_unreadable_historical_parquet_before_scans(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    current_date = date(2024, 6, 28)
    _write_daily_bars_partition(cfg, current_date, ["600519.SH"])
    broken = cfg.curated_root / "daily_bars" / "trade_date=2024-06-27"
    broken.mkdir(parents=True)
    (broken / "part-broken.parquet").write_bytes(b"not a parquet file")

    run_audit(cfg, "run-broken-history", current_date, {})

    import json

    payload = json.loads(
        (cfg.meta_root / "quality" / "findings" / "run-broken-history.json").read_text(
            encoding="utf-8"
        )
    )
    contract = [
        f
        for f in payload["findings"]
        if f.get("dataset") == "daily_bars" and f.get("check") == "schema_contract"
    ]
    assert contract and any(f["unreadable_files"] == 1 for f in contract)


def test_historical_contract_requires_core_fields_but_allows_nullable_evolution(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    trade_date = date(2024, 6, 28)
    path = cfg.curated_root / "daily_bars" / f"trade_date={trade_date.isoformat()}"
    path.mkdir(parents=True)
    row = _bar_row("600519.SH", trade_date)
    row.pop("amount")  # nullable legacy column added after the original file
    pl.DataFrame([row]).write_parquet(path / "part-0.parquet")
    pl.DataFrame([_bar_row("600520.SH", trade_date)]).write_parquet(path / "part-1.parquet")

    findings = audit_curated_dataset(
        "daily_bars",
        "trade_date",
        cfg.curated_root / "daily_bars",
        trade_date,
        full=True,
    )
    assert not [f for f in findings if f.get("check") == "schema_contract"]

    row.pop("close")
    pl.DataFrame([row]).write_parquet(path / "part-0.parquet")
    findings = audit_curated_dataset(
        "daily_bars",
        "trade_date",
        cfg.curated_root / "daily_bars",
        trade_date,
        full=True,
    )
    contract = [f for f in findings if f.get("check") == "schema_contract"]
    assert contract and "missing required columns" in contract[0]["sample"][0]["message"]


def test_run_audit_persists_source_diff_findings_in_run_file(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    source_finding = {
        "dataset": "daily_bars",
        "check": "backup_coverage_gap",
        "severity": "warning",
        "message": "backup is missing one primary key",
    }
    monkeypatch.setattr(
        "cnequity.quality.audit.run_source_diffs",
        lambda *args, **kwargs: [source_finding],
    )

    count = run_audit(cfg, "run-source-diff", date(2024, 6, 28), {})

    import json

    payload = json.loads(
        (cfg.meta_root / "quality" / "findings" / "run-source-diff.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_finding in payload["findings"]
    assert count == len(payload["findings"])


def test_full_health_includes_a_fresh_source_diff(tmp_path, monkeypatch):
    from cnequity.quality.audit import lake_health

    cfg = Config(data_root=tmp_path / "data")
    source_finding = {
        "dataset": "daily_bars",
        "check": "price_drift",
        "severity": "warning",
        "message": "primary and backup differ",
    }
    monkeypatch.setattr("cnequity.quality.audit._collect_lake_findings", lambda *a, **k: [])
    monkeypatch.setattr(
        "cnequity.quality.audit.run_source_diffs",
        lambda *args, **kwargs: [source_finding],
    )
    monkeypatch.setattr(
        "cnequity.quality.historical_validity.historical_universe_validity",
        lambda *args, **kwargs: {
            "universe_ready": True,
            "window": {"start": None, "end": None},
            "blockers": [],
        },
    )

    health = lake_health(cfg, date(2024, 6, 28))

    assert source_finding in health["warning_findings"]
