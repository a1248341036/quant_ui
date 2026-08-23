import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from cnequity.config import Config, FailoverDatasetSpec
from cnequity.quality.source_diff import diff_dataset
from cnequity.storage.source_snapshots import (
    SnapshotStore,
    clean_source_snapshots,
)


def _bars_df(symbol: str, close: float, trade_date: date = date(2024, 6, 28)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "trade_date": [trade_date],
            "close": [close],
            "open": [close - 10.0],
            "high": [close + 10.0],
            "low": [close - 20.0],
            "volume": [1000],
            "amount": [1_000_000.0],
            "source": ["eastmoney"],
            "data_version": ["v1"],
            "fetched_at": ["2024-06-28T00:00:00+00:00"],
        }
    )


def test_snapshot_store_roundtrip(tmp_path):
    root = tmp_path / "data"
    store = SnapshotStore(root / "meta")
    path = store.write(
        "daily_bars",
        _bars_df("600519.SH", 1800.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-1",
        batch_id="backup",
        trade_date=date(2024, 6, 28),
    )
    assert path is not None
    out = store.read_latest("daily_bars", source="eastmoney")
    assert out.height == 1


def test_list_files_matches_run_id_as_a_path_component(tmp_path):
    store = SnapshotStore(tmp_path / "meta")
    for run_id, symbol in (("run-1", "600519.SH"), ("run-10", "000001.SZ")):
        store.write(
            "daily_bars",
            _bars_df(symbol, 1800.0),
            source="eastmoney",
            data_version="v1",
            run_id=run_id,
            trade_date=date(2024, 6, 28),
        )

    files = store.list_files("daily_bars", source="eastmoney", run_id="run-1")

    assert len(files) == 1
    assert "run_id=run-1" in files[0].parts


def test_read_latest_dedupes_overlapping_batches_by_primary_key(tmp_path):
    store = SnapshotStore(tmp_path / "meta")
    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1800.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-1",
        batch_id="batch-0",
        trade_date=date(2024, 6, 28),
    )
    newer = _bars_df("600519.SH", 1900.0).with_columns(
        pl.lit("2024-06-28T01:00:00+00:00").alias("fetched_at")
    )
    store.write(
        "daily_bars",
        newer,
        source="eastmoney",
        data_version="v1",
        run_id="run-1",
        batch_id="batch-1",
        trade_date=date(2024, 6, 28),
    )

    out = store.read_latest("daily_bars", source="eastmoney")
    assert out.height == 1
    assert out["close"].to_list() == [1900.0]


def test_read_latest_does_not_mix_data_versions_with_same_run_id(tmp_path):
    store = SnapshotStore(tmp_path / "meta")
    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1800.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-same",
        trade_date=date(2024, 6, 28),
    )
    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1900.0).with_columns(pl.lit("v2").alias("data_version")),
        source="eastmoney",
        data_version="v2",
        run_id="run-same",
        trade_date=date(2024, 6, 28),
    )

    out = store.read_latest("daily_bars", source="eastmoney")

    assert out.height == 1
    assert out["close"].to_list() == [1900.0]
    assert out["data_version"].to_list() == ["v2"]


def test_read_latest_uses_newest_run_only(tmp_path):
    """read_latest must not concat every historical run_id (would grow unbounded)."""
    store = SnapshotStore(tmp_path / "meta")
    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1800.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-old",
        trade_date=date(2024, 6, 28),
    )
    old_dir = (
        tmp_path
        / "meta"
        / "source_snapshots"
        / "daily_bars"
        / "source=eastmoney"
        / "data_version=v1"
        / "run_id=run-old"
    )
    # Ensure mtime ordering: old run older than new.
    older = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
    for path in [old_dir, *old_dir.rglob("*")]:
        if path.exists():
            Path(path).touch()
            os.utime(path, (older, older))

    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1900.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-new",
        trade_date=date(2024, 6, 29),
    )
    out = store.read_latest("daily_bars", source="eastmoney")
    assert out.height == 1
    assert out["close"][0] == 1900.0


def test_read_latest_uses_logical_time_after_directory_restore(tmp_path):
    """Archive restore must not make a newer snapshot look older by mtime."""
    store = SnapshotStore(tmp_path / "meta")
    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1800.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-old",
        trade_date=date(2024, 6, 28),
    )
    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1900.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-new",
        trade_date=date(2024, 6, 29),
    )

    base = tmp_path / "meta" / "source_snapshots" / "daily_bars" / "source=eastmoney"
    old_dir = base / "data_version=v1" / "run_id=run-old"
    new_dir = base / "data_version=v1" / "run_id=run-new"
    old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    new_ts = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
    # Simulate a restore that assigns the newer run the older filesystem mtime.
    for path in [old_dir, *old_dir.rglob("*")]:
        os.utime(path, (old_ts, old_ts))
    for path in [new_dir, *new_dir.rglob("*")]:
        os.utime(path, (new_ts, new_ts))

    out = store.read_latest("daily_bars", source="eastmoney")
    assert out["close"][0] == 1900.0


def test_clean_source_snapshots_keeps_newest_and_recent(tmp_path):
    meta = tmp_path / "meta"
    store = SnapshotStore(meta)
    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1800.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-stale",
        trade_date=date(2024, 6, 1),
    )
    store.write(
        "daily_bars",
        _bars_df("600519.SH", 1900.0),
        source="eastmoney",
        data_version="v1",
        run_id="run-fresh",
        trade_date=date(2024, 6, 28),
    )
    stale = (
        meta
        / "source_snapshots"
        / "daily_bars"
        / "source=eastmoney"
        / "data_version=v1"
        / "run_id=run-stale"
    )
    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).timestamp()
    metadata = stale / "_snapshot.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["updated_at"] = datetime.fromtimestamp(old_ts, timezone.utc).isoformat()
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    for path in [stale, *stale.rglob("*")]:
        os.utime(path, (old_ts, old_ts))

    result = clean_source_snapshots(meta, retention_days=14, dry_run=False)
    assert any("run-stale" in p for p in result.removed_run_dirs)
    assert any("run-fresh" in p for p in result.kept_run_dirs)
    assert not stale.exists()
    assert store.read_latest("daily_bars", source="eastmoney")["close"][0] == 1900.0


def test_source_diff_detects_price_drift(tmp_path):
    root = tmp_path / "data"
    curated = root / "curated" / "daily_bars" / "trade_date=2024-06-28"
    curated.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [date(2024, 6, 28)],
            "close": [1800.0],
            "volume": [1000],
            "open": [1790.0],
            "high": [1810.0],
            "low": [1780.0],
            "amount": [1.0],
            "source": ["tdx_protocol"],
            "data_version": ["v1"],
            "fetched_at": ["2024-06-28T00:00:00+00:00"],
        }
    ).write_parquet(curated / "part-0.parquet")

    store = SnapshotStore(root / "meta")
    store.write(
        "daily_bars",
        pl.DataFrame(
            {
                "symbol": ["600519.SH"],
                "trade_date": [date(2024, 6, 28)],
                "close": [1802.0],
                "volume": [1000],
                "open": [1790.0],
                "high": [1810.0],
                "low": [1780.0],
                "amount": [1.0],
                "source": ["eastmoney"],
                "data_version": ["v1"],
                "fetched_at": ["2024-06-28T00:00:00+00:00"],
            }
        ),
        source="eastmoney",
        data_version="v1",
        run_id="run-1",
        batch_id="backup",
        trade_date=date(2024, 6, 28),
    )

    cfg = Config(
        data_root=root,
        failover_enabled=True,
        failover_datasets=[
            FailoverDatasetSpec(
                name="daily_bars",
                primary="tdx_protocol",
                backup="eastmoney",
                compare_fields=["close", "volume"],
                price_tolerance_bps=10.0,
            )
        ],
    )
    diffs = diff_dataset(cfg, cfg.failover_datasets[0], trade_date=date(2024, 6, 28))
    price_diffs = [d for d in diffs if d.get("check") == "price_drift"]
    assert len(price_diffs) == 1
    assert price_diffs[0]["bps"] > 10.0


def test_source_diff_dedupes_duplicate_primary_rows_before_join(tmp_path):
    day = date(2024, 6, 28)
    root = tmp_path / "data"
    partition = root / "curated" / "daily_bars" / f"trade_date={day.isoformat()}"
    partition.mkdir(parents=True)
    old = _bars_df("600519.SH", 1800.0, day).with_columns(
        pl.lit("tdx_protocol").alias("source"),
        pl.lit("2024-06-28T00:00:00+00:00").alias("fetched_at"),
    )
    new = _bars_df("600519.SH", 1810.0, day).with_columns(
        pl.lit("tdx_protocol").alias("source"),
        pl.lit("2024-06-28T01:00:00+00:00").alias("fetched_at"),
    )
    old.write_parquet(partition / "part-old.parquet")
    new.write_parquet(partition / "part-new.parquet")
    _write_backup_bars(root, ["600519.SH"], day)

    diffs = diff_dataset(
        _source_diff_config(root),
        _source_diff_config(root).failover_datasets[0],
        trade_date=day,
    )
    price_diffs = [finding for finding in diffs if finding.get("check") == "price_drift"]
    assert len(price_diffs) == 1
    assert price_diffs[0]["primary_value"] == 1810.0


def _source_diff_config(root: Path) -> Config:
    return Config(
        data_root=root,
        failover_enabled=True,
        failover_datasets=[
            FailoverDatasetSpec(
                name="daily_bars",
                primary="tdx_protocol",
                backup="eastmoney",
                compare_fields=["close"],
                price_tolerance_bps=10.0,
            )
        ],
    )


def _write_primary_bars(root: Path, symbols: list[str], trade_date: date) -> None:
    frame = pl.concat(
        [_bars_df(symbol, 1800.0 + index, trade_date) for index, symbol in enumerate(symbols)]
    ).with_columns(pl.lit("tdx_protocol").alias("source"))
    part = root / "curated" / "daily_bars" / f"trade_date={trade_date.isoformat()}"
    part.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(part / "part-0.parquet")


def _write_backup_bars(root: Path, symbols: list[str], trade_date: date) -> None:
    frame = pl.concat(
        [_bars_df(symbol, 1800.0 + index, trade_date) for index, symbol in enumerate(symbols)]
    )
    SnapshotStore(root / "meta").write(
        "daily_bars",
        frame,
        source="eastmoney",
        data_version="v1",
        run_id="run-coverage",
        trade_date=trade_date,
    )


def test_source_diff_reports_backup_date_mismatch_as_warning(tmp_path):
    day = date(2024, 6, 28)
    _write_primary_bars(tmp_path, ["600519.SH"], day)
    _write_backup_bars(tmp_path, ["600519.SH"], date(2024, 6, 27))

    diffs = diff_dataset(
        _source_diff_config(tmp_path),
        _source_diff_config(tmp_path).failover_datasets[0],
        trade_date=day,
    )

    finding = next(d for d in diffs if d["check"] == "backup_missing_for_date")
    assert finding["severity"] == "warning"
    assert finding["primary_unique_keys"] == 1
    assert finding["backup_unique_keys"] == 0


def test_source_diff_reports_primary_missing_for_date_as_warning(tmp_path):
    day = date(2024, 6, 28)
    _write_backup_bars(tmp_path, ["600519.SH"], day)

    cfg = _source_diff_config(tmp_path)
    diffs = diff_dataset(cfg, cfg.failover_datasets[0], trade_date=day)

    finding = next(d for d in diffs if d["check"] == "primary_missing_for_date")
    assert finding["severity"] == "warning"
    assert finding["primary_unique_keys"] == 0
    assert finding["backup_unique_keys"] == 1


def test_source_diff_reports_partial_backup_coverage(tmp_path):
    day = date(2024, 6, 28)
    _write_primary_bars(tmp_path, ["600519.SH", "600520.SH"], day)
    _write_backup_bars(tmp_path, ["600519.SH", "600521.SH"], day)

    cfg = _source_diff_config(tmp_path)
    diffs = diff_dataset(cfg, cfg.failover_datasets[0], trade_date=day)

    finding = next(d for d in diffs if d["check"] == "backup_coverage_gap")
    assert finding["severity"] == "warning"
    assert finding["primary_unique_keys"] == 2
    assert finding["backup_unique_keys"] == 2
    assert finding["overlap_unique_keys"] == 1
    assert finding["missing_backup_keys"] == 1


def test_source_diff_reports_primary_coverage_gap(tmp_path):
    day = date(2024, 6, 28)
    _write_primary_bars(tmp_path, ["600519.SH"], day)
    _write_backup_bars(tmp_path, ["600519.SH", "600520.SH"], day)

    cfg = _source_diff_config(tmp_path)
    diffs = diff_dataset(cfg, cfg.failover_datasets[0], trade_date=day)

    finding = next(d for d in diffs if d["check"] == "primary_coverage_gap")
    assert finding["severity"] == "warning"
    assert finding["primary_unique_keys"] == 1
    assert finding["backup_unique_keys"] == 2
    assert finding["overlap_unique_keys"] == 1
    assert finding["missing_primary_keys"] == 1
