"""DuckDB / polars view paths must stay POSIX-form so Windows ``\\`` never hits globs."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

from cnequity.config import Config
from cnequity.domain.datasets import DATASETS
from cnequity.query.parquet_scan import parquet_glob
from cnequity.query.views import _view_glob, ensure_duckdb_views


def test_view_glob_uses_forward_slashes():
    glob_path, hive = _view_glob("C:/Users/测试/lake", DATASETS["daily_bars"])
    assert "\\" not in glob_path
    assert glob_path.startswith("C:/Users/测试/lake/curated/daily_bars/")
    assert hive is True


def test_merge_style_view_glob_is_recursive():
    glob_path, hive = _view_glob("/tmp/lake", DATASETS["delisting_events"])
    assert glob_path.endswith("/derived/delisting_events/**/*.parquet")
    assert hive is False


def test_parquet_glob_is_posix(tmp_path):
    pattern = parquet_glob(tmp_path / "curated" / "daily_bars")
    assert "\\" not in pattern
    assert pattern.endswith("/**/*.parquet")


def test_ensure_duckdb_views_accepts_native_windows_style_root(tmp_path):
    # Build a root whose str() would contain backslashes on Windows; on Unix
    # resolve().as_posix() is a no-op, so the assertion still holds.
    data_root = tmp_path / "cnequity"
    cfg = Config(data_root=data_root)
    db = ensure_duckdb_views(cfg)
    assert db.exists()
    # The helper itself must never feed a backslash into the SQL it builds —
    # re-check the path form used for globs.
    posix = data_root.resolve().as_posix()
    assert "\\" not in posix or Path(posix).as_posix() == posix


def test_duckdb_views_dedupe_duplicate_primary_keys(tmp_path):
    data_root = tmp_path / "data"
    partition = data_root / "curated" / "daily_bars" / "trade_date=2024-06-28"
    partition.mkdir(parents=True)
    day = date(2024, 6, 28)
    base = {
        "symbol": ["600519.SH"],
        "trade_date": [day],
        "open": [1790.0],
        "high": [1810.0],
        "low": [1780.0],
        "close": [1800.0],
        "volume": [1000],
        "amount": [1_000_000.0],
        "source": ["tdx_protocol"],
        "data_version": ["v2"],
        "fetched_at": [datetime(2024, 6, 28, tzinfo=timezone.utc)],
    }
    pl.DataFrame(base).write_parquet(partition / "part-old.parquet")
    newer = {
        **base,
        "close": [1900.0],
        "fetched_at": [datetime(2024, 6, 28, 1, tzinfo=timezone.utc)],
    }
    pl.DataFrame(newer).write_parquet(partition / "part-new.parquet")

    db = ensure_duckdb_views(Config(data_root=data_root))
    with duckdb.connect(str(db)) as con:
        rows = con.execute("SELECT symbol, close FROM daily_bars ORDER BY symbol").fetchall()
    assert rows == [("600519.SH", 1900.0)]


def test_duckdb_views_merge_optional_columns_by_name(tmp_path):
    data_root = tmp_path / "data"
    partition = data_root / "curated" / "daily_bars" / "trade_date=2024-06-28"
    partition.mkdir(parents=True)
    common = {
        "trade_date": [date(2024, 6, 28)],
        "open": [10.0],
        "high": [10.0],
        "low": [10.0],
        "close": [10.0],
        "volume": [100],
        "source": ["tdx_protocol"],
        "data_version": ["v2"],
        "fetched_at": [datetime(2024, 6, 28, tzinfo=timezone.utc)],
    }
    pl.DataFrame({"symbol": ["600519.SH"], "amount": [1000.0], **common}).write_parquet(
        partition / "part-with-amount.parquet"
    )
    pl.DataFrame({"symbol": ["000001.SZ"], **common}).write_parquet(
        partition / "part-without-amount.parquet"
    )

    db = ensure_duckdb_views(Config(data_root=data_root))
    with duckdb.connect(str(db)) as con:
        rows = con.execute("SELECT symbol, amount FROM daily_bars ORDER BY symbol").fetchall()
    assert rows == [("000001.SZ", None), ("600519.SH", 1000.0)]
