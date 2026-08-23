from __future__ import annotations

from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.external import tushare_wide
from cnequity.query import list_datasets, load


def _write_year(root, year: int, rows: dict) -> None:
    path = root / str(year) / str(year) / "day"
    path.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(path / "stock_daily.parquet")


def _make_config(tmp_path, archive):
    return Config(
        data_root=tmp_path / "cne-meta",
        external_tushare_wide_enabled=True,
        external_tushare_wide_root=archive,
    )


def test_reads_yearly_tushare_wide_archive_without_curated_copy(tmp_path):
    archive = tmp_path / "quant_dataset"
    _write_year(
        archive,
        2025,
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 3)],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "vol": [100.0, 200.0],
            "amount": [12.0, 25.0],
        },
    )
    cfg = _make_config(tmp_path, archive)

    bars = load("daily_bars", start="2025-01-03", config=cfg)
    assert bars.to_dicts()[0]["symbol"] == "000001.SZ"
    assert bars.to_dicts()[0]["volume"] == 20_000
    assert bars.to_dicts()[0]["amount"] == 25_000.0
    assert bars.to_dicts()[0]["source"] == "quant_dataset_tushare"
    assert not (cfg.curated_root / "daily_bars").exists()

    row = list_datasets(config=cfg).filter(pl.col("dataset") == "daily_bars").to_dicts()[0]
    assert row["has_data"] is True
    assert row["coverage_start"] == date(2025, 1, 2)
    assert row["coverage_end"] == date(2025, 1, 3)
    assert row["watermark"] == date(2025, 1, 3)


def test_fetched_at_is_stable_across_reads(tmp_path):
    """fetched_at must not change between reads of the same row."""
    archive = tmp_path / "quant_dataset"
    _write_year(
        archive,
        2025,
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [date(2025, 1, 2)],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "vol": [100.0],
            "amount": [12.0],
        },
    )
    cfg = _make_config(tmp_path, archive)

    first = load("daily_bars", start="2025-01-02", end="2025-01-02", config=cfg)
    second = load("daily_bars", start="2025-01-02", end="2025-01-02", config=cfg)

    assert first["fetched_at"][0] == second["fetched_at"][0]


def test_coverage_bounds_reads_only_first_and_last_file(tmp_path):
    """coverage_bounds should use the first and last yearly file, not a full scan."""
    archive = tmp_path / "quant_dataset"
    for year, month in [(2023, 1), (2024, 6), (2025, 12)]:
        _write_year(
            archive,
            year,
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [date(year, month, 15)],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "vol": [100.0],
                "amount": [12.0],
            },
        )
    cfg = _make_config(tmp_path, archive)

    first, last = tushare_wide.coverage_bounds(cfg)
    assert first == date(2023, 1, 15)
    assert last == date(2025, 12, 15)


def test_zero_volume_placeholder_rows_filtered(tmp_path):
    """Zero-volume carried-price rows should be excluded from CNE daily_bars."""
    archive = tmp_path / "quant_dataset"
    _write_year(
        archive,
        2025,
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 3)],
            "open": [10.0, 0.0],
            "high": [11.0, 0.0],
            "low": [9.0, 0.0],
            "close": [10.5, 0.0],
            "vol": [100.0, 0.0],
            "amount": [12.0, 0.0],
        },
    )
    cfg = _make_config(tmp_path, archive)

    bars = load("daily_bars", start="2025-01-02", end="2025-01-03", config=cfg)
    assert bars.height == 1
    assert bars["trade_date"][0] == date(2025, 1, 2)
