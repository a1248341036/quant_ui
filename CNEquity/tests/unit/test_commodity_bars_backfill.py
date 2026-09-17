import tempfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from cnequity.config import Config
from cnequity.steps import commodity


def test_commodity_bars_step_backfill():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(data_root=Path(tmpdir))
        cfg.sources["eastmoney"] = True
        cfg.sources["sina"] = True
        cfg._backfill = True
        cfg._backfill_start = date(2026, 9, 14)
        cfg._backfill_end = date(2026, 9, 17)

        sample_df = pl.DataFrame(
            {
                "symbol": ["AU0.SHF", "AU0.SHF"],
                "name": ["沪金主连", "沪金主连"],
                "exchange": ["SHF", "SHF"],
                "trade_date": [date(2026, 9, 14), date(2026, 9, 17)],
                "open": [900.0, 910.0],
                "high": [920.0, 930.0],
                "low": [890.0, 905.0],
                "close": [915.0, 925.0],
                "volume": [1000, 1200],
                "amount": [None, None],
                "open_interest": [100.0, 110.0],
                "source": ["sina", "sina"],
            }
        )

        orig = commodity.fetch_commodity_bars
        try:
            commodity.fetch_commodity_bars = lambda *a, **k: sample_df
            result = commodity.step_commodity_bars(cfg, date(2026, 9, 17), "run-bf-test", {})
            assert result["rows_written"] == 2
        finally:
            commodity.fetch_commodity_bars = orig


def test_commodity_bars_step_backfill_rejects_out_of_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(data_root=Path(tmpdir))
        cfg.sources["eastmoney"] = True
        cfg.sources["sina"] = True
        cfg._backfill = True
        cfg._backfill_start = date(2026, 9, 14)
        cfg._backfill_end = date(2026, 9, 17)

        out_of_bounds_df = pl.DataFrame(
            {
                "symbol": ["AU0.SHF"],
                "name": ["沪金主连"],
                "exchange": ["SHF"],
                "trade_date": [date(2026, 9, 10)],
                "open": [900.0],
                "high": [920.0],
                "low": [890.0],
                "close": [915.0],
                "volume": [1000],
                "amount": [None],
                "open_interest": [100.0],
                "source": ["sina"],
            }
        )

        orig = commodity.fetch_commodity_bars
        try:
            commodity.fetch_commodity_bars = lambda *a, **k: out_of_bounds_df
            with pytest.raises(RuntimeError, match=r"outside 2026-09-14\.\.2026-09-17"):
                commodity.step_commodity_bars(cfg, date(2026, 9, 17), "run-bf-test", {})
        finally:
            commodity.fetch_commodity_bars = orig
