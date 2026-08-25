"""TDX outage fallback: step_instruments degrades to a baostock roster."""

from datetime import date

import polars as pl
import pytest

from cnequity.config import Config
from cnequity.steps.reference import _instruments_from_baostock


class _FakeResultSet:
    def __init__(self, rows):
        self._rows = list(rows)
        self._i = -1
        self.error_code = "0"
        self.error_msg = ""

    def next(self):
        self._i += 1
        return self._i < len(self._rows)

    def get_row_data(self):
        return list(self._rows[self._i])


class _FakeBaostock:
    def __init__(self, rows):
        self._rows = rows

    def login(self):
        return type("R", (), {"error_code": "0", "error_msg": ""})()

    def logout(self):
        pass

    def query_stock_basic(self):
        return _FakeResultSet(self._rows)


# code, code_name, ipoDate, outDate, type, status
_ROWS = [
    ("sh.600519", "贵州茅台", "2001-08-27", "", "1", "1"),
    ("sh.510300", "沪深300ETF", "2012-05-28", "", "5", "1"),
    ("sz.000003", "PT金田A", "1991-01-14", "2002-06-14", "1", "0"),
]


def test_fallback_returns_tagged_baostock_roster(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data", sources={"baostock": True})
    monkeypatch.setattr(
        "cnequity.adapters.baostock.instruments.import_baostock",
        lambda: _FakeBaostock(_ROWS),
    )

    df = _instruments_from_baostock(cfg, RuntimeError("tdx down"))

    assert set(df["source"]) == {"baostock"}
    by_symbol = {r["symbol"]: r for r in df.iter_rows(named=True)}
    assert by_symbol["600519.SH"]["asset_type"] == "stock"
    assert by_symbol["510300.SH"]["asset_type"] == "etf"
    assert by_symbol["000003.SZ"]["delist_date"] == date(2002, 6, 14)
    assert by_symbol["600519.SH"]["list_date"] == date(2001, 8, 27)


def test_fallback_raises_when_baostock_disabled(tmp_path):
    cfg = Config(data_root=tmp_path / "data", sources={"baostock": False})

    with pytest.raises(RuntimeError, match="no fallback roster"):
        _instruments_from_baostock(cfg, RuntimeError("tdx down"))


def test_fallback_raises_on_empty_roster(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data", sources={"baostock": True})
    index_only = [("sh.000001", "上证综指", "1991-07-15", "", "2", "1")]
    monkeypatch.setattr(
        "cnequity.adapters.baostock.instruments.import_baostock",
        lambda: _FakeBaostock(index_only),
    )

    with pytest.raises(RuntimeError, match="returned no rows"):
        _instruments_from_baostock(cfg, RuntimeError("tdx down"))
