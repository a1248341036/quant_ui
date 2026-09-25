"""EastMoney valuation metrics fetch (PE/PB/PS/market cap)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from cnequity.adapters.eastmoney.valuation import fetch_valuation_metrics, fetch_valuation_metrics_tushare


class _Client:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_fetch_valuation_metrics_maps_fields(monkeypatch):
    raw = [
        {
            "f12": "600519",
            "f13": 1,
            "f9": "35.2",
            "f23": "12.1",
            "f45": "8.4",
            "f20": "2.1e12",
            "f21": "2.0e12",
        }
    ]
    monkeypatch.setattr(
        "cnequity.adapters.eastmoney.valuation.fetch_clist_pages",
        lambda client, fields: raw,
    )
    client = _Client()
    df = fetch_valuation_metrics(date(2024, 6, 28), client=client)
    assert client.closed is False  # caller-provided client must not be closed
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["symbol"] == "600519.SH"
    assert row["trade_date"] == date(2024, 6, 28)
    assert row["pe_ttm"] == 35.2
    assert row["pb"] == 12.1
    assert row["ps_ttm"] == 8.4
    assert row["total_mv"] == 2.1e12
    assert row["float_mv"] == 2.0e12


def test_fetch_valuation_metrics_dedupes_symbols(monkeypatch):
    raw = [
        {
            "f12": "600519",
            "f13": 1,
            "f9": "35.2",
            "f23": "12.1",
            "f45": "8.4",
            "f20": "2.1e12",
            "f21": "2.0e12",
        },
        {
            "f12": "600519",
            "f13": 1,
            "f9": "36.2",
            "f23": "12.2",
            "f45": "8.5",
            "f20": "2.2e12",
            "f21": "2.1e12",
        },
    ]
    monkeypatch.setattr(
        "cnequity.adapters.eastmoney.valuation.fetch_clist_pages",
        lambda client, fields: raw,
    )
    df = fetch_valuation_metrics(date(2024, 6, 28), client=_Client())
    assert df.height == 1
    assert df["pe_ttm"][0] == 36.2


def test_fetch_valuation_metrics_owns_and_closes_default_client(monkeypatch):
    created: list[_Client] = []

    def _factory(**kwargs):
        client = _Client()
        created.append(client)
        return client

    monkeypatch.setattr("cnequity.adapters.eastmoney.valuation.EastMoneyClient", _factory)
    monkeypatch.setattr(
        "cnequity.adapters.eastmoney.valuation.fetch_clist_pages",
        lambda client, fields: [],
    )
    df = fetch_valuation_metrics(date(2024, 6, 28))
    assert df.is_empty()
    assert created[0].closed is True


def test_fetch_valuation_metrics_empty_when_no_rows(monkeypatch):
    monkeypatch.setattr(
        "cnequity.adapters.eastmoney.valuation.fetch_clist_pages",
        lambda client, fields: [],
    )
    df = fetch_valuation_metrics(date(2024, 6, 28), client=_Client())
    assert df.is_empty()


def test_fetch_valuation_metrics_rejects_unmappable_clist_rows(monkeypatch):
    monkeypatch.setattr(
        "cnequity.adapters.eastmoney.valuation.fetch_clist_pages",
        lambda client, fields: [{"f12": "123456"}],
    )
    # clist_rows_to_symbols raises on unmappable rows, so the step-level guard
    # in fundamentals.py catches it; the adapter re-raises via clist_rows_to_symbols.
    with pytest.raises(RuntimeError, match="valuation_metrics clist returned 1 unmappable"):
        fetch_valuation_metrics(date(2024, 6, 28), client=_Client())


def test_fetch_valuation_metrics_closes_owned_client_on_failure(monkeypatch):
    created: list[_Client] = []

    def _factory(**kwargs):
        client = _Client()
        created.append(client)
        return client

    monkeypatch.setattr("cnequity.adapters.eastmoney.valuation.EastMoneyClient", _factory)
    monkeypatch.setattr(
        "cnequity.adapters.eastmoney.valuation.fetch_clist_pages",
        lambda client, fields: (_ for _ in ()).throw(RuntimeError("clist down")),
    )
    with pytest.raises(RuntimeError, match="clist down"):
        fetch_valuation_metrics(date(2024, 6, 28))
    assert created[0].closed is True


# ─── Tushare Fallback Tests ───────────────────────────────────────────────


def test_tushare_fallback_maps_fields_and_unit(monkeypatch):
    """Tushare daily_basic 字段映射 + 万元→元 ×1e4 单位换算。"""
    # Tushare raw data: total_mv/circ_mv are in 万元
    tushare_df = pl.DataFrame({
        "ts_code": ["600519.SH", "000001.SZ"],
        "trade_date": [date(2024, 6, 28)] * 2,
        "pe_ttm": [17.6, 4.4],
        "pb": [6.2, 0.49],
        "ps_ttm": [8.1, 3.0],
        "total_mv": [2100000.0, 22700.0],   # 万元
        "circ_mv": [2000000.0, 22000.0],     # 万元
    })
    # Patch at the source module since _get_pro/_fetch_with_retry are
    # imported locally inside fetch_valuation_metrics_tushare.
    monkeypatch.setattr(
        "cnequity.external.tushare_fetch._get_pro",
        lambda config: None,
    )
    monkeypatch.setattr(
        "cnequity.external.tushare_fetch._fetch_with_retry",
        lambda pro, api, **kwargs: tushare_df,
    )

    class MockConfig:
        external_tushare_wide_interval = 0.3

    df = fetch_valuation_metrics_tushare(date(2024, 6, 28), config=MockConfig())

    assert df.height == 2
    # 验证单位换算：2100000 万元 → 2.1e10 元 (×1e4)
    row = df.filter(pl.col("symbol") == "600519.SH").row(0, named=True)
    assert row["total_mv"] == 2100000.0 * 10000
    assert row["float_mv"] == 2000000.0 * 10000
    # source 标记为 tushare
    assert df["source"].to_list() == ["tushare", "tushare"]


def test_tushare_fallback_filters_universe(monkeypatch):
    """universe 过滤：只保留 daily_bars 中存在的 symbol。"""
    tushare_df = pl.DataFrame({
        "ts_code": ["600519.SH", "999999.SZ"],  # 999999 不在 universe
        "trade_date": [date(2024, 6, 28)] * 2,
        "pe_ttm": [17.6, 5.5],
        "pb": [6.2, 0.5],
        "ps_ttm": [8.1, 3.0],
        "total_mv": [2100000.0, 100.0],
        "circ_mv": [2000000.0, 50.0],
    })
    monkeypatch.setattr(
        "cnequity.external.tushare_fetch._get_pro",
        lambda config: None,
    )
    monkeypatch.setattr(
        "cnequity.external.tushare_fetch._fetch_with_retry",
        lambda pro, api, **kwargs: tushare_df,
    )

    class MockConfig:
        external_tushare_wide_interval = 0.3

    df = fetch_valuation_metrics_tushare(
        date(2024, 6, 28),
        universe={"600519.SH"},
        config=MockConfig(),
    )
    # 999999.SZ 被过滤掉（不在 universe 中）
    assert df.height == 1
    assert df["symbol"][0] == "600519.SH"


def test_tushare_fallback_empty_returns_empty(monkeypatch):
    """Tushare 返回空数据时返回空 DataFrame。"""
    monkeypatch.setattr(
        "cnequity.external.tushare_fetch._get_pro",
        lambda config: None,
    )
    monkeypatch.setattr(
        "cnequity.external.tushare_fetch._fetch_with_retry",
        lambda pro, api, **kwargs: pl.DataFrame(),
    )

    class MockConfig:
        external_tushare_wide_interval = 0.3

    df = fetch_valuation_metrics_tushare(date(2024, 6, 28), config=MockConfig())
    assert df.is_empty()
