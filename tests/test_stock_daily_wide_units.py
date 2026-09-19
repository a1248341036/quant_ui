"""stock_daily_wide 插件单位归一化（amount 千元→元、vol 手→股、circ_mv/total_mv 万元→元）。

背景：Tushare 宽表原始口径 amount=千元、vol=手、circ_mv/total_mv=万元，
若不换算，build_panel_from_hq 的 vwap = amount/volume 恒为正确值的 1/10
（vwap/close 恒等于 0.1）。本测试锁定 load() 出口的单位契约。
"""
from __future__ import annotations

import pandas as pd
import pytest

from alphaagent.data.adapters.plugins import stock_daily_wide


def _raw_wide() -> pd.DataFrame:
    """模拟 CNE stock_daily_wide 原始行（Tushare 口径：amount 千元、vol 手、mv 万元）。"""
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ts_code": ["000001.SZ", "000002.SZ"],
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            # 千元：10.5 元 × 100 手 × 100 股/手 = 105000 元 = 105 千元
            "amount": [105.0, 410.0],
            # 手：100 手 / 200 手
            "vol": [100.0, 200.0],
            "adj_factor": [1.0, 1.0],
            # 万元：10.5 元 × 10000 股 = 105000 元 = 10.5 万元
            "circ_mv": [10.5, 41.0],
            "total_mv": [12.0, 44.0],
        }
    )


@pytest.fixture(autouse=True)
def _fake_cne_load(monkeypatch):
    """替换 cnequity.query.reader.load 返回原始宽表，绕过真实 CNE 数据湖。"""
    import polars as pl
    import cnequity.query.reader as reader

    def fake_load(dataset, *, start=None, end=None, config=None):
        assert dataset == "stock_daily_wide"
        return pl.from_pandas(_raw_wide())

    monkeypatch.setattr(reader, "load", fake_load)


def test_load_normalizes_units() -> None:
    pdf = stock_daily_wide.load("stock_daily_wide")
    row = pdf.iloc[0]

    # amount: 千元 → 元（105 千元 → 105000 元）
    assert row["amount"] == pytest.approx(105000.0)
    # vol: 手 → 股（100 手 → 10000 股）
    assert row["vol"] == pytest.approx(10000.0)
    # circ_mv/total_mv: 万元 → 元
    assert row["circ_mv"] == pytest.approx(105000.0)
    assert row["total_mv"] == pytest.approx(120000.0)

    # 换算后 vwap = amount/volume 应接近 close（10.5 元）
    vwap = row["amount"] / row["vol"]
    assert vwap == pytest.approx(10.5, rel=1e-6)


def test_load_keeps_ohlc_untouched() -> None:
    pdf = stock_daily_wide.load("stock_daily_wide")
    row = pdf.iloc[0]
    assert row["open"] == pytest.approx(10.0)
    assert row["high"] == pytest.approx(11.0)
    assert row["low"] == pytest.approx(9.0)
    assert row["close"] == pytest.approx(10.5)
    assert row["adj_factor"] == pytest.approx(1.0)


def test_end_to_end_engine_amount_is_yuan() -> None:
    """端到端：插件 load → registry 列映射 → build_panel_from_hq → engine_frame。

    回归保护：插件出口归一化后，下游 alpha_panel_to_engine_frame 不得再 ×1000
    （二次换算会把 amount 放大 1000 倍）。engine amount 应 ≈ close × volume。
    """
    from alphaagent.data.adapters.registry import PluginRegistry
    from core.panel_schema import alpha_panel_to_engine_frame

    registry = PluginRegistry()
    registry.discover()
    panel = registry.build_panel(
        start="2024-01-02", end="2024-01-02",
        include_fundamentals=False, universe_mask=False,
    )
    engine = alpha_panel_to_engine_frame(panel, asset_type="stock")

    # 000001.SZ: close=10.5, volume=10000 股 → amount=105000 元
    row = engine[engine["code"] == "000001.SZ"].iloc[0]
    assert row["amount"] == pytest.approx(105000.0)
    assert row["amount"] == pytest.approx(row["close"] * 10000.0, rel=1e-6)

    # 000002.SZ: close=20.5, volume=20000 股 → amount=410000 元
    row2 = engine[engine["code"] == "000002.SZ"].iloc[0]
    assert row2["amount"] == pytest.approx(410000.0)
    assert row2["amount"] == pytest.approx(row2["close"] * 20000.0, rel=1e-6)
