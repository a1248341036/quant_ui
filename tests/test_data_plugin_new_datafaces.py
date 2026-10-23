"""新数据面插件（block_trades / regulatory_events / share_structure）回归测试。

三个数据集均来自 CNE curated（tushare/东财口径），插件将其聚合为日频稀疏薄列。
测试用 monkeypatch 覆盖 `_curated_root` 指向 tmp_path，给出迷你分区 parquet，验证
load() 的输出 shape / 关键列 / 首事件裁剪口径，不依赖真实 CNE 数据。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from alphaagent.data.adapters.plugins import _pitlib
from alphaagent.data.adapters.plugins import block_trades, regulatory_events, share_structure


def _write_dataset(root: Path, dataset: str, table: pl.DataFrame, by: str = "date") -> None:
    """写单个 parquet（不做 hive 分区，read_curated 对无分区文件整包读取）。"""
    d = root / dataset
    d.mkdir(parents=True, exist_ok=True)
    table.write_parquet(d / "data.parquet")


# ─────────────────── block_trades ───────────────────

def test_block_trades_aggregates_daily(tmp_path, monkeypatch):
    raw = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "trade_date": [
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 3),
            ],
            "price": [10.0, 10.1, 5.0],
            "volume": [100.0, 200.0, 300.0],
            "amount": [1000.0, 2000.0, 1500.0],
            "premium_ratio": [-0.05, 0.03, 0.0],
        }
    )
    _write_dataset(tmp_path, "block_trades", raw)
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = block_trades.load("block_trades", start="2024-01-01", end="2024-01-10")

    assert isinstance(out, pd.DataFrame)
    assert {"bt_cnt", "bt_amount", "bt_premium_wavg", "bt_days_since"} <= set(out.columns)

    day2 = out[out["trade_date"] == pd.Timestamp("2024-01-02")].set_index("symbol")
    assert day2.loc["000001.SZ", "bt_cnt"] == 2
    assert day2.loc["000001.SZ", "bt_amount"] == pytest.approx(3000.0)
    # 加权折溢价 = (-0.05*1000 + 0.03*2000)/3000 = 0.0027...
    assert day2.loc["000001.SZ", "bt_premium_wavg"] == pytest.approx(-0.05 * 1000 / 3000 + 0.03 * 2000 / 3000)

    day3 = out[out["trade_date"] == pd.Timestamp("2024-01-03")].set_index("symbol")
    assert day3.loc["000002.SZ", "bt_cnt"] == 1
    assert day3.loc["000002.SZ", "bt_days_since"] == 0


# ─────────────────── regulatory_events ───────────────────

def test_regulatory_events_count_severity_and_flags(tmp_path, monkeypatch):
    raw = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "event_date": [
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 3),
                dt.date(2024, 1, 2),
            ],
            "event_type": ["penalty", "investigation", "warning_letter"],
            "event_id": ["e1", "e2", "e3"],
            "title": ["t", "t", "t"],
        }
    )
    _write_dataset(tmp_path, "regulatory_events", raw)
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = regulatory_events.load("regulatory_events", start="2024-01-01", end="2024-01-10")

    assert isinstance(out, pd.DataFrame)
    assert {"reg_cnt_20d", "reg_days_since", "reg_severity", "reg_flag_investigation"} <= set(out.columns)

    # 000001 在 01-02 有 1 条 penalty，01-03 1 条 investigation → 01-03 计数=2
    d3 = out[(out["symbol"] == "000001.SZ") & (out["trade_date"] == pd.Timestamp("2024-01-03"))].iloc[0]
    assert d3["reg_cnt_20d"] == 2
    assert d3["reg_severity"] == 5  # investigation=5
    assert d3["reg_flag_investigation"] == 1

    # 000002 01-02 warning_letter → 严重度 1
    d2 = out[(out["symbol"] == "000002.SZ") & (out["trade_date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert d2["reg_severity"] == 1
    assert d2["reg_flag_warning_letter"] == 1
    assert d2["reg_flag_investigation"] == 0


def test_regulatory_events_skips_rows_before_first_event(tmp_path, monkeypatch):
    """首事件前的行不产出（PIT：事件不可知）。"""
    raw = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "event_date": [dt.date(2024, 1, 5)],
            "event_type": ["penalty"],
            "event_id": ["e1"],
            "title": ["t"],
        }
    )
    _write_dataset(tmp_path, "regulatory_events", raw)
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = regulatory_events.load("regulatory_events", start="2024-01-01", end="2024-01-10")
    assert out["trade_date"].min() == pd.Timestamp("2024-01-05")


# ─────────────────── share_structure ───────────────────

def test_share_structure_counts_and_float_ratio(tmp_path, monkeypatch):
    raw = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "change_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 4)],
            "total_shares": [100000.0, 100000.0],
            "float_shares": [50000.0, 60000.0],
            "restricted_shares": [50000.0, 40000.0],
            "free_float_shares": [50000.0, 60000.0],
            "change_reason": ["增发A股上市", "转增股上市"],
            "announce_date": [dt.date(2024, 1, 1), dt.date(2024, 1, 3)],
        }
    )
    _write_dataset(tmp_path, "share_structure", raw)
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = share_structure.load("share_structure", start="2024-01-01", end="2024-01-10")

    assert isinstance(out, pd.DataFrame)
    assert {"ss_cnt_20d", "ss_days_since", "ss_float_ratio", "ss_flag_plac", "ss_flag_stock"} <= set(out.columns)

    d4 = out[out["trade_date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert d4["ss_cnt_20d"] == 2
    assert d4["ss_float_ratio"] == pytest.approx(0.6)
    assert d4["ss_flag_stock"] == 1
    assert d4["ss_flag_plac"] == 0

    d2 = out[out["trade_date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert d2["ss_flag_plac"] == 1
    assert d2["ss_float_ratio"] == pytest.approx(0.5)


def test_share_structure_float_ratio_nan_when_total_zero(tmp_path, monkeypatch):
    raw = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "change_date": [dt.date(2024, 1, 2)],
            "total_shares": [0.0],
            "float_shares": [0.0],
            "restricted_shares": [0.0],
            "free_float_shares": [0.0],
            "change_reason": ["增发A股上市"],
            "announce_date": [dt.date(2024, 1, 1)],
        }
    )
    _write_dataset(tmp_path, "share_structure", raw)
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = share_structure.load("share_structure", start="2024-01-01", end="2024-01-10")
    assert pd.isna(out["ss_float_ratio"].iloc[0])