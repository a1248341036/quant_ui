"""CNE panel 的 is_trade / is_st / not_st 标记列（历史 bug：预建空列导致恒为 NaN）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.data.adapters.cnequity import _column_all_nan, _enrich_trade_flags
from alphaagent.factor.metrics import st_mask

ST_ROWS = {("000001.SZ", 0), ("000001.SZ", 1), ("000002.SZ", 2)}


def _panel() -> pd.DataFrame:
    """3 只标的 × 4 个交易日；is_trade/not_st 为 OUTPUT_COLUMNS 预建的全 NaN 列。"""
    dates = pd.bdate_range("2024-01-02", periods=4)
    instruments = ["000001.SZ", "000002.SZ", "510300.SH"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    return pd.DataFrame(
        {
            "volume": np.tile([1.0e5, 0.0, 5.0e4], len(dates)),
            "is_st": np.nan,
            "is_trade": np.nan,
            "not_st": np.nan,
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def _clean_mask_cache():
    st_mask.clear_cache()
    yield
    st_mask.clear_cache()


@pytest.fixture
def st_dataset(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"symbol": sym, "trade_date": pd.Timestamp(pd.bdate_range("2024-01-02", periods=4)[i])}
            for sym, i in ST_ROWS
        ]
    )
    root = tmp_path / "stock_st" / "trade_date=2024-01"
    root.mkdir(parents=True)
    df.to_parquet(root / "part-merged.parquet", index=False)
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(tmp_path / "stock_st"))
    monkeypatch.delenv("ALPHA_ST_MASK", raising=False)


def _expected_st_flag(panel: pd.DataFrame) -> np.ndarray:
    dates = pd.bdate_range("2024-01-02", periods=4)
    inst = panel.index.get_level_values("instrument")
    dt = panel.index.get_level_values("datetime")
    return np.array(
        [(i, int(pd.Index(dates).get_loc(d))) in ST_ROWS for i, d in zip(inst, dt)], dtype=int
    )


def test_prebuilt_nan_columns_are_filled(st_dataset) -> None:
    """核心回归：列已存在（全 NaN）时仍必须计算，而不是跳过。"""
    panel = _panel()
    _enrich_trade_flags(panel)

    assert panel["is_trade"].notna().all()
    assert panel["not_st"].notna().all()
    assert panel["is_st"].notna().all()
    expected = _expected_st_flag(panel)
    assert np.array_equal(panel["is_st"].to_numpy(dtype=int), expected)
    assert np.array_equal(panel["not_st"].to_numpy(dtype=int), 1 - expected)
    # is_trade = volume > 0（含停牌 0 成交行为 0）
    assert np.array_equal(
        panel["is_trade"].to_numpy(dtype=int), (panel["volume"] > 0).to_numpy(dtype=int)
    )
    assert panel["is_trade"].dtype == np.int8 and panel["not_st"].dtype == np.int8


def test_etf_rows_not_marked_st(st_dataset) -> None:
    panel = _panel()
    _enrich_trade_flags(panel)
    etf = panel.xs("510300.SH", level="instrument")
    assert (etf["is_st"] == 0).all()
    assert (etf["not_st"] == 1).all()


def test_missing_dataset_falls_back_to_panel_is_st(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(tmp_path / "missing"))
    panel = _panel()
    panel["is_st"] = np.tile([1.0, 0.0, 0.0], 4)  # CNE stock_daily_wide 的 is_st
    _enrich_trade_flags(panel)
    assert np.array_equal(
        panel["not_st"].to_numpy(dtype=int), 1 - panel["is_st"].to_numpy(dtype=int)
    )
    assert panel["is_trade"].notna().all()


def test_missing_dataset_and_empty_is_st_yields_not_st_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(tmp_path / "missing"))
    panel = _panel()  # is_st 全 NaN
    _enrich_trade_flags(panel)
    assert (panel["not_st"] == 1).all()


def test_existing_values_are_not_overwritten(tmp_path, monkeypatch) -> None:
    """数据集不可用时：已有真实 is_trade/not_st 保持原值（只补缺失/空列）。"""
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(tmp_path / "missing"))
    panel = _panel()
    panel["is_trade"] = 7
    panel["not_st"] = 0
    _enrich_trade_flags(panel)
    assert (panel["is_trade"] == 7).all()
    assert (panel["not_st"] == 0).all()


def test_truth_source_wins_when_available(st_dataset) -> None:
    """数据集可用时：is_st/not_st 按唯一真源重写（即使已带旧值），is_trade 保留。"""
    panel = _panel()
    panel["is_st"] = 1  # 旧的 CNE 口径（recall 仅 ~0.29）
    panel["not_st"] = 0
    panel["is_trade"] = 7
    _enrich_trade_flags(panel)
    expected = _expected_st_flag(panel)
    assert np.array_equal(panel["is_st"].to_numpy(dtype=int), expected)
    assert np.array_equal(panel["not_st"].to_numpy(dtype=int), 1 - expected)
    assert (panel["is_trade"] == 7).all()


def test_enrich_is_idempotent(st_dataset) -> None:
    panel = _panel()
    _enrich_trade_flags(panel)
    first = panel.copy()
    _enrich_trade_flags(panel)
    pd.testing.assert_frame_equal(panel, first)


def test_column_all_nan_helper() -> None:
    frame = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, np.nan]})
    assert _column_all_nan(frame, "a") is True
    assert _column_all_nan(frame, "b") is False
    assert _column_all_nan(frame, "missing") is True
