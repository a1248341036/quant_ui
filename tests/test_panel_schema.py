"""core.panel_schema：面板契约、单位换算与 alpha→engine 转换的单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.panel_schema import (
    AMOUNT_CNE_TO_ENGINE,
    TURNOVER_PERCENT_TO_RATIO,
    alpha_panel_to_engine_frame,
    validate_alpha_panel,
    validate_engine_panel,
)


def _alpha_panel(*, n_days: int = 6) -> pd.DataFrame:
    """合成 AlphaAgent 面板：2 只股票 × n_days 交易日。

    amount 单位千元、turnover_rate 单位 %（CNE/tushare 口径）。
    """
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    inst = ["000001.SZ"] * n_days + ["600000.SH"] * n_days
    n = n_days * 2
    df = pd.DataFrame({
        "open": np.linspace(10.0, 11.0, n),
        "high": np.linspace(10.2, 11.2, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10.1, 11.1, n),
        "volume": np.full(n, 1000.0),
        "amount": np.tile(np.linspace(100.0, 600.0, n_days), 2),  # 千元
        "turnover_rate": np.tile(np.linspace(0.4, 0.9, n_days), 2),  # %
        "adjfactor": np.ones(n),
    }, index=pd.MultiIndex.from_arrays(
        [list(dates) * 2, inst], names=["datetime", "instrument"],
    ))
    return df


def test_alpha_validation_rejects_flat_index():
    panel = _alpha_panel().reset_index()
    with pytest.raises(ValueError, match="alpha_panel_requires_multiindex"):
        validate_alpha_panel(panel)


def test_alpha_validation_rejects_missing_columns():
    panel = _alpha_panel().drop(columns=["turnover_rate"])
    with pytest.raises(ValueError, match="alpha_panel_missing_columns"):
        validate_alpha_panel(panel)


def test_engine_validation_rejects_missing_columns():
    panel = _alpha_panel().reset_index().rename(
        columns={"datetime": "date", "instrument": "code"})
    panel = panel.drop(columns=["turnover_rate"])
    # 缺少引擎必需列 turnover/turn20/am20
    with pytest.raises(ValueError, match="engine_panel_missing_columns"):
        validate_engine_panel(panel)


def test_conversion_units():
    out = alpha_panel_to_engine_frame(_alpha_panel(n_days=6))
    # amount 千元 → 元
    assert out["amount"].iloc[0] == pytest.approx(100.0 * AMOUNT_CNE_TO_ENGINE)
    # turnover % → 比例
    assert out["turnover"].iloc[0] == pytest.approx(0.4 / TURNOVER_PERCENT_TO_RATIO)
    # 输出列与顺序
    assert list(out.columns[:8]) == [
        "date", "code", "open", "high", "low", "close", "turnover", "amount",
    ]
    assert {"am20", "turn20"} <= set(out.columns)
    # 索引扁平化
    assert "date" in out.columns and "code" in out.columns
    assert not isinstance(out.index, pd.MultiIndex)


def test_conversion_rolling_columns():
    out = alpha_panel_to_engine_frame(_alpha_panel(n_days=6))
    # 每只股票独立滚动：min_periods=5，第 5 天才有值
    a20 = out[out["code"] == "000001.SZ"]["am20"].to_numpy()
    assert np.isnan(a20[:4]).all()
    assert np.isfinite(a20[4]).all()
    # 第 5 个元素 = 前 5 天 amount 均值（千元→元后）
    expected = np.mean(np.linspace(100.0, 500.0, 5)) * AMOUNT_CNE_TO_ENGINE
    assert a20[4] == pytest.approx(expected)


def test_conversion_turnover_rank_preserved():
    """turnover % → 比例后应保持截面/时序排序单调性（引擎只依赖 >0 与排名）。"""
    panel = _alpha_panel(n_days=6)
    out = alpha_panel_to_engine_frame(panel)
    before = panel["turnover_rate"].to_numpy()
    after = out["turnover"].to_numpy()
    assert np.all(np.argsort(before) == np.argsort(after))
    assert np.all(after > 0)


def test_conversion_rejects_bad_datetime():
    panel = _alpha_panel(n_days=2)
    panel.index = panel.index.set_levels(
        ["not-a-date", "2026-01-06"], level=0)
    with pytest.raises(ValueError, match="alpha_panel_invalid_datetime"):
        alpha_panel_to_engine_frame(panel)