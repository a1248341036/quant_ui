# -*- coding: utf-8 -*-
"""组合分数尾随 WMA 平滑测试：权重方向、NaN 语义、常数不变性。"""

import numpy as np
import pandas as pd

from scripts.train_ml_composite import _wma_smooth_scores


def _panel_index(days: int = 30, stocks: int = 3) -> pd.MultiIndex:
    dts = pd.bdate_range("2024-01-01", periods=days)
    return pd.MultiIndex.from_product([dts, [f"S{i}" for i in range(stocks)]],
                                      names=["datetime", "instrument"])


def test_constant_series_unchanged():
    idx = _panel_index()
    values = np.full(len(idx), 0.7, dtype=np.float32)
    out = _wma_smooth_scores(values, pd.DataFrame(index=idx), 5)
    np.testing.assert_allclose(out, 0.7, atol=1e-6)


def test_trailing_weights_latest_dominates():
    """窗口内最近一日权重最大：单调递增序列的平滑值应被最新值拉近。"""
    idx = _panel_index(days=10, stocks=1)
    values = np.arange(len(idx), dtype=np.float32)
    out = _wma_smooth_scores(values, pd.DataFrame(index=idx), 3)
    # WMA-3 at t = (x_t*3 + x_{t-1}*2 + x_{t-2}*1)/6
    t = 9
    expected = (values[t] * 3 + values[t - 1] * 2 + values[t - 2] * 1) / 6
    assert abs(out[t] - expected) < 1e-6
    # 平滑值位于最新值与前值之间（滞后性）
    assert values[t - 2] < out[t] < values[t]


def test_nan_stays_nan_and_window_skips_gaps():
    idx = _panel_index(days=6, stocks=1)
    values = np.full(len(idx), 1.0, dtype=np.float32)
    values[:3] = np.nan  # OOS 前无分数
    out = _wma_smooth_scores(values, pd.DataFrame(index=idx), 5)
    assert np.isnan(out[:3]).all()
    np.testing.assert_allclose(out[3:], 1.0, atol=1e-6)


def test_output_length_and_index_alignment():
    idx = _panel_index(days=20, stocks=4)
    values = np.random.default_rng(3).normal(size=len(idx)).astype(np.float32)
    out = _wma_smooth_scores(values, pd.DataFrame(index=idx), 4)
    assert out.shape == values.shape
