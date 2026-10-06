# -*- coding: utf-8 -*-
"""`ensure_sorted` 短路（A）与 `winsorized_ic` 单遍化（B）的回归门禁。

评审文档：``docs/review/ocr_review_2026-10-07_submit_split_materialize.md``
- A：已排序面板必须**原样返回同一对象**（热路径零拷贝的来源），乱序时返回排序副本且不改动入参；
- B：`winsorized_ic` 必须与 `evaluate_on_panel(cross_sectional_winsorize_values(...))["ic"]`
  **逐位一致**（hold=1/20），且对非单调面板显式拒绝（M-3），公开入口在 `__all__` 中（L-1）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alphaagent.factor.metrics as metrics_pkg
from alphaagent.data.panel import ensure_sorted
from alphaagent.factor.metrics import evaluate_on_panel, winsorized_ic
from alphaagent.factor.metrics.ic import cross_sectional_winsorize_values

LABEL = "label_20d_close_to_close"


def _panel(days: int = 60, insts: int = 120, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=days), [f"S{i:03d}" for i in range(insts)]],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(seed)
    values = (rng.standard_t(df=2, size=len(idx)) * 1e3).astype(np.float32)
    values[rng.random(len(idx)) < 0.03] = np.nan
    label = (rng.normal(0, 0.02, len(idx)) + values.astype(np.float64) * 1e-6).astype(np.float32)
    label[rng.random(len(idx)) < 0.03] = np.nan
    return pd.DataFrame({LABEL: label}, index=idx), values


@pytest.mark.parametrize("holding_days", [1, 20])
def test_winsorized_ic_is_bit_identical_to_full_evaluate(holding_days: int):
    """B 的等价性契约：逐位相等（不是 approx）。"""
    panel, values = _panel()
    old = evaluate_on_panel(
        cross_sectional_winsorize_values(values, panel),
        panel,
        label_col=LABEL,
        holding_days=holding_days,
    )["ic"]
    new = winsorized_ic(values, panel, label_col=LABEL, holding_days=holding_days)

    assert new == old
    assert np.isfinite(new)


def test_winsorized_ic_rejects_unsorted_panel():
    """M-3：非单调面板会改变按持有期重采样的日点 → 显式拒绝而非静默分叉。"""
    panel, values = _panel()
    with pytest.raises(ValueError, match="已按索引排序"):
        winsorized_ic(
            values[::-1], panel.iloc[::-1], label_col=LABEL, holding_days=20
        )


def test_winsorized_ic_is_exported():
    """L-1：公开入口必须在 __all__ 中（否则 `import *` / API 枚举拿不到）。"""
    assert "winsorized_ic" in metrics_pkg.__all__
    assert callable(metrics_pkg.winsorized_ic)


def test_ensure_sorted_returns_same_object_for_sorted_panel():
    """A 的核心：已排序面板不得再拷贝（15GB take 的根因）。"""
    panel, _ = _panel(days=5, insts=3)
    assert ensure_sorted(panel) is panel


def test_ensure_sorted_sorts_unsorted_panel_without_mutating_input():
    """乱序时必须返回排序副本，且不原地改动入参。"""
    panel, _ = _panel(days=5, insts=3)
    shuffled = panel.iloc[::-1]
    before = shuffled.index.tolist()

    out = ensure_sorted(shuffled)

    assert out is not shuffled
    assert out.index.is_monotonic_increasing
    assert shuffled.index.tolist() == before
    np.testing.assert_allclose(out.to_numpy(), shuffled.sort_index().to_numpy())
