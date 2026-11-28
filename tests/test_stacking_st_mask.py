# -*- coding: utf-8 -*-
"""组合层 ST 剔除：训练特征与 label 都要按 ST 行置 NaN。

背景（2026-11-26）：ST 掩码只接入了因子评估层（``EvaluationEngine.evaluate`` 里
``mask_st_values`` 把 ST 行的因子值置 NaN），但 ML 组合的训练/预测走的是
``materialize_factor`` 自算链路，**从未经过该掩码**。结果是"用干净的单因子、
在含 ST 的样本上学"：模型学到 ST 溢价，Top10 全是 ST 股
（2026-10 实测 9~10 只，剔除 ST 后样本外超额从 +14.7% 掉到 +3.8%）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor.stacking.dataset import _apply_st_mask, build_dataset_from_values
from alphaagent.factor.stacking.dataset import FactorEntry


def _panel() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=4)
    idx = pd.MultiIndex.from_product(
        [days, ["000001.SZ", "000002.SZ", "600000.SH"]],
        names=["datetime", "instrument"],
    )
    n = len(idx)
    rng = np.random.default_rng(7)
    close = 10 + rng.normal(size=n)
    return pd.DataFrame(
        {"close": close, "adj_close": close, "open": 10 + rng.normal(size=n)},
        index=idx,
    )


def test_st_rows_become_nan_in_feature_and_label(monkeypatch):
    panel = _panel()
    mask = np.zeros(len(panel), dtype=bool)
    mask[1::3] = True  # 000002 全部日期
    monkeypatch.setattr(
        "alphaagent.factor.metrics.st_mask.st_row_mask", lambda *a, **k: mask
    )
    feat = np.ones((len(panel), 2), dtype=np.float32)
    label = np.ones(len(panel), dtype=np.float64)
    f2, l2 = _apply_st_mask(feat, label, panel)
    assert np.isnan(f2[mask]).all(), "ST 行特征应置 NaN"
    assert np.isfinite(f2[~mask]).all(), "非 ST 行特征不应被改动"
    assert np.isnan(l2[mask]).all(), "ST 行 label 应置 NaN"
    assert np.isfinite(l2[~mask]).all()


def test_fail_open_when_mask_unavailable(monkeypatch):
    panel = _panel()
    monkeypatch.setattr(
        "alphaagent.factor.metrics.st_mask.st_row_mask", lambda *a, **k: None
    )
    feat = np.ones((len(panel), 2), dtype=np.float32)
    label = np.ones(len(panel), dtype=np.float64)
    f2, l2 = _apply_st_mask(feat, label, panel)
    assert np.isfinite(f2).all() and np.isfinite(l2).all()


def test_length_mismatch_is_ignored(monkeypatch):
    panel = _panel()
    monkeypatch.setattr(
        "alphaagent.factor.metrics.st_mask.st_row_mask",
        lambda *a, **k: np.ones(len(panel) + 5, dtype=bool),
    )
    feat = np.ones((len(panel), 1), dtype=np.float32)
    label = np.ones(len(panel), dtype=np.float64)
    f2, l2 = _apply_st_mask(feat, label, panel)
    assert np.isfinite(f2).all() and np.isfinite(l2).all()


def test_build_dataset_drops_st_rows(monkeypatch):
    """端到端：build_dataset_from_values 产出的 dataset 里 ST 行特征全 NaN。"""
    panel = _panel()
    mask = np.zeros(len(panel), dtype=bool)
    mask[0::3] = True
    monkeypatch.setattr(
        "alphaagent.factor.metrics.st_mask.st_row_mask", lambda *a, **k: mask
    )
    entry = FactorEntry(factor_id="test_f1", name="f1", expr="ret1", library="test")
    values = np.linspace(0.1, 1.0, len(panel))
    ds = build_dataset_from_values(
        panel=panel,
        factor_values=[(entry, values)],
        label_days=1,
        mining_end=pd.Timestamp("2024-01-02"),
        size_neutral=False,
        max_corr=1.0,
    )
    assert np.isnan(ds.feature_matrix[mask]).all()
    assert np.isnan(ds.label[mask]).all()
