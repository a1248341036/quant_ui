# -*- coding: utf-8 -*-
"""特征权重捕获测试：ridge |coef| 份额 / lgbm gain 份额，跨折归一平均、降序、合计 1。"""

import numpy as np
import pandas as pd

from alphaagent.factor.stacking.model import fit_predict_walkforward, walk_forward_splits


def _synthetic(n_days: int = 260, n_stocks: int = 30, seed: int = 7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([days, codes], names=["datetime", "instrument"])
    signal = rng.normal(0, 1, len(idx)).astype(np.float32)
    noise = rng.normal(0, 1, (len(idx), 3)).astype(np.float32)
    feats = np.column_stack([signal, noise])
    label = (signal * 0.01 + rng.normal(0, 0.01, len(idx))).astype(np.float32)
    return feats, label, pd.Series(np.repeat(days, n_stocks))


def test_ridge_feature_weights_captured():
    feats, label, dts = _synthetic()
    folds = walk_forward_splits(pd.DatetimeIndex(dts.unique()), train_start=dts.min(),
                                train_months=4, step_months=3, purge_days=5)
    names = ["alpha_signal", "noise_a", "noise_b", "noise_c"]
    pred, report, weights, _contrib = fit_predict_walkforward(
        feats, label, dts, folds, kind="ridge", feature_names=names
    )
    assert weights is not None and len(weights) == 4
    # 降序
    ws = [w["weight"] for w in weights]
    assert ws == sorted(ws, reverse=True)
    # 合计 ≈ 1
    assert abs(sum(ws) - 1.0) < 1e-6
    # 真信号特征应排第一（其系数显著大于噪声）
    assert weights[0]["name"] == "alpha_signal"
    # 与特征名对齐
    assert {w["name"] for w in weights} == set(names)


def test_lgbm_feature_weights_captured():
    feats, label, dts = _synthetic()
    folds = walk_forward_splits(pd.DatetimeIndex(dts.unique()), train_start=dts.min(),
                                train_months=4, step_months=3, purge_days=5)
    names = ["alpha_signal", "noise_a", "noise_b", "noise_c"]
    pred, report, weights, _contrib = fit_predict_walkforward(
        feats, label, dts, folds, kind="lgbm", feature_names=names
    )
    assert weights is not None
    # round(_, 6) 的展示舍入允许累计误差
    assert abs(sum(w["weight"] for w in weights) - 1.0) < 1e-4
    assert weights[0]["name"] == "alpha_signal"


def test_no_feature_names_returns_none():
    feats, label, dts = _synthetic()
    folds = walk_forward_splits(pd.DatetimeIndex(dts.unique()), train_start=dts.min(),
                                train_months=4, step_months=3, purge_days=5)
    _, _, weights, contrib = fit_predict_walkforward(feats, label, dts, folds, kind="ridge")
    assert weights is None
    assert contrib is None


def test_ridge_permutation_contribution():
    """置换贡献：真信号因子打乱后 IC 损失最大；噪声因子贡献≈0 或负。"""
    feats, label, dts = _synthetic()
    folds = walk_forward_splits(pd.DatetimeIndex(dts.unique()), train_start=dts.min(),
                                train_months=4, step_months=3, purge_days=5)
    names = ["alpha_signal", "noise_a", "noise_b", "noise_c"]
    _, _, _, contrib = fit_predict_walkforward(
        feats, label, dts, folds, kind="ridge", feature_names=names
    )
    assert contrib is not None and len(contrib) == 4
    # 降序排列
    drops = [c["ic_drop"] for c in contrib]
    assert drops == sorted(drops, reverse=True)
    # 真信号排第一且贡献为正（打乱后 IC 明显下降）
    assert contrib[0]["name"] == "alpha_signal"
    assert contrib[0]["ic_drop"] > 0.005
    # ic_drop_rel 存在且首项显著
    assert contrib[0]["ic_drop_rel"] is not None and contrib[0]["ic_drop_rel"] > 0.3


def test_lgbm_permutation_contribution():
    feats, label, dts = _synthetic()
    folds = walk_forward_splits(pd.DatetimeIndex(dts.unique()), train_start=dts.min(),
                                train_months=4, step_months=3, purge_days=5)
    names = ["alpha_signal", "noise_a", "noise_b", "noise_c"]
    _, _, _, contrib = fit_predict_walkforward(
        feats, label, dts, folds, kind="lgbm", feature_names=names
    )
    assert contrib is not None
    assert contrib[0]["name"] == "alpha_signal"
