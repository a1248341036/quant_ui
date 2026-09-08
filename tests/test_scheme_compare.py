# -*- coding: utf-8 -*-
"""D 族对照测试：不挑因子（等权/ICIR/HRP 简单投票）的合成信号方向性与健壮性。"""

import numpy as np
import pandas as pd

from alphaagent.factor.stacking.model import (
    _hrp_weights_from_ic,
    fit_scheme_compare_scores,
    scheme_compare_report,
    walk_forward_splits,
)


def _synthetic(n_days: int = 520, n_stocks: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([days, codes], names=["datetime", "instrument"])
    strong = rng.normal(0, 1, len(idx)).astype(np.float32)
    noise = rng.normal(0, 1, (len(idx), 3)).astype(np.float32)
    feats = np.column_stack([strong, noise])
    label = (strong * 0.01 + rng.normal(0, 0.01, len(idx))).astype(np.float32)
    dts = pd.Series(np.repeat(days, n_stocks))
    folds = walk_forward_splits(
        pd.DatetimeIndex(dts.unique()), train_start=dts.min(),
        train_months=6, step_months=4, purge_days=5,
    )
    return feats, label, dts, folds


def _mean_oos_ic(scores: dict[str, np.ndarray], label: np.ndarray, dts: pd.Series, folds) -> dict[str, float]:
    from alphaagent.factor.stacking import daily_spearman_ic

    out: dict[str, float] = {}
    for name, arr in scores.items():
        mask = np.isfinite(arr) & np.isfinite(label)
        if mask.sum() < 20:
            out[name] = float("nan")
            continue
        ic = daily_spearman_ic(arr[mask], label[mask], dts[mask])
        out[name] = float(ic.mean()) if len(ic) else float("nan")
    return out


def test_schemes_capture_strong_signal():
    """强信号因子在场时，三种不挑方案 OOS IC 都应显著为正。"""
    feats, label, dts, folds = _synthetic()
    scores = fit_scheme_compare_scores(feats, label, dts, folds)
    assert set(scores) == {"equal", "icir", "hrp"}
    ics = _mean_oos_ic(scores, label, dts, folds)
    for name, ic in ics.items():
        assert ic == ic, f"{name} OOS 无有效 IC"
        assert ic > 0.01, f"{name} OOS IC={ic:.4f} 应显著为正"


def test_schemes_robust_to_missing_members():
    """成员大面积缺失时按在场归一：仍有值的成员正常投票，不产生 NaN 扩散。"""
    feats, label, dts, folds = _synthetic(seed=9)
    rng = np.random.default_rng(3)
    drop = rng.random(feats.shape) < 0.35
    drop[:, 0] = False  # 强信号成员永远在场
    feats = feats.copy()
    feats[drop] = np.nan
    scores = fit_scheme_compare_scores(feats, label, dts, folds)
    ics = _mean_oos_ic(scores, label, dts, folds)
    for name, ic in ics.items():
        assert ic == ic, f"{name} 缺失场景 OOS 无有效 IC"
        assert ic > 0.005, f"{name} 缺失场景 IC={ic:.4f} 应保持为正"


def test_scheme_compare_report_shape():
    """报告结构：stacked 参考行在前、每行含三口径 + ic_gap、gap 对齐 stacked 为 0。"""
    feats, label, dts, folds = _synthetic()
    stacked = np.full(len(label), np.nan, dtype=np.float32)
    scores = fit_scheme_compare_scores(feats, label, dts, folds)
    # 参考组合：给一个"近似强信号"作为代理（报告侧只要求结构正确、口径自洽）
    stacked[:] = feats[:, 0]
    rep = scheme_compare_report(
        scores, stacked, label, dts,
        first_oos=folds[0].oos_dates.min(), label_horizon=5,
    )
    rows = rep["schemes"]
    assert rows and rows[0]["scheme"] == "stacked"
    assert rows[0]["ic_gap"] == 0.0
    names = {r["scheme"] for r in rows}
    assert {"equal", "icir", "hrp"} <= names
    for r in rows[1:]:
        for key in ("ic_mean", "ic_ir", "oos_sharpe", "oos_max_drawdown", "n_days", "ic_gap"):
            assert key in r
        assert r["ic_gap"] == round(r["ic_mean"] - rows[0]["ic_mean"], 6)
        assert r["label"] and r["note"]


def test_hrp_weights_from_ic_sanity():
    """HRP 配权：非负、和 1；强相关两列应被聚到同团（权重整体向另一列倾斜）。"""
    rng = np.random.default_rng(4)
    n = 160
    x1 = rng.normal(0.05, 1, n)
    frame = pd.DataFrame({
        0: x1,
        1: x1 * 0.99 + rng.normal(0, 0.01, n),   # 与 0 几乎同源
        2: rng.normal(0.02, 1, n),               # 独立噪声
    })
    w = _hrp_weights_from_ic(frame)
    assert w is not None
    assert len(w) == 3
    assert np.all(w >= 0)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    # 同源对(0,1)应共享一半团预算：二者合计权重 ≈ 独立列(2) 的权重
    assert abs((w[0] + w[1]) - w[2]) < 0.35


def test_hrp_weights_insufficient_returns_none():
    """样本/成员不足时 HRP 返回 None（调用方回退等权）。"""
    assert _hrp_weights_from_ic(pd.DataFrame({0: [1.0, 2.0, 3.0]})) is None
    assert _hrp_weights_from_ic(pd.DataFrame({0: [1.0], 1: [2.0]})) is None
