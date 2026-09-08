# -*- coding: utf-8 -*-
"""mRMR 因子推荐测试：强因子优先、高相关克隆被压后、输出结构与 k 上限。"""

import numpy as np
import pandas as pd

from alphaagent.factor.stacking.model import mrmr_rank_features


def _synthetic(seed: int = 21):
    rng = np.random.default_rng(seed)
    n_days, n_stocks = 600, 40
    days = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"{i:06d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([days, codes], names=["datetime", "instrument"])
    n = len(idx)
    strong = rng.normal(0, 1, n).astype(np.float32)
    clone = (strong + rng.normal(0, 0.75, n)).astype(np.float32)   # corr(strong)≈0.8，质量略低
    weak = (strong * 0.55 + rng.normal(0, 0.85, n)).astype(np.float32)  # 相关≈0.54，质量更低
    feats = np.column_stack([strong, clone, weak])
    label = (strong * 0.01 + rng.normal(0, 0.0095, n)).astype(np.float32)
    dts = pd.Series(np.repeat(days, n_stocks))
    return feats, label, dts


def test_mrmr_strong_first_clone_deferred():
    feats, label, dts = _synthetic()
    ranking = mrmr_rank_features(
        feats, label, dts, ["strong", "clone", "weak"],
        window_start="2023-03-01", window_end="2023-12-29", k=3, beta=0.7,
    )
    names = [r["name"] for r in ranking]
    # 强信号首个入选（与克隆质量差 ~0.15 IC，远超抽样噪声）
    assert names[0] == "strong"
    # 克隆被去重机制压到后面（首轮冗余惩罚后不应排在前面）
    assert names.index("clone") > 0
    by_name = {r["name"]: r for r in ranking}
    assert by_name["clone"]["redundancy"] > 0.7   # 与已选 strong 高度相关
    assert by_name["weak"]["redundancy"] < 0.6
    # 得分单调降
    scores = [r["score"] for r in ranking]
    assert scores == sorted(scores, reverse=True)


def test_mrmr_k_cap_and_structure():
    feats, label, dts = _synthetic(seed=5)
    ranking = mrmr_rank_features(
        feats, label, dts, ["strong", "clone", "weak"],
        window_start="2023-03-01", window_end="2023-12-29", k=99,
    )
    assert len(ranking) == 3  # k 超过成员数时输出全部成员
    for r in ranking:
        assert set(r) == {"rank", "name", "ic_mean", "redundancy", "score"}
        assert r["rank"] >= 1
    assert [r["rank"] for r in ranking] == [1, 2, 3]


def test_mrmr_single_member():
    """单成员：正常窗口下输出唯一成员，不崩溃。"""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n).astype(np.float32)
    y = (x * 0.01).astype(np.float32)
    dts = pd.Series(np.repeat(pd.bdate_range("2023-01-02", periods=25), 20))
    ranking = mrmr_rank_features(
        x.reshape(-1, 1), y, dts, ["only"],
        window_start="2023-01-10", window_end="2023-01-20", k=5,
    )
    assert len(ranking) == 1 and ranking[0]["name"] == "only"
