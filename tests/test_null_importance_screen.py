# -*- coding: utf-8 -*-
"""Null Importance 预筛选测试：真信号通过、噪声/高分布属性因子被压下、可复现、审计完整。"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from alphaagent.factor.stacking.screening import (
    apply_screening,
    null_importance_scores,
)


def _synthetic(seed: int = 7, n_days: int = 300, n_stocks: int = 30):
    """三类因子：signal（IC≈0.4 真信号）/ noise（纯随机）/ style（行内固定
    排名——连续、可切分点多的"分布属性好看"特征，与标签独立）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"{i:06d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([days, codes], names=["datetime", "instrument"])
    n = len(idx)
    sig_true = rng.normal(0, 1, n)
    signal = (sig_true + rng.normal(0, 0.5, n)).astype(np.float32)
    noise = rng.normal(0, 1, n).astype(np.float32)
    style = np.tile(np.linspace(0.0, 1.0, n_stocks), n_days).astype(np.float32)
    feats = np.column_stack([signal, noise, style])
    label = (0.01 * sig_true + rng.normal(0, 0.02, n)).astype(np.float32)
    dts = pd.Series(np.repeat(days, n_stocks))
    return feats, label, dts


def test_signal_passes_noise_and_style_score_lower():
    """灵魂断言：真信号 score 显著为正且最高；噪声与高分布属性因子被压下。"""
    feats, label, dts = _synthetic()
    rows = null_importance_scores(
        feats, label, dts, ["signal", "noise", "style"],
        window_start="2023-01-02", window_end="2024-12-31",
        n_runs=20, num_boost_round=60,
    )
    by_name = {r["name"]: r for r in rows}
    # 真信号：score > 0（真实重要性超过噪声本底 75 分位）
    assert by_name["signal"]["score"] > 0
    assert by_name["signal"]["passed"] is True
    # 判别力：signal 分数高于 noise 与 style
    assert by_name["signal"]["score"] > by_name["noise"]["score"]
    assert by_name["signal"]["score"] > by_name["style"]["score"]
    # 输出按 score 降序
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_structure_and_determinism():
    """输出字段齐全；同参数两次调用结果逐字段一致（种子固定）。"""
    feats, label, dts = _synthetic(seed=11)
    kwargs = dict(window_start="2023-01-02", window_end="2024-12-31",
                  n_runs=10, num_boost_round=40)
    rows1 = null_importance_scores(feats, label, dts, ["signal", "noise", "style"], **kwargs)
    rows2 = null_importance_scores(feats, label, dts, ["signal", "noise", "style"], **kwargs)
    for r1, r2 in zip(rows1, rows2):
        assert set(r1) == {"name", "actual", "null_p75", "score", "passed"}
        assert r1 == r2


def test_higher_min_score_never_passes_more():
    """通过线调高后，passed 名单只能是原名单的子集。"""
    feats, label, dts = _synthetic(seed=3)
    base = null_importance_scores(
        feats, label, dts, ["signal", "noise", "style"],
        window_start="2023-01-02", window_end="2024-12-31",
        n_runs=10, num_boost_round=40,
    )
    strict = null_importance_scores(
        feats, label, dts, ["signal", "noise", "style"],
        window_start="2023-01-02", window_end="2024-12-31",
        n_runs=10, num_boost_round=40, min_score=5.0,
    )
    assert {r["name"] for r in strict if r["passed"]} <= {r["name"] for r in base if r["passed"]}


def test_insufficient_window_all_fail():
    """窗口无样本：无法区分真信号与本底，全部判 fail 并标注原因（宁缺毋滥）。"""
    feats, label, dts = _synthetic(n_days=30)
    rows = null_importance_scores(
        feats, label, dts, ["signal", "noise", "style"],
        window_start="2030-01-01", window_end="2031-01-01",
        n_runs=5, num_boost_round=20,
    )
    assert len(rows) == 3
    assert all(r["passed"] is False for r in rows)
    assert all("insufficient_window_samples" in (r.get("reason") or "") for r in rows)


def test_apply_screening_drops_and_audits():
    """列裁剪 + entries 同步 + dropped 审计（library 取自裁剪前映射，不错位）。"""
    ds = SimpleNamespace(
        feature_matrix=np.column_stack(
            [np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
             np.full(10, 2.0, dtype=np.float32)]
        ),
        feature_names=["a", "b", "c"],
        entries=[
            SimpleNamespace(name="a", library="production_x"),
            SimpleNamespace(name="b", library="production_x"),
            SimpleNamespace(name="c", library="candidate_x"),
        ],
        dropped=[{"name": "pre", "library": "", "reason": "existing"}],
    )
    rows = [
        {"name": "a", "actual": 100.0, "null_p75": 10.0, "score": 2.2, "passed": True},
        {"name": "b", "actual": 5.0, "null_p75": 10.0, "score": -0.8, "passed": False},
        {"name": "c", "actual": 1.0, "null_p75": 1.0, "score": -0.7, "passed": False},
    ]
    apply_screening(ds, rows)
    assert ds.feature_names == ["a"]
    assert [e.name for e in ds.entries] == ["a"]
    assert ds.feature_matrix.shape == (10, 1)
    dropped = {d["name"]: d for d in ds.dropped}
    assert set(dropped) == {"pre", "b", "c"}
    assert dropped["b"]["library"] == "production_x"
    assert dropped["c"]["library"] == "candidate_x"
    assert "null_importance_score" in dropped["b"]["reason"]


def test_apply_screening_preserves_insufficient_window_reason():
    """M2 回归：rows 自带 reason（样本不足/未真正执行筛选）时，dropped 审计必须
    原样透传，不得伪装成"正常筛选后 score 不过线"。"""
    ds = SimpleNamespace(
        feature_matrix=np.zeros((4, 2), dtype=np.float32),
        feature_names=["a", "b"],
        entries=[
            SimpleNamespace(name="a", library="candidate_x"),
            SimpleNamespace(name="b", library="production_x"),
        ],
        dropped=[],
    )
    rows = [
        {"name": "a", "actual": None, "null_p75": None, "score": None,
         "passed": False, "reason": "insufficient_window_samples=12"},
        {"name": "b", "actual": None, "null_p75": None, "score": None,
         "passed": False, "reason": "insufficient_window_samples=12"},
    ]
    apply_screening(ds, rows)
    assert ds.feature_names == []
    dropped = {d["name"]: d for d in ds.dropped}
    assert dropped["a"]["reason"] == "insufficient_window_samples=12"
    assert dropped["b"]["reason"] == "insufficient_window_samples=12"


def test_apply_screening_drops_to_zero_is_extraordinary():
    """M1 语义：筛选把特征清到 0 列是可能的（样本不足全 fail），调用方应能据此
    显式中止，而不是让空特征矩阵流进训练。此处验证 apply_screening 的边界行为
    （0 列），train_ml_composite 侧已加 <2 防护。"""
    ds = SimpleNamespace(
        feature_matrix=np.zeros((4, 1), dtype=np.float32),
        feature_names=["a"],
        entries=[SimpleNamespace(name="a", library="candidate_x")],
        dropped=[],
    )
    rows = [
        {"name": "a", "actual": None, "null_p75": None, "score": None,
         "passed": False, "reason": "insufficient_window_samples=12"},
    ]
    apply_screening(ds, rows)
    assert ds.feature_names == []
    assert ds.feature_matrix.shape == (4, 0)
    assert [d["name"] for d in ds.dropped] == ["a"]
