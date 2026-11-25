# -*- coding: utf-8 -*-
"""weight_mode="score"：按分数秩加权建仓。

契约：
- 默认 "equal" 行为完全不变（向后兼容）；
- "score" 只改票内分配，总仓位与等权一致；
- 非法取值回落等权，不抛错。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import CODES, END, START
from core.engine import run_backtest


def _scores(panel: pd.DataFrame) -> pd.DataFrame:
    """构造一个固定分数梯度：000004 最高，000001 最低。"""
    dts = sorted(panel["date"].unique())
    vals = {c: float(i + 1) for i, c in enumerate(CODES)}
    return pd.DataFrame({c: [vals[c]] * len(dts) for c in CODES},
                        index=pd.DatetimeIndex(dts))


def _run(panel, **kw):
    return run_backtest(
        panel=panel, codes=CODES, factor="pred", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
        external_scores=_scores(panel), amount_q=0.0,
        buy_cost=0.0, sell_cost=0.0, **kw)


def test_score_mode_changes_path_but_not_position_count(panel):
    eq = _run(panel)
    sc = _run(panel, weight_mode="score")
    assert len(sc["nav"]) == len(eq["nav"])
    # 权重不同 → 净值路径不同（同样选股、不同配比）
    assert not np.allclose(eq["nav"].to_numpy(dtype=float), sc["nav"].to_numpy(dtype=float))


def test_unknown_weight_mode_falls_back_to_equal(panel):
    eq = _run(panel)
    odd = _run(panel, weight_mode="nonsense")
    assert np.allclose(eq["nav"].to_numpy(dtype=float), odd["nav"].to_numpy(dtype=float))


def test_rank_weights_favor_higher_score():
    """秩权重公式：3 只时 = 1/6, 2/6, 3/6，和为 1。"""
    scores = np.array([1.0, 2.0, 3.0])
    r = np.argsort(np.argsort(scores)).astype(float) + 1.0
    w = r / r.sum()
    assert np.allclose(w, np.array([1 / 6, 2 / 6, 3 / 6]))
    assert w.sum() == pytest.approx(1.0)


def test_equal_mode_default_unchanged(panel):
    """不传 weight_mode 与显式 equal 完全一致。"""
    a = _run(panel)
    b = _run(panel, weight_mode="equal")
    assert np.allclose(a["nav"].to_numpy(dtype=float), b["nav"].to_numpy(dtype=float))
