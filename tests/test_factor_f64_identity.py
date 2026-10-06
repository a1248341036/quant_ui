# -*- coding: utf-8 -*-
"""factor_f64 缓存身份契约回归（2026-10-06 OCR critical C-1）。

旧键 ``(id(factor.index), str(factor.name))`` 在本用例的条件下必然串台：两个 Series
共享同一 index 对象、都不带 name（submit 路径的 ``metric_series = pd.Series(values,
index=panel.index)`` 正是如此，``str(None) == "None"``）→ 第 2 次调用命中第 1 个
因子的数组。后果是组合指标的换手/收益/深度曲线全用上一个因子的值算，而换手恰是
stage_one 硬门的输入（误判通过/拒绝）。

回归点：键必须是 factor 的**对象身份**（``id(factor)`` + weakref 校验），
同一 Series 对象仍命中缓存，不同 Series 一律算各自的数组。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alphaagent.factor.metrics._factor_cache as factor_cache
from alphaagent.factor.metrics import quantile_portfolio_metrics


@pytest.fixture(autouse=True)
def _clean_factor_cache():
    factor_cache._cache.clear()
    factor_cache._factor_f64_override = None
    yield
    factor_cache._cache.clear()
    factor_cache._factor_f64_override = None


def _index(n_days: int = 6, n_inst: int = 80) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-01", periods=n_days, freq="D"),
            [f"S{i:03d}" for i in range(n_inst)],
        ],
        names=["datetime", "instrument"],
    )


def _factor(index: pd.MultiIndex, seed: int) -> pd.Series:
    """submit 路径同款：float32 + index=panel.index + 不带 name。"""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=len(index)).astype(np.float32), index=index)


def test_shared_index_and_absent_name_do_not_share_values():
    """旧键的成立条件（同 index 对象 + 同 name）下，两个因子必须各返回自己的数组。"""
    index = _index(n_days=2, n_inst=5)
    a = _factor(index, 1)
    b = _factor(index, 2)
    assert a.index is b.index
    assert a.name is None and b.name is None

    arr_a = factor_cache.factor_f64(a)
    arr_b = factor_cache.factor_f64(b)

    np.testing.assert_allclose(arr_a, a.to_numpy(dtype=np.float64))
    np.testing.assert_allclose(arr_b, b.to_numpy(dtype=np.float64))
    assert not np.allclose(arr_a, arr_b)
    assert not arr_b.flags.writeable  # 共享数组必须只读


def test_same_series_object_still_hits_cache():
    """同一次 submit 的全窗口 + tradable lens 两次调用传同一对象 → 仍命中（省一次拷贝）。"""
    index = _index(n_days=2, n_inst=5)
    s = _factor(index, 3)
    assert factor_cache.factor_f64(s) is factor_cache.factor_f64(s)


def test_quantile_portfolio_not_contaminated_across_factors():
    """端到端：先算 A 再算 B，B 的组合指标必须等于「绕过缓存」的 B。"""
    index = _index()
    rng = np.random.default_rng(9)
    label = pd.Series(rng.normal(0, 0.02, len(index)), index=index)
    factor_a = _factor(index, 11)
    factor_b = _factor(index, 12)
    kw = dict(n_groups=10, cost_bps=0.0, holding_days=1, depth_ks=(5, 10, 20, 50, 100))

    metrics_a = quantile_portfolio_metrics(factor_a, label, **kw)
    metrics_b_cached = quantile_portfolio_metrics(factor_b, label, **kw)  # 旧实现此处被 A 污染

    factor_cache._factor_f64_override = lambda s: np.asarray(
        s.to_numpy(dtype=np.float64, copy=True)
    )
    fresh_b = quantile_portfolio_metrics(factor_b, label, **kw)

    turnover_a = metrics_a["avg_daily_side_turnover"]
    turnover_b = fresh_b["avg_daily_side_turnover"]
    # 夹具自检：A/B 换来确实可区分，否则本用例证明不了"串台已修"
    assert turnover_a != pytest.approx(turnover_b)
    assert metrics_b_cached["avg_daily_side_turnover"] == pytest.approx(turnover_b)
    assert metrics_b_cached["top_group_annualized_return"] == pytest.approx(
        fresh_b["top_group_annualized_return"]
    )
    assert metrics_b_cached["top_group_sharpe"] == pytest.approx(fresh_b["top_group_sharpe"])
