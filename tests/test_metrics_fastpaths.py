"""metrics 快路径一致性门禁。

逐日 IC / 十分位分组等 metric 计算有两条路径：
- 快路径：datetime 连续区间切片（``_day_slices``）+ 快速等频分箱（``_fast_equal_freq_codes``）；
- 回落路径：groupby + ``xs`` + ``pd.qcut``（面板未排序 / 边界浮点重合时触发）。

本门禁用同一数据强制分别走两条路径，断言输出一致；并抽查快速分箱与
``pd.qcut`` 的等价性（连续数据走快路径，边界重合必须回落）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor import metrics as M


def _panel(n_days=120, n_inst=40, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-01", periods=n_days), [f"S{i:03d}" for i in range(n_inst)]],
        names=["datetime", "instrument"],
    )
    n = len(idx)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_inst)), axis=0)).ravel(order="F")
    c2 = close.reshape(n_days, n_inst)
    ret = np.zeros((n_days, n_inst))
    ret[1:] = c2[1:] / c2[:-1] - 1
    f = rng.normal(0, 1, n) + 0.2 * ret.ravel(order="F")
    label = pd.Series(ret.ravel(order="F") + rng.normal(0, 0.005, n), index=idx)
    factor = pd.Series(f, index=idx, name="f")
    # 注入 NaN
    mask = rng.random(n) < 0.03
    return factor[~mask], label[~mask], rng


def test_fast_equal_freq_codes_matches_qcut():
    rng = np.random.default_rng(0)
    used_fast = 0
    for trial in range(300):
        m = int(rng.integers(40, 500))
        K = int(rng.integers(2, 12))
        if trial % 3 == 0:
            x = rng.normal(size=m)
        elif trial % 3 == 1:
            x = rng.integers(0, max(m // 2, 5), size=m).astype(float)
        else:
            x = rng.normal(size=m) * np.where(rng.random(m) < 0.1, 1.0, 3.0)
        fast = M._fast_equal_freq_codes(x, K)
        ref = np.asarray(pd.qcut(pd.Series(x), K, labels=False, duplicates="drop"), dtype=float)
        if fast is None:
            continue  # 边界重合 → 回落，正确行为
        used_fast += 1
        assert np.array_equal(fast.astype(float), ref), f"trial {trial} 分箱不一致"
    assert used_fast > 100, "快路径覆盖率异常（几乎全在回落）"


def test_cross_sectional_ic_fast_vs_fallback():
    factor, label, _ = _panel()
    fast_ic = M.cross_sectional_ic(factor, label, min_pairs=5)
    fast_ric = M.cross_sectional_rank_ic(factor, label, min_pairs=5)
    with _patched_day_slices(None):
        old_ic = M.cross_sectional_ic(factor, label, min_pairs=5)
        old_ric = M.cross_sectional_rank_ic(factor, label, min_pairs=5)
    assert np.allclose(fast_ic.to_numpy(), old_ic.to_numpy(), equal_nan=True, atol=1e-12)
    assert np.allclose(fast_ric.to_numpy(), old_ric.to_numpy(), equal_nan=True, atol=1e-12)


def test_decile_and_quantile_portfolio_fast_vs_fallback():
    factor, label, rng = _panel(seed=9)
    means_fast = M._compute_daily_decile_mean_labels(factor, label, n_deciles=10, min_stocks=30)
    qp_fast = M.quantile_portfolio_metrics(factor, label, cost_bps=0.0)
    ls_fast = M.daily_long_short_series(factor, label)

    with _patched_day_slices(None), _patched_fast_codes(None):
        means_old = M._compute_daily_decile_mean_labels(factor, label, n_deciles=10, min_stocks=30)
        qp_old = M.quantile_portfolio_metrics(factor, label, cost_bps=0.0)
        ls_old = M.daily_long_short_series(factor, label)

    common = sorted(set(means_fast) & set(means_old))
    assert len(common) > 50
    for ts in common:
        a, b = means_fast[ts], means_old[ts]
        assert len(a) == len(b)
        assert np.allclose(a, b, equal_nan=True, atol=1e-12)
    for key in ("top_group_annualized_return", "top_group_sharpe", "monotonicity", "avg_daily_side_turnover"):
        a, b = qp_fast[key], qp_old[key]
        if isinstance(a, float) and np.isnan(a):
            assert np.isnan(b)
        else:
            assert abs(a - b) <= 1e-9 * max(1.0, abs(a)), f"{key}: {a} vs {b}"
    la, lb = ls_fast.align(ls_old, join="inner")
    assert np.allclose(la.to_numpy(), lb.to_numpy(), equal_nan=True, atol=1e-12)


def test_size_neutralize_fast_vs_fallback():
    factor, label, rng = _panel(seed=11)
    panel = pd.DataFrame(
        {"float_cap": rng.lognormal(9, 0.5, len(factor))},
        index=factor.index,
    )
    fast = M.cross_sectional_size_neutralize_values(factor.to_numpy(np.float64), panel)
    with _patched_day_slices(None):
        old = M.cross_sectional_size_neutralize_values(factor.to_numpy(np.float64), panel)
    assert np.allclose(fast, old, equal_nan=True, atol=1e-12)

import contextlib


@contextlib.contextmanager
def _patched_day_slices(value):
    """value=None 强制回落 groupby+xs 旧路径。"""
    old = M._day_slices_override
    M._day_slices_override = (lambda index, time_level="datetime": value) if value is None else value
    try:
        yield
    finally:
        M._day_slices_override = old


@contextlib.contextmanager
def _patched_fast_codes(value):
    old = M._fast_equal_freq_codes_override
    M._fast_equal_freq_codes_override = (lambda xf, n_groups: value) if value is None else value
    try:
        yield
    finally:
        M._fast_equal_freq_codes_override = old
