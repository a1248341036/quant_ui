"""穷举子集组合扫描的快速逐日 IC 与 dataset.daily_spearman_ic 对拍。

chunk_daily_spearman_ic 用序数秩（NaN 末位排序）做分块向量化，语义与
authoritative 实现一致（dropna 后双方在联合有效行上取秩再 Pearson）。
生产输入是连续 rank-zscore 分数与连续收益标签（并列概率为零），无并列时
两者必须逐位一致；并列场景不在对拍范围（序数秩 vs 平均秩有理论差异）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from scripts.scan_factor_subsets import (
    chunk_daily_spearman_ic,
    combo_horizon,
    enumerate_subsets,
    summarize_ics,
)
from alphaagent.factor.stacking.dataset import daily_spearman_ic


def _make_rows(n_days: int, n_stocks: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    dts = np.repeat(dates.to_numpy(), n_stocks)  # 行序按日期分块
    label = rng.normal(0, 0.02, n_days * n_stocks)
    return dts, label


def _day_bounds(dts) -> list[tuple[int, int]]:
    d = pd.Series(dts).to_numpy()
    changes = np.flatnonzero(d[1:] != d[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(d)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def test_chunk_ic_matches_daily_spearman_ic() -> None:
    rng = np.random.default_rng(11)
    n_days, n_stocks, n_factors = 40, 200, 5
    dts, label = _make_rows(n_days, n_stocks)
    factors = rng.normal(0, 1, (n_days * n_stocks, n_factors))
    combos = [(0, 1), (0, 2, 3), (1, 3, 4), (2, 4), (0, 1, 2, 3, 4)]
    S = np.column_stack([factors[:, list(c)].mean(axis=1) for c in combos])

    fast = chunk_daily_spearman_ic(S, label, _day_bounds(dts))
    for ci, c in enumerate(combos):
        ref = daily_spearman_ic(S[:, ci], label, pd.Series(dts)).to_numpy()
        np.testing.assert_allclose(fast[:, ci], ref, atol=1e-10)
        m_fast = summarize_ics(fast[:, ci])
        assert m_fast["n_days"] == len(ref)
        assert m_fast["ic_mean"] == pytest.approx(float(ref.mean()), abs=1e-10)
        assert m_fast["ic_ir"] == pytest.approx(float(ref.mean() / ref.std(ddof=1)), abs=1e-10)


def test_chunk_ic_nan_mask_matches_dropna_reference() -> None:
    rng = np.random.default_rng(13)
    n_days, n_stocks = 30, 150
    dts, label = _make_rows(n_days, n_stocks)
    a = rng.normal(0, 1, n_days * n_stocks)
    b = rng.normal(0, 1, n_days * n_stocks)
    a[rng.random(a.size) < 0.1] = np.nan  # 因子覆盖不完全
    b[rng.random(b.size) < 0.1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        score = np.nanmean(np.vstack([a, b]), axis=0)

    fast = chunk_daily_spearman_ic(score[:, None], label, _day_bounds(dts))[:, 0]
    ref = daily_spearman_ic(score, label, pd.Series(dts)).to_numpy()  # dropna 语义
    np.testing.assert_allclose(fast, ref, atol=1e-10)


def test_chunk_ic_label_nan_also_dropped() -> None:
    # 标签 NaN（如前向收益尾部）与分数 NaN 的联合掩码语义
    rng = np.random.default_rng(19)
    n_days, n_stocks = 20, 100
    dts, label = _make_rows(n_days, n_stocks)
    score = rng.normal(0, 1, n_days * n_stocks)
    score[rng.random(score.size) < 0.05] = np.nan
    label[rng.random(label.size) < 0.05] = np.nan
    fast = chunk_daily_spearman_ic(score[:, None], label, _day_bounds(dts))[:, 0]
    ref = daily_spearman_ic(score, label, pd.Series(dts)).to_numpy()
    np.testing.assert_allclose(fast, ref, atol=1e-10)


def test_chunk_ic_small_days_marked_nan() -> None:
    rng = np.random.default_rng(17)
    n_days, n_stocks = 10, 20  # 每日 20 行 < MIN_COMBOS_PER_DAY(30)
    dts, label = _make_rows(n_days, n_stocks)
    score = rng.normal(0, 1, n_days * n_stocks)
    out = chunk_daily_spearman_ic(score[:, None], label, _day_bounds(dts))
    assert np.isnan(out).all()


def test_enumerate_subsets_and_horizon() -> None:
    subs = enumerate_subsets(4, 2, 4)
    assert len(subs) == 11  # C(4,2)+C(4,3)+C(4,4)
    assert all(len(s) >= 2 for s in subs)
    with pytest.raises(ValueError):
        enumerate_subsets(4, 1, 4)
    assert combo_horizon([None, 20, None], 5) == 20
    assert combo_horizon([1, 20], 5) == 10  # 中位数
    assert combo_horizon([None, None], 5) == 5  # 回退
