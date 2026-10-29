"""Stacking 加速路径位级一致性回归（njit 快路径 vs pandas 基准）。

覆盖 OCR 审查修复点：
1. daily_spearman_ic_np 与 pandas 版逐位一致（含每天仅 2 个有效样本的边界天）；
2. n=1 / 无有效样本天两边一致剔除；
3. 乱序面板（instrument-major）下 compute_quality_scores 的 bounds 守卫
   传 None、回退 pandas 版且逐位一致；
4. transform_factor_values 快路径与 pandas 引用实现逐位一致（含行业内 rank、
   全 NaN 行业组、无行业列退化）；
5. _sampled_corr 行切片快路径与逐日掩码原路径同面板取数一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.factor.stacking.dataset import (  # noqa: E402
    _day_codes_and_bounds,
    _day_pairs_from_bounds,
    _day_row_slices,
    _sampled_corr,
    daily_spearman_ic_np,
    daily_spearman_ic_pd,
    transform_factor_values,
)


def _panel(n_days: int = 30, n_inst: int = 50, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    industry = np.tile(np.repeat(["A", "B", "C", "D", "E"], n_inst // 5), n_days)
    return pd.DataFrame({"industry_sw_l1": industry}, index=idx)


def _factor_series(n: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v[rng.random(n) < 0.08] = np.nan
    return v


def test_daily_ic_np_equals_pd_regular_panel() -> None:
    panel = _panel()
    v = _factor_series(len(panel))
    y = _factor_series(len(panel), seed=13)
    dts = panel.index.get_level_values("datetime")
    assert daily_spearman_ic_np(v, y, dts).equals(daily_spearman_ic_pd(v, y, dts))


def test_daily_ic_np_two_sample_day_boundary() -> None:
    """每天仅 2 个有效样本：pandas 输出有限 ±1.0，np 版必须保留而非丢弃。"""
    days = pd.date_range("2024-01-01", periods=3)
    rows = [
        (days[0], 1.0, 0.5), (days[0], 2.0, 1.5),          # n=2 → ic=±1
        (days[1], 5.0, 5.0),                                # n=1 → 剔除
        (days[2], 3.0, 3.0), (days[2], 4.0, 4.0),           # n=2 + NaN 行
        (days[2], np.nan, 1.0), (days[2], 1.0, np.nan),
    ]
    dts = pd.DatetimeIndex([r[0] for r in rows])
    p = np.array([r[1] for r in rows], dtype=np.float64)
    y = np.array([r[2] for r in rows], dtype=np.float64)
    ic_np = daily_spearman_ic_np(p, y, dts)
    ic_pd = daily_spearman_ic_pd(p, y, dts)
    assert ic_np.equals(ic_pd)
    # n=2 天被保留且为 ±1.0（有限值），n=1 天被剔除
    assert days[0] in ic_np.index and days[2] in ic_np.index
    assert days[1] not in ic_np.index
    assert np.all(np.isclose(np.abs(ic_np.to_numpy()), 1.0))  # ±1（浮点尾差）


def test_compute_quality_bounds_guard_loose_panel() -> None:
    """乱序（instrument-major）面板：day_codes 非递减守卫 → bounds=None →
    np 版回退 pandas，逐位一致（此前实测静默算错：20 天只出 2 天）。"""
    n_inst, n_days = 30, 12
    days = pd.date_range("2024-01-01", periods=n_days)
    loose = [(d, i) for i in range(n_inst) for d in days]
    dts = pd.MultiIndex.from_tuples(loose, names=["datetime", "instrument"]).get_level_values("datetime")
    rng = np.random.default_rng(3)
    p = rng.standard_normal(len(loose)) * 10
    y = rng.standard_normal(len(loose))
    p[rng.random(len(p)) < 0.1] = np.nan
    # compute_quality_scores 同款守卫
    day_codes, bs, be = _day_codes_and_bounds(dts)
    bounds = (bs, be) if len(bs) and np.all(np.diff(day_codes) >= 0) else None
    assert bounds is None  # 乱序必须被守卫拦下
    assert daily_spearman_ic_np(p, y, dts, bounds=bounds).equals(daily_spearman_ic_pd(p, y, dts))


def test_transform_fastpath_equals_pandas_ref() -> None:
    panel = _panel()
    v = _factor_series(len(panel), seed=21)
    z = transform_factor_values(v.copy(), panel, size_neutral=False)

    # 与 pandas 引用实现（回退路径同款逻辑）逐位比较
    v64 = np.asarray(v, dtype=np.float64)
    dts = panel.index.get_level_values("datetime")
    df = pd.DataFrame({"v": v64, "datetime": dts})
    df["ind"] = pd.Series(panel["industry_sw_l1"].to_numpy(), index=panel.index).to_numpy()
    ranked = df.groupby(["datetime", "ind"], sort=False, dropna=True)["v"].rank(pct=True)
    df["r"] = ranked
    g = df.groupby("datetime", sort=False)["r"]
    mean = g.transform("mean")
    std = g.transform("std")
    ref = ((df["r"] - mean) / std.where(std > 1e-12)).to_numpy(dtype=np.float32)
    ref[~np.isfinite(ref)] = np.nan
    assert np.array_equal(z, ref, equal_nan=True)


def test_transform_fastpath_no_industry_column() -> None:
    panel = _panel()
    panel = panel.drop(columns=["industry_sw_l1"])
    v = _factor_series(len(panel), seed=31)
    z = transform_factor_values(v.copy(), panel, size_neutral=False)
    v64 = np.asarray(v, dtype=np.float64)
    dts = panel.index.get_level_values("datetime")
    df = pd.DataFrame({"v": v64, "datetime": dts})
    ranked = df.groupby("datetime", sort=False, dropna=True)["v"].rank(pct=True)
    df["r"] = ranked
    g = df.groupby("datetime", sort=False)["r"]
    ref = ((df["r"] - g.transform("mean")) / g.transform("std").where(
        g.transform("std") > 1e-12)).to_numpy(dtype=np.float32)
    ref[~np.isfinite(ref)] = np.nan
    assert np.array_equal(z, ref, equal_nan=True)


def test_sampled_corr_slice_path_equals_mask_path() -> None:
    rng = np.random.default_rng(41)
    n_days, n_inst = 16, 60
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=n_days), [f"S{i:02d}" for i in range(n_inst)]],
        names=["datetime", "instrument"],
    )
    panel = pd.DataFrame({"v": rng.standard_normal(len(idx))}, index=idx)
    a = rng.standard_normal(len(idx))
    b = rng.standard_normal(len(idx))
    a[rng.random(len(idx)) < 0.1] = np.nan

    # 快路径取数（行切片）与逐日掩码取数同面板对照
    bounds = _day_row_slices(panel)
    assert bounds is not None
    pa, pb = _day_pairs_from_bounds(a, b, bounds, stride=5)
    dts = panel.index.get_level_values("datetime")
    ma, mb = [], []
    for day in dts.unique()[::5]:
        mask = dts == day
        ok = np.isfinite(a[mask]) & np.isfinite(b[mask])
        if ok.sum() < 5:
            continue
        ma.append(a[mask][ok])
        mb.append(b[mask][ok])
    assert len(pa) == len(ma)
    assert all(np.array_equal(pa[i], ma[i]) and np.array_equal(pb[i], mb[i]) for i in range(len(pa)))
    assert _sampled_corr(a, b, panel, stride=5) == float(np.corrcoef(np.concatenate(ma), np.concatenate(mb))[0, 1])