"""DSL 慢算子一致性门禁：boundaries 并行快路径 vs 旧逐品种路径数值一致。

每个用例在同一输入上分别跑「快路径」（稳定归组 + Numba 并行内核）与「旧路径」
（monkeypatch ``_boundaries_fast`` 返回 None 强制回落逐品种 groupby / 纯 Python），
断言逐位一致（float32 相等、NaN 位置一致）。覆盖 NaN 注入、跳空边界、零成交量、
乱序面板、动态窗、多 method/implementation 组合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.dsl.core import operators as ops

pytest.importorskip("numba")

N_INST = 40
N_DAYS = 300
TOL = 1e-6


def _build_panels(seed: int = 7, perm_seed: int = 11):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=N_DAYS)
    inst = [f"S{i:03d}" for i in range(N_INST)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    n = len(idx)

    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (N_DAYS, N_INST)), axis=0)).ravel(order="F")
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    volume = rng.lognormal(10, 1, n)
    float_cap = rng.lognormal(9, 0.5, n)

    nan_mask = rng.random(n) < 0.03
    for arr in (close, open_, high, low, volume, float_cap):
        arr[nan_mask] = np.nan
    hi2 = high.reshape(N_DAYS, N_INST)
    lo2 = low.reshape(N_DAYS, N_INST)
    jump = rng.choice(np.arange(1, N_DAYS), size=20, replace=False)
    lo2[jump[:10]] *= 1.10   # 向上跳空
    hi2[jump[10:]] *= 0.90   # 向下跳空
    volume.reshape(N_DAYS, N_INST)[rng.random((N_DAYS, N_INST)) < 0.05] = 0.0

    def P(a, order=None):
        return pd.DataFrame(a if order is None else a[order], index=idx if order is None else idx[order])

    panels = {k: P(v) for k, v in
              (("open", open_), ("high", high), ("low", low), ("close", close),
               ("volume", volume), ("float_cap", float_cap))}
    perm = rng.permutation(n)
    panels_shuffled = {k: P(v.to_numpy()[:, 0], perm) for k, v in panels.items()}
    return panels, panels_shuffled


@pytest.fixture(scope="module")
def panels():
    return _build_panels()[0]


@pytest.fixture(scope="module")
def panels_shuffled():
    return _build_panels()[1]


_REAL_BOUNDARIES_FAST = ops._boundaries_fast


def _compare(name, call):
    """同一输入上先跑旧路径（回落）再跑新路径（快路径），断言逐位一致。"""
    ops._boundaries_fast = lambda *a, **k: None
    try:
        old_out = call().iloc[:, 0].to_numpy(dtype=np.float64)
    finally:
        ops._boundaries_fast = _REAL_BOUNDARIES_FAST
    new_out = call().iloc[:, 0].to_numpy(dtype=np.float64)

    both_nan = np.isnan(old_out) & np.isnan(new_out)
    same = (old_out == new_out) | both_nan  # 相同值（含 ±inf）视为一致
    with np.errstate(invalid="ignore"):
        diff = np.where(same, 0.0, np.abs(old_out - new_out))
    nan_mismatch = int(np.sum(np.isnan(old_out) != np.isnan(new_out)))
    max_diff = float(np.max(diff)) if np.isfinite(diff).any() else 0.0
    assert nan_mismatch == 0, f"{name}: NaN 位置不一致（{nan_mismatch} 处）"
    assert max_diff <= TOL, f"{name}: 最大偏差 {max_diff:.3e} 超容差 {TOL}"


PRICE_GAP_FIELDS = [
    ("PRICE_GAP_SIZE", ops.PRICE_GAP_SIZE),
    ("PRICE_GAP_FILL", ops.PRICE_GAP_FILL),
    ("PRICE_GAP_FLOOR", ops.PRICE_GAP_FLOOR),
    ("PRICE_GAP_CEILING", ops.PRICE_GAP_CEILING),
    ("PRICE_GAP_EVENT", ops.PRICE_GAP_EVENT),
    ("PRICE_GAP_BARS", ops.TS_LAST_ARGGAP),
]


@pytest.mark.parametrize("name,fn", PRICE_GAP_FIELDS)
def test_price_gap_fields(panels, name, fn):
    o, h, l, c = (panels[k] for k in ("open", "high", "low", "close"))
    _compare(name, lambda: fn(o, h, l, c, 0.0))


def test_price_gap_min_pct(panels):
    o, h, l, c = (panels[k] for k in ("open", "high", "low", "close"))
    _compare("PRICE_GAP_FILL(min_pct=0.005)", lambda: ops.PRICE_GAP_FILL(o, h, l, c, 0.005))


CHIP_CASES = [
    ("CHIP_PEAK_LOC", lambda p: ops.CHIP_PEAK_LOC(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"])),
    ("CHIP_ENTROPY", lambda p: ops.CHIP_ENTROPY(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"])),
    ("CHIP_COM_W_GAP", lambda p: ops.CHIP_COM_W_GAP(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"])),
    ("CHIP_MASS_ASYM", lambda p: ops.CHIP_MASS_ASYM(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"])),
    ("CHIP_PEAK_SHARPNESS/curv", lambda p: ops.CHIP_PEAK_SHARPNESS(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, "curvature")),
    ("CHIP_PEAK_SHARPNESS/fwhm", lambda p: ops.CHIP_PEAK_SHARPNESS(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, "fwhm")),
    ("CHIP_PEAK_SHARPNESS/combined", lambda p: ops.CHIP_PEAK_SHARPNESS(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, "combined")),
    ("CHIP_BIMODAL/simple", lambda p: ops.CHIP_BIMODAL_SCORE(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, "simple")),
    ("CHIP_BIMODAL/dip", lambda p: ops.CHIP_BIMODAL_SCORE(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, "dip")),
    ("CHIP_WASS_DIST/moment", lambda p: ops.CHIP_WASS_DIST(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, 10, "moment")),
    ("CHIP_WASS_DIST/transport", lambda p: ops.CHIP_WASS_DIST(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, 10, "transport")),
    ("CHIP_ENTROPY/uniform", lambda p: ops.CHIP_ENTROPY(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, "uniform")),
    ("CHIP_ENTROPY/tri", lambda p: ops.CHIP_ENTROPY(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"], 32, "tri")),
]


@pytest.mark.parametrize("name,fn", CHIP_CASES)
def test_chip_family(panels, name, fn):
    _compare(name, lambda: fn(panels))


def test_chip_wass_dynamic_window(panels):
    rng = np.random.default_rng(3)
    dyn_w = pd.DataFrame(
        rng.integers(20, 80, len(panels["close"])).astype(float),
        index=panels["close"].index,
    )
    _compare(
        "CHIP_WASS_DIST/动态窗",
        lambda: ops.CHIP_WASS_DIST(panels["close"], panels["low"], panels["high"],
                                   panels["volume"], dyn_w, panels["float_cap"], 32, 10, "moment"),
    )


OTHER_CASES = [
    ("WICK_EFFICIENCY", lambda p: ops.WICK_EFFICIENCY(p["open"], p["high"], p["low"], p["close"], 3)),
    ("CROWD_SHARE/high", lambda p: ops.CROWD_SHARE(p["close"], p["volume"], 20, "high", 0.9)),
    ("CROWD_SHARE/low", lambda p: ops.CROWD_SHARE(p["close"], p["volume"], 20, "low", 0.7)),
    ("CROWD_SHARE/equal_freq", lambda p: ops.CROWD_SHARE(p["close"], p["volume"], 20, 5, 3)),
    ("CROWD_MEAN_RATIO", lambda p: ops.CROWD_MEAN_RATIO(p["close"], p["float_cap"], 20, "high", 0.8)),
    ("CROWD_CONTRAST", lambda p: ops.CROWD_CONTRAST(p["close"], p["float_cap"], 20, 0.6)),
    ("CROWD_RANK_WEIGHTED", lambda p: ops.CROWD_RANK_WEIGHTED(p["close"], p["float_cap"], 20, p["volume"])),
    ("VOLUME_CLOCK_VPIN/tick", lambda p: ops.VOLUME_CLOCK_VPIN(p["close"], p["volume"], 5, 5e5)),
    ("VOLUME_CLOCK_VPIN/lee_ready", lambda p: ops.VOLUME_CLOCK_VPIN(p["close"], p["volume"], 5, 5e5, "lee_ready")),
    ("MUTUAL_INFO_LAG/lag1", lambda p: ops.MUTUAL_INFO_LAG(p["close"], p["volume"], 30, 1)),
    ("MUTUAL_INFO_LAG/lag4", lambda p: ops.MUTUAL_INFO_LAG(p["close"], p["volume"], 30, 4, n_bins=6)),
]


@pytest.mark.parametrize("name,fn", OTHER_CASES)
def test_other_operators(panels, name, fn):
    _compare(name, lambda: fn(panels))


UNARY_CASES = [
    ("DELTA", lambda p: ops.DELTA(p["close"], 5)),
    ("TS_PCTCHANGE", lambda p: ops.TS_PCTCHANGE(p["close"], 3)),
    ("DELAY", lambda p: ops.DELAY(p["close"], 4)),
    ("EMA", lambda p: ops.EMA(p["close"], 10)),
    ("WMA", lambda p: ops.WMA(p["close"], 10)),
    ("SMA", lambda p: ops.SMA(p["close"], 8)),
    ("TS_CUMPROD", lambda p: ops.TS_CUMPROD(p["close"], 100.0)),
    ("TS_ZSCORE", lambda p: ops.TS_ZSCORE(p["close"], 20)),
    ("TS_QUANTILE", lambda p: ops.TS_QUANTILE(p["close"], 20, 0.25)),
    ("TS_ARGMEDIAN", lambda p: ops.TS_ARGMEDIAN(p["close"], 15)),
    ("TS_ARGNTH", lambda p: ops.TS_ARGNTH(p["close"], 15, 2)),
    ("TS_PERMUTATION_ENTROPY", lambda p: ops.TS_PERMUTATION_ENTROPY(p["close"], 30, 3)),
    ("TS_LAST_ARGPEAK", lambda p: ops.TS_LAST_ARGPEAK(p["close"], 5)),
    ("TS_LAST_PEAK", lambda p: ops.TS_LAST_PEAK(p["close"], 5)),
    ("TS_AMPARGTROUGH", lambda p: ops.TS_AMPARGTROUGH(p["close"], 5)),
    ("TS_AMPTROUGH", lambda p: ops.TS_AMPTROUGH(p["close"], 5)),
    ("TS_EFFICIENCY_RATIO", lambda p: ops.TS_EFFICIENCY_RATIO(p["close"], 20)),
    ("TS_CORR", lambda p: ops.TS_CORR(p["close"], p["volume"], 20)),
    ("TS_COV", lambda p: ops.TS_COV(p["close"], p["volume"], 20)),
    ("TS_RANKCORR", lambda p: ops.TS_RANKCORR(p["close"], p["volume"], 20)),
    ("TS_TREND_RANK", lambda p: ops.TS_TREND_RANK(p["close"], 20)),
]


@pytest.mark.parametrize("name,fn", UNARY_CASES)
def test_unary_and_corr_family(panels, name, fn):
    _compare(name, lambda: fn(panels))


@pytest.mark.parametrize("name,fn", [
    ("CHIP_ENTROPY", lambda p: ops.CHIP_ENTROPY(p["close"], p["low"], p["high"], p["volume"], 60, p["float_cap"])),
    ("PRICE_GAP_FILL", lambda p: ops.PRICE_GAP_FILL(p["open"], p["high"], p["low"], p["close"])),
])
def test_shuffled_panel_old_vs_new(panels_shuffled, name, fn):
    """乱序面板上新旧路径必须一致（不依赖面板行序的归组机制）。"""
    _compare(f"{name}/乱序面板", lambda: fn(panels_shuffled))
