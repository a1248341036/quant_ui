"""DSL 行业/轮动算子门禁：CS_GROUP_* 族 + TS_AUTOCORR + TS_RSQ + SCALE。

覆盖：
- 语义：CS_GROUP_MEAN/SUM/MEDIAN/STD/ZSCORE/COUNT 与逐 (datetime, group)
  手写 pandas 聚合逐位一致（assert_allclose(equal_nan=True)）；NaN/+/-(inf)/单样本组/
  常数组/ddof 越界边界；SCALE 与 ``x * (a / sum|x|)`` 一致。
- 一致性：Numba 快路径 vs pandas 回落路径（ALPHA_DSL_CS_ACCEL=0）逐位一致；
  乱序面板自动回落。
- 糖封装：TS_AUTOCORR(x,w,lag) == TS_CORR(x, DELAY(x,lag), w)。
- TS_RSQ：与逐品种滚动 lstsq R2 一致（p=3/4/5）；短窗 p=1 全 NaN、p=0 拒绝。
- 目录可见性：operator_catalog_markdown(tier="full") 含全部新签名。
- 守卫：TS_AUTOCORR/TS_RSQ 负窗被拦；SCALE 负 a 豁免。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.dsl.core import operators as ops
from alphaagent.dsl.registry import build_operator_namespace
from alphaagent.dsl.core.guard import wrap_lookahead_guard

pytest.importorskip("numba")

TOL = 1e-4
N_INST = 40
N_DAYS = 60


def _build(n_inst: int = N_INST, n_days: int = N_DAYS, seed: int = 11, shuffle: bool = False):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    n = len(idx)

    x = rng.normal(size=n)
    g = (rng.integers(8, size=n)).astype(float)
    g[n // 4 :] = g[n // 4 :] % 10              # 后段出现 8/9 号组
    x[::13] = np.nan
    g[1::17] = np.nan
    extreme = rng.random(n) < 0.004
    x[extreme] = np.where(rng.random(int(extreme.sum())) < 0.5, np.inf, -np.inf)

    if shuffle:
        perm = rng.permutation(n)
        x, g = x[perm], g[perm]
        idx = idx[perm]
    return {
        "x": pd.DataFrame(x, index=idx, columns=["x"]),
        "g": pd.DataFrame(g, index=idx, columns=["g"]),
    }


@pytest.fixture(scope="module")
def panels():
    return _build()


@pytest.fixture(scope="module")
def tiny_panels():
    """窄面板：单样本组、组内全部 NaN、常数组必然出现。"""
    return _build(n_inst=4, n_days=20, seed=5)


def _group_ref(xv: np.ndarray, gv: np.ndarray, days, kind: str, ddof: int = 1) -> np.ndarray:
    """手写 (datetime, group) 聚合参考。

    语义与算子对齐：只有 ``g`` 有限且 ``x`` 有限的行参与；COUNT 只要求 ``g`` 有限。
    MEAN/SUM/MEDIAN 组内有效样本 >= 1；STD/ZSCORE 要求 n>ddof 且 std 有限非零。
    """
    out = np.full(len(xv), np.nan)
    if kind == "count":
        v = np.isfinite(gv)
        ser = pd.DataFrame({"day": days, "g": gv, "x": gv})[v]
        agg = ser.groupby(["day", "g"])["x"].transform("size")
        out[v] = agg.to_numpy()
        return out
    v = np.isfinite(gv) & np.isfinite(xv)
    df = pd.DataFrame({"day": days, "g": gv, "x": xv})[v]
    if kind == "std":
        agg = df.groupby(["day", "g"])["x"].transform(lambda s: s.std(ddof=ddof))
        agg = agg.where(np.isfinite(agg) & (agg != 0.0))
    elif kind == "median":
        agg = df.groupby(["day", "g"])["x"].transform("median")
    elif kind == "zscore":
        m = df.groupby(["day", "g"])["x"].transform("mean")
        s = df.groupby(["day", "g"])["x"].transform(lambda s: s.std(ddof=ddof))
        s_arr = s.to_numpy()
        agg = np.where(np.isfinite(s_arr) & (s_arr != 0.0),
                       (df["x"].to_numpy() - m.to_numpy()) / s_arr, np.nan)
    else:  # mean / sum
        agg = df.groupby(["day", "g"])["x"].transform(kind)
    out[v] = agg if isinstance(agg, np.ndarray) else agg.to_numpy()
    return out


def _assert_close(name: str, got: np.ndarray, ref: np.ndarray) -> None:
    nan_mismatch = int(np.sum(np.isnan(got) != np.isnan(ref)))
    assert nan_mismatch == 0, f"{name}: NaN 位置不一致（{nan_mismatch} 处）"
    np.testing.assert_allclose(
        got, ref, rtol=TOL, atol=TOL, equal_nan=True, err_msg=name
    )


CASES = [
    ("CS_GROUP_MEAN", "mean", lambda x, g, ddof: ops.CS_GROUP_MEAN(x, g)),
    ("CS_GROUP_SUM", "sum", lambda x, g, ddof: ops.CS_GROUP_SUM(x, g)),
    ("CS_GROUP_MEDIAN", "median", lambda x, g, ddof: ops.CS_GROUP_MEDIAN(x, g)),
    ("CS_GROUP_STD", "std", lambda x, g, ddof: ops.CS_GROUP_STD(x, g)),
    ("CS_GROUP_STD(ddof=0)", "std", lambda x, g, ddof: ops.CS_GROUP_STD(x, g, ddof=0)),
    ("CS_GROUP_ZSCORE", "zscore", lambda x, g, ddof: ops.CS_GROUP_ZSCORE(x, g)),
]


@pytest.mark.parametrize("name,kind,fn", CASES)
def test_group_semantics(panels, tiny_panels, name, kind, fn):
    ddof = 0 if "ddof=0" in name else 1
    xv = panels["x"].iloc[:, 0].to_numpy(dtype=float)
    gv = panels["g"].iloc[:, 0].to_numpy(dtype=float)
    days = panels["x"].index.get_level_values("datetime")
    got = fn(panels["x"], panels["g"], ddof).iloc[:, 0].to_numpy(dtype=np.float64)
    _assert_close(name, got, _group_ref(xv, gv, days, kind, ddof))

    xv_t = tiny_panels["x"].iloc[:, 0].to_numpy(dtype=float)
    gv_t = tiny_panels["g"].iloc[:, 0].to_numpy(dtype=float)
    days_t = tiny_panels["x"].index.get_level_values("datetime")
    got_t = fn(tiny_panels["x"], tiny_panels["g"], ddof).iloc[:, 0].to_numpy(dtype=np.float64)
    _assert_close(name + "/tiny", got_t, _group_ref(xv_t, gv_t, days_t, kind, ddof))


def test_group_count(panels):
    gv = panels["g"].iloc[:, 0].to_numpy(dtype=float)
    days = panels["g"].index.get_level_values("datetime")
    got = ops.CS_GROUP_COUNT(panels["g"]).iloc[:, 0].to_numpy(dtype=np.float64)
    _assert_close("CS_GROUP_COUNT", got, _group_ref(gv, gv, days, "count", 1))


def test_group_single_member_and_constant():
    """单样本组：MEAN/SUM/MEDIAN=自身；STD/ZSCORE=NaN。常数组：STD/ZSCORE=NaN。"""
    days = pd.bdate_range("2021-01-04", periods=3)
    inst = ["S0", "S1", "S2", "S3"]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    xv = np.array([1.0, 2.0, 2.0, 5.0] * 3)         # S1/S2 常数组
    gv = np.array([0.0, 1.0, 1.0, 2.0] * 3)         # S1/S2 同组常量
    x = pd.DataFrame(xv, index=idx, columns=["x"])
    g = pd.DataFrame(gv, index=idx, columns=["g"])
    assert np.all(ops.CS_GROUP_MEAN(x, g).iloc[:, 0].to_numpy() == xv)
    sum_expected = np.array([1.0, 4.0, 4.0, 5.0] * 3)  # 常数组 S1/S2 求和为 4
    assert np.all(ops.CS_GROUP_SUM(x, g).iloc[:, 0].to_numpy() == sum_expected)
    assert np.all(ops.CS_GROUP_MEDIAN(x, g).iloc[:, 0].to_numpy() == xv)
    std = ops.CS_GROUP_STD(x, g).iloc[:, 0].to_numpy()
    zsc = ops.CS_GROUP_ZSCORE(x, g).iloc[:, 0].to_numpy()
    # 单例组 S0/S3 与常数组 S1/S2：ddof 越界或 std=0 → 全部 NaN
    assert np.isnan(std).all()
    assert np.isnan(zsc).all()


def test_scale_semantics(panels):
    xv = panels["x"].iloc[:, 0].to_numpy(dtype=float)
    days = panels["x"].index.get_level_values("datetime").unique()
    n_inst = len(panels["x"].index.get_level_values("instrument").unique())
    for a in (1.0, -2.5):
        got = ops.SCALE(panels["x"], a).iloc[:, 0].to_numpy(dtype=np.float64)
        ref = np.full(len(xv), np.nan)
        for d in range(len(days)):
            st, en = d * n_inst, (d + 1) * n_inst
            v = xv[st:en]
            fin = np.isfinite(v)
            if not fin.any():
                continue
            s = float(np.abs(v[fin]).sum())
            if s <= 0.0 or not np.isfinite(s):
                continue
            ref[st:en] = np.where(fin, v * (a / s), np.nan)
        _assert_close(f"SCALE(a={a})", got, ref)


def test_ts_autocorr_sugar(panels):
    for w, lag in ((5, 1), (6, 2)):
        auto = ops.TS_AUTOCORR(panels["x"], w, lag).iloc[:, 0].to_numpy(dtype=np.float64)
        ref = ops.TS_CORR(
            panels["x"], ops.DELAY(panels["x"], lag), w
        ).iloc[:, 0].to_numpy(dtype=np.float64)
        nan_mismatch = int(np.sum(np.isnan(auto) != np.isnan(ref)))
        assert nan_mismatch == 0, f"TS_AUTOCORR(w={w},lag={lag}): NaN 不一致（{nan_mismatch}）"
        np.testing.assert_allclose(auto, ref, rtol=1e-6, atol=1e-6, equal_nan=True)


def _roll_rsq_ref_series(y: pd.Series, p: int) -> np.ndarray:
    out = np.full(len(y), np.nan)
    xx = np.arange(1.0, p + 1.0)
    for i in range(p - 1, len(y)):
        win = y.iloc[i - p + 1 : i + 1].to_numpy(dtype=float)
        v = ~np.isnan(win)  # 与 _roll_rsq_numba 的 notna 口径一致
        if v.sum() < 2:
            continue
        yv, xv = win[v], xx[v]
        with np.errstate(invalid="ignore", over="ignore"):
            X = np.column_stack([xv, np.ones(len(xv))])
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            pred = beta[0] * xv + beta[1]
            ss_res = float(np.sum((yv - pred) ** 2))
            ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        if ss_tot <= 0.0 or not np.isfinite(ss_tot):
            continue
        out[i] = 1.0 - ss_res / ss_tot
    return out


def test_ts_rsq_matches_rolling_lstsq(panels):
    for p in (3, 4, 5):
        got = ops.TS_RSQ(panels["x"], p).iloc[:, 0].to_numpy(dtype=np.float64)
        ref = np.full(len(panels["x"]), np.nan)
        for _, sub in panels["x"].groupby(level="instrument", sort=False):
            pos = panels["x"].index.get_indexer(sub.index)
            ref[pos] = _roll_rsq_ref_series(sub.iloc[:, 0], p)
        nan_mismatch = int(np.sum(np.isnan(got) != np.isnan(ref)))
        assert nan_mismatch == 0, f"TS_RSQ(p={p}): NaN 不一致（{nan_mismatch}）"
        np.testing.assert_allclose(got, ref, rtol=TOL, atol=TOL, equal_nan=True)


def test_ts_rsq_short_window(tiny_panels):
    # p=1 恒有效样本 <2 → 全 NaN
    assert np.isnan(ops.TS_RSQ(tiny_panels["x"], 1).iloc[:, 0].to_numpy()).all()
    with pytest.raises(ValueError):
        ops.TS_RSQ(tiny_panels["x"], 0)


def test_accel_matches_pandas_fallback(monkeypatch):
    p = _build(seed=3)
    x, g = p["x"], p["g"]
    cases = {
        "CS_GROUP_MEAN": lambda: ops.CS_GROUP_MEAN(x, g),
        "CS_GROUP_SUM": lambda: ops.CS_GROUP_SUM(x, g),
        "CS_GROUP_STD": lambda: ops.CS_GROUP_STD(x, g),
        "CS_GROUP_MEDIAN": lambda: ops.CS_GROUP_MEDIAN(x, g),
        "CS_GROUP_ZSCORE": lambda: ops.CS_GROUP_ZSCORE(x, g),
        "CS_GROUP_COUNT": lambda: ops.CS_GROUP_COUNT(g),
        "SCALE": lambda: ops.SCALE(x),
        "SCALE(a=-2)": lambda: ops.SCALE(x, -2.0),
    }
    from alphaagent.dsl.core import cs_accel

    assert cs_accel.cs_accel_enabled(), "numba 快路径未启用，本门禁失去意义"

    def call(use_accel: bool):
        if use_accel:
            monkeypatch.delenv("ALPHA_DSL_CS_ACCEL", raising=False)
        else:
            monkeypatch.setenv("ALPHA_DSL_CS_ACCEL", "0")
        return fn().iloc[:, 0].to_numpy(dtype=np.float64)

    for name, fn in cases.items():
        old = call(False)
        new = call(True)
        nan_mismatch = int(np.sum(np.isnan(old) != np.isnan(new)))
        assert nan_mismatch == 0, f"{name}: NaN 位置不一致（{nan_mismatch}）"
        same = (old == new) | (np.isnan(old) & np.isnan(new))
        with np.errstate(invalid="ignore"):
            diff = np.where(same, 0.0, np.abs(old - new))
        max_diff = float(np.max(diff)) if diff.size else 0.0
        assert max_diff <= 1e-6, f"{name}: 快/慢路径最大偏差 {max_diff:.3e}"


def test_unsorted_panel_falls_back():
    p = _build(shuffle=True)
    out = ops.CS_GROUP_MEAN(p["x"], p["g"])
    assert out.shape == p["x"].shape
    assert np.isfinite(out.iloc[:, 0].to_numpy()).sum() > 0


def test_catalog_visibility():
    from alphaagent.dsl.catalog import operator_catalog_markdown

    md = operator_catalog_markdown(tier="full")
    for sig in [
        "CS_GROUP_MEAN(x, group)",
        "CS_GROUP_SUM(x, group)",
        "CS_GROUP_STD(x, group, ddof=1)",
        "CS_GROUP_MEDIAN(x, group)",
        "CS_GROUP_ZSCORE(x, group, ddof=1)",
        "CS_GROUP_COUNT(group)",
        "SCALE(df, a=1.0)",
        "TS_AUTOCORR(x, window, lag=1)",
        "TS_RSQ(df, p=5, n_jobs=1)",
    ]:
        assert sig in md, f"catalog 缺签名 {sig}"


def test_guard_blocks_negative_window_but_exempts_scale(panels):
    ns = wrap_lookahead_guard(build_operator_namespace())
    x = panels["x"]
    with pytest.raises(ValueError):
        ns["TS_AUTOCORR"](x, -5, 1)
    with pytest.raises(ValueError):
        ns["TS_RSQ"](x, -5)
    # SCALE 负 a 合法（截面归一，无时间维度）
    out = ns["SCALE"](x, -1)
    assert out.shape == x.shape
