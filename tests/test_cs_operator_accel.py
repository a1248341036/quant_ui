"""CS_* 截面算子 Numba 快路径 vs pandas 回落路径 一致性门禁。

2026-10-08：``RANK`` / ``CS_WINSORIZE`` / ``CS_BUCKET`` / ``CS_NEUTRALIZE`` /
``CS_GROUP_RANK`` / ``CS_ZSCORE`` / ``CS_RESIDUALIZE`` 从 pandas 逐日实现改为
``cs_accel`` 的 Numba 逐日并行内核（实测 7.78M 行面板上快 9~40 倍）。本门禁在
同一输入上跑两条路径并要求一致：

- NaN 位置逐个一致；
- 取值最大绝对偏差 ≤ 1e-6（float32 输出的舍入差；CS_BUCKET 输出为整数档号，
  该容差仍能抓住任何错分档）。

覆盖：并列值（rank/winsorize/qcut 的并列分支）、常数截面、±inf（各算子口径不同：
RANK / CS_WINSORIZE / CS_ZSCORE 视 inf 为有效值（ZSCORE 含 inf 截面整段 NaN），
CS_BUCKET / CS_NEUTRALIZE / CS_GROUP_RANK / CS_RESIDUALIZE 按 isfinite 过滤）、
组内单样本、有效样本 < n_bins 的截面、多档 CS_BUCKET、CS_ZSCORE 的 ddof 两档、
CS_RESIDUALIZE 的常数控制/逐位精确共线/双控制变量、乱序面板（快路径必须自动回落）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.dsl.core import operators as ops

pytest.importorskip("numba")

TOL = 1e-6
N_INST = 40
N_DAYS = 120


def _build(n_inst: int = N_INST, n_days: int = N_DAYS, seed: int = 11, shuffle: bool = False):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    n = len(idx)

    x = rng.normal(size=n)
    cap = np.exp(rng.normal(9, 0.5, size=n))          # 连续分组键（模拟市值）
    x[: n // 4] = np.round(x[: n // 4], 1)            # 制造并列值
    x[n // 3 : n // 3 + n_inst] = 3.0                 # 一个常数截面
    missing = rng.random(n) < 0.05
    x[missing] = np.nan
    cap[missing] = np.nan
    extreme = rng.random(n) < 0.002
    x[extreme] = np.where(rng.random(int(extreme.sum())) < 0.5, np.inf, -np.inf)  # ±inf 双向口径
    cap[extreme] = np.nan

    if shuffle:
        perm = rng.permutation(n)
        x, cap = x[perm], cap[perm]
        idx = idx[perm]

    return {
        "x": pd.DataFrame(x, index=idx, columns=["x"]),
        "group": pd.DataFrame(cap, index=idx, columns=["g"]),
    }


@pytest.fixture(scope="module")
def panels():
    return _build()


@pytest.fixture(scope="module")
def tiny_panels():
    """3 个品种的窄面板：组内单样本 / 有效样本 < n_bins 的截面都会出现。"""
    return _build(n_inst=3, n_days=25, seed=5)


def _call(fn, monkeypatch, use_accel: bool):
    if use_accel:
        monkeypatch.delenv("ALPHA_DSL_CS_ACCEL", raising=False)
    else:
        monkeypatch.setenv("ALPHA_DSL_CS_ACCEL", "0")
    return fn().iloc[:, 0].to_numpy(dtype=np.float64)


def _assert_same(name, fn, monkeypatch, active: bool = True):
    old = _call(fn, monkeypatch, use_accel=False)
    new = _call(fn, monkeypatch, use_accel=True)
    nan_mismatch = int(np.sum(np.isnan(old) != np.isnan(new)))
    assert nan_mismatch == 0, f"{name}: NaN 位置不一致（{nan_mismatch} 处）"
    same = (old == new) | (np.isnan(old) & np.isnan(new))  # ±inf 相同视为一致
    with np.errstate(invalid="ignore"):
        diff = np.where(same, 0.0, np.abs(old - new))
    max_diff = float(np.max(diff)) if diff.size else 0.0
    assert max_diff <= TOL, f"{name}: 最大偏差 {max_diff:.3e} 超容差 {TOL}"
    if active:
        # 快路径确实生效（引擎入口返回 None 说明回落了，门禁就等于空转）
        from alphaagent.dsl.core import cs_accel

        assert cs_accel.cs_accel_enabled(), "numba 快路径未启用，本门禁失去意义"


CASE_NAMES = [
    "RANK",
    "CS_RANK 别名",
    "CS_WINSORIZE(0.01,0.99)",
    "CS_WINSORIZE(0.1,0.9)",
    "CS_BUCKET(2)",
    "CS_BUCKET(10)",
    "CS_BUCKET(13)",
    "CS_BUCKET(x)",
    "CS_NEUTRALIZE(x,bucket10)",
    "CS_NEUTRALIZE(x,group)",
    "CS_GROUP_RANK(x,bucket10)",
    "CS_GROUP_RANK(x,group)",
    "CS_ZSCORE",
    "CS_ZSCORE(ddof=0)",
    "CS_RESIDUALIZE(x,group)",
    "CS_RESIDUALIZE(x,group,log)",
    "CS_RESIDUALIZE(x,group,2x)",
    "CS_RESIDUALIZE(x,group,group)",
    "CS_RESIDUALIZE(x,const)",
]


def _cases(p):
    x, g = p["x"], p["group"]
    bucket10 = ops.CS_BUCKET(g, 10)
    garr = g.iloc[:, 0].to_numpy(dtype=float)
    g2x = pd.DataFrame(garr * 2.0, index=g.index, columns=["g2"])    # ×2 逐位精确 → 共线
    with np.errstate(invalid="ignore"):
        glog = pd.DataFrame(np.log(garr), index=g.index, columns=["glog"])  # 真实双控制
    gconst = pd.DataFrame(
        np.where(np.isnan(garr), np.nan, 7.0), index=g.index, columns=["gc"]
    )
    return {
        "RANK": lambda: ops.RANK(x),
        "CS_RANK 别名": lambda: ops.CS_RANK(x),
        "CS_WINSORIZE(0.01,0.99)": lambda: ops.CS_WINSORIZE(x, 0.01, 0.99),
        "CS_WINSORIZE(0.1,0.9)": lambda: ops.CS_WINSORIZE(x, 0.1, 0.9),
        "CS_BUCKET(2)": lambda: ops.CS_BUCKET(g, 2),
        "CS_BUCKET(10)": lambda: ops.CS_BUCKET(g, 10),
        "CS_BUCKET(13)": lambda: ops.CS_BUCKET(g, 13),
        "CS_BUCKET(x)": lambda: ops.CS_BUCKET(x, 5),
        "CS_NEUTRALIZE(x,bucket10)": lambda: ops.CS_NEUTRALIZE(x, bucket10),
        "CS_NEUTRALIZE(x,group)": lambda: ops.CS_NEUTRALIZE(x, g),
        "CS_GROUP_RANK(x,bucket10)": lambda: ops.CS_GROUP_RANK(x, bucket10),
        "CS_GROUP_RANK(x,group)": lambda: ops.CS_GROUP_RANK(x, g),
        "CS_ZSCORE": lambda: ops.CS_ZSCORE(x),
        "CS_ZSCORE(ddof=0)": lambda: ops.CS_ZSCORE(x, ddof=0),
        "CS_RESIDUALIZE(x,group)": lambda: ops.CS_RESIDUALIZE(x, g),
        "CS_RESIDUALIZE(x,group,log)": lambda: ops.CS_RESIDUALIZE(x, g, glog),
        "CS_RESIDUALIZE(x,group,2x)": lambda: ops.CS_RESIDUALIZE(x, g, g2x),
        "CS_RESIDUALIZE(x,group,group)": lambda: ops.CS_RESIDUALIZE(x, g, g),
        "CS_RESIDUALIZE(x,const)": lambda: ops.CS_RESIDUALIZE(x, gconst),
    }


@pytest.mark.parametrize("name", CASE_NAMES)
def test_accel_matches_pandas(panels, monkeypatch, name):
    _assert_same(name, _cases(panels)[name], monkeypatch)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_tiny_panel_edge_cases(tiny_panels, monkeypatch, name):
    """窄面板：组内单样本、< n_bins 截面、并列值边界。"""
    _assert_same(f"{name}/窄面板", _cases(tiny_panels)[name], monkeypatch)


def test_unsorted_panel_falls_back(monkeypatch):
    """乱序面板：快路径返回 None（内部回落 pandas），结果仍与 pandas 路径一致。"""
    p = _build(shuffle=True)
    x = p["x"]
    from alphaagent.dsl.core import cs_accel

    arr = x.iloc[:, 0].to_numpy(dtype=float)
    from alphaagent.dsl.core.ops_kit import datetime_group_bounds

    assert datetime_group_bounds(x) is None, "乱序面板不应产出归组边界"
    monkeypatch.delenv("ALPHA_DSL_CS_ACCEL", raising=False)
    assert cs_accel.rank_pct(arr, None) is None
    assert cs_accel.bucket(arr, None, 10) is None
    out = ops.RANK(x)
    ref = x.iloc[:, 0].groupby(level="datetime", sort=False).rank(pct=True, method="average")
    assert np.allclose(
        np.nan_to_num(out.iloc[:, 0].to_numpy(), nan=-9),
        np.nan_to_num(ref.to_numpy(), nan=-9),
        atol=TOL,
    )


def test_env_switch_disables_fast_path(monkeypatch):
    """ALPHA_DSL_CS_ACCEL=0 时不再走内核（排障 / 对照开关）。"""
    p = _build(n_inst=5, n_days=10, seed=3)
    from alphaagent.dsl.core import cs_accel

    monkeypatch.setenv("ALPHA_DSL_CS_ACCEL", "0")
    assert not cs_accel.cs_accel_enabled()
    arr = p["x"].iloc[:, 0].to_numpy(dtype=float)
    bounds = np.array([0, len(arr)], dtype=np.int64)
    assert cs_accel.rank_pct(arr, bounds) is None
    assert cs_accel.winsorize(arr, bounds, 0.05, 0.95) is None
    assert cs_accel.neutralize(arr, arr, bounds) is None
    assert cs_accel.group_rank_pct(arr, arr, bounds) is None
    assert cs_accel.bucket(arr, bounds, 4) is None
    assert cs_accel.zscore(arr, bounds, 1) is None
    assert cs_accel.residualize(arr, arr.reshape(-1, 1), bounds) is None
    # 算子本身仍可用（回落 pandas 路径）
    assert ops.RANK(p["x"]).shape == p["x"].shape


def test_group_rank_single_sample_is_one(monkeypatch):
    """组内单样本 → rank(pct=True)=1.0（锁定语义，勿改回文档里写的 0.5）。"""
    idx = pd.MultiIndex.from_tuples(
        [("2021-01-04", "A"), ("2021-01-04", "B"), ("2021-01-04", "C")],
        names=["datetime", "instrument"],
    )
    x = pd.DataFrame([1.0, 2.0, 3.0], index=idx, columns=["x"])
    g = pd.DataFrame([0.0, 1.0, 1.0], index=idx, columns=["g"])
    out = ops.CS_GROUP_RANK(x, g).iloc[:, 0].to_numpy()
    assert out[0] == pytest.approx(1.0)      # 组 0 只有一个样本
    assert out[1] == pytest.approx(0.5)      # 组 1 两个样本
    assert out[2] == pytest.approx(1.0)


def _mini_index():
    days = pd.bdate_range("2021-01-04", periods=4)
    idx = pd.MultiIndex.from_product(
        [days, [f"S{i}" for i in range(6)]], names=["datetime", "instrument"]
    )
    return idx, np.asarray(idx.get_level_values("datetime")), days


def test_zscore_constant_day_is_nan(monkeypatch):
    """某日截面全为同一值 → std=0 → 整段 NaN（新旧一致，锁定语义）。"""
    idx, day_arr, days = _mini_index()
    rng = np.random.default_rng(9)
    y = rng.normal(size=len(idx))
    d0 = day_arr == days[0]
    y[d0] = 2.5
    P = pd.DataFrame(y, index=idx, columns=["v"])
    _assert_same("CS_ZSCORE 常数截面", lambda: ops.CS_ZSCORE(P), monkeypatch)
    out = ops.CS_ZSCORE(P).iloc[:, 0].to_numpy()
    assert np.isnan(out[d0]).all(), "std=0 的截面必须整段 NaN"
    assert np.isfinite(out[~d0]).mean() > 0.9


def test_residualize_degenerate_segments(monkeypatch):
    """常数控制 / 逐位精确共线 / 有效样本 < k+2 → 整段 NaN，正常段有值。"""
    idx, day_arr, days = _mini_index()
    rng = np.random.default_rng(11)
    y = rng.normal(size=len(idx))
    z1 = rng.normal(size=len(idx))
    P = lambda a: pd.DataFrame(a, index=idx, columns=["v"])

    # 1) 控制变量某日为常数 → 该日 NaN（matrix_rank 必判缺秩）
    zc = z1.copy()
    zc[day_arr == days[0]] = 3.14
    _assert_same("残差化·常数控制", lambda: ops.CS_RESIDUALIZE(P(y), P(zc)), monkeypatch)
    out = ops.CS_RESIDUALIZE(P(y), P(zc)).iloc[:, 0].to_numpy()
    assert np.isnan(out[day_arr == days[0]]).all()
    assert np.isfinite(out[day_arr != days[0]]).mean() > 0.8

    # 2) 双控制变量逐位精确共线（z2 = 2·z1，×2 是位精确缩放）→ 全部截面 NaN
    z2 = 2.0 * z1
    _assert_same(
        "残差化·精确共线", lambda: ops.CS_RESIDUALIZE(P(y), P(z1), P(z2)), monkeypatch
    )
    assert np.isnan(ops.CS_RESIDUALIZE(P(y), P(z1), P(z2)).iloc[:, 0].to_numpy()).all()

    # 3) 某日只剩 2 个有效行（k=1 需 ≥3）→ 该日 NaN
    y2 = y.copy()
    d3 = day_arr == days[2]
    d3_pos = np.flatnonzero(d3)
    y2[d3_pos[:4]] = np.nan
    _assert_same("残差化·低样本", lambda: ops.CS_RESIDUALIZE(P(y2), P(z1)), monkeypatch)
    out2 = ops.CS_RESIDUALIZE(P(y2), P(z1)).iloc[:, 0].to_numpy()
    assert np.isnan(out2[d3]).all()
    assert np.isfinite(out2[~d3]).mean() > 0.8


def test_winsorize_nan_quantile_bounds(monkeypatch):
    """-inf 截面使 q_lo 插值出 NaN → 整段 NaN，与 np.clip 传播一致。

    OCR 2026-10-08 finding：q_lo=NaN 且 q_hi 有限时旧内核会错输出 q_hi（有限值）。
    注意不对称性：+inf 只会把 q_hi 插值成 **+inf**（a + g·(b−a) = finite + g·inf），
    clip 上界无效但两路径一致输出有限值，不产生 NaN——NaN 边界只可能来自
    -inf 作插值下端点（sorted 升序下 q_hi 区间的下端点不可能为 -inf）。"""
    days = pd.bdate_range("2021-03-01", periods=3)
    idx = pd.MultiIndex.from_product(
        [days, [f"S{i}" for i in range(100)]], names=["datetime", "instrument"]
    )
    day_arr = np.asarray(idx.get_level_values("datetime"))
    rng = np.random.default_rng(4)
    vals = rng.normal(size=len(idx))
    pos0 = np.flatnonzero(day_arr == days[0])
    vals[pos0] = np.linspace(1.0, 100.0, 100)
    vals[pos0[0]] = -np.inf                    # 单个 -inf：q_lo 插值跨 -inf → NaN
    pos1 = np.flatnonzero(day_arr == days[1])
    vals[pos1] = np.linspace(1.0, 100.0, 100)
    vals[pos1[0]] = np.inf                     # 单个 +inf：q_hi → +inf，clip 上界无效
    P = pd.DataFrame(vals, index=idx, columns=["v"])
    _assert_same("WINSORIZE·NaN 分位边界", lambda: ops.CS_WINSORIZE(P, 0.01, 0.99), monkeypatch)
    out = ops.CS_WINSORIZE(P, 0.01, 0.99).iloc[:, 0].to_numpy()
    assert np.isnan(out[day_arr == days[0]]).all(), "q_lo=NaN 的截面必须整段 NaN"
    assert np.isfinite(out[day_arr == days[1]]).mean() > 0.9, "q_hi=+inf 只是 clip 上界无效，非 NaN"
    assert np.isfinite(out[day_arr == days[2]]).mean() > 0.9  # 正常截面不受影响


def test_bucket_all_equal_segment_is_nan(monkeypatch):
    """截面内有限值全相等且数量 ≥ n_bins → 两条路径都整段 NaN。

    OCR 2026-10-08 曾误报此处分叉（以为 qcut 会抛 ValueError 走 rank 重建）；
    实证 pd.qcut 对全等值不抛异常而是返回全 NaN codes，快慢路径一致。用例锁定该语义。"""
    days = pd.bdate_range("2021-04-01", periods=3)
    idx = pd.MultiIndex.from_product(
        [days, [f"S{i}" for i in range(40)]], names=["datetime", "instrument"]
    )
    day_arr = np.asarray(idx.get_level_values("datetime"))
    rng = np.random.default_rng(5)
    vals = rng.normal(size=len(idx))
    vals[day_arr == days[0]] = 3.0                    # 40 个全等值（≥ n_bins=5）
    P = pd.DataFrame(vals, index=idx, columns=["v"])
    _assert_same("BUCKET·全等值截面", lambda: ops.CS_BUCKET(P, 5), monkeypatch)
    out = ops.CS_BUCKET(P, 5).iloc[:, 0].to_numpy()
    assert np.isnan(out[day_arr == days[0]]).all(), "全等值截面必须整段 NaN（pd.qcut 同语义）"
    assert np.isfinite(out[day_arr != days[0]]).mean() > 0.9
