"""CS_* 截面算子 Numba 快路径 vs pandas 回落路径 一致性门禁。

2026-10-08：``RANK`` / ``CS_WINSORIZE`` / ``CS_BUCKET`` / ``CS_NEUTRALIZE`` /
``CS_GROUP_RANK`` 从 pandas 逐日实现改为 ``cs_accel`` 的 Numba 逐日并行内核
（实测 7.78M 行面板上快 12~44 倍）。本门禁在同一输入上跑两条路径并要求一致：

- NaN 位置逐个一致；
- 取值最大绝对偏差 ≤ 1e-6（float32 输出的舍入差；CS_BUCKET 输出为整数档号，
  该容差仍能抓住任何错分档）。

覆盖：并列值（rank/winsorize/qcut 的并列分支）、常数截面、±inf（各算子口径不同：
RANK / CS_WINSORIZE 视 inf 为有效值，CS_BUCKET / CS_NEUTRALIZE / CS_GROUP_RANK
按 isfinite 过滤）、组内单样本、有效样本 < n_bins 的截面、多档 CS_BUCKET、
乱序面板（快路径必须自动回落）。
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
    x[extreme] = np.inf                               # ±inf 口径差异
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
]


def _cases(p):
    x, g = p["x"], p["group"]
    bucket10 = ops.CS_BUCKET(g, 10)
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
