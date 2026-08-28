"""DSL 慢算子性能门禁。

固定合成面板上对 2026-08 优化过的算子（boundaries 按品种并行层）做耗时预算断言，
防止重构静默退化（如并行路径被改坏回落串行、pandas 分组开销回归）。

预算按 720k 行面板（600 品种 × 1200 日）标定，约为当前实测耗时的 4~6 倍余量；
若numba 缺失或并行被环境变量禁用，本门禁跳过（回落正确性由一致性门禁保证）。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from alphaagent.dsl.core import operators as ops

pytest.importorskip("numba")

N_INST = 600
N_DAYS = 1200


def _make_panel() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2021-01-04", periods=N_DAYS)
    inst = [f"S{i:04d}" for i in range(N_INST)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    n = len(idx)

    close = 100.0 * np.exp(
        np.cumsum(rng.normal(0, 0.02, (N_DAYS, N_INST)), axis=0)
    ).ravel(order="F")
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    volume = rng.lognormal(10, 1, n)
    float_cap = rng.lognormal(9, 0.5, n)

    def P(a):
        return pd.DataFrame(a, index=idx)

    return {
        "open": P(open_),
        "high": P(high),
        "low": P(low),
        "close": P(close),
        "volume": P(volume),
        "float_cap": P(float_cap),
    }


@pytest.fixture(scope="module")
def pnl() -> dict[str, pd.DataFrame]:
    return _make_panel()


@pytest.fixture(scope="module")
def warm(pnl):
    """小面板触发 JIT 编译（磁盘缓存命中后为毫秒级），避免编译时间计入预算。"""
    small = {k: v.iloc[:240] for k, v in pnl.items()}
    ops.PRICE_GAP_FILL(small["open"], small["high"], small["low"], small["close"])
    ops.CHIP_ENTROPY(small["close"], small["low"], small["high"], small["volume"], 20, small["float_cap"])
    ops.CHIP_PEAK_LOC(small["close"], small["low"], small["high"], small["volume"], 20, small["float_cap"])
    ops.CHIP_WASS_DIST(small["close"], small["low"], small["high"], small["volume"], 20, small["float_cap"])
    ops.CHIP_BIMODAL_SCORE(small["close"], small["low"], small["high"], small["volume"], 20, small["float_cap"])
    ops.WICK_EFFICIENCY(small["open"], small["high"], small["low"], small["close"], 3)
    ops.CROWD_SHARE(small["close"], small["volume"], 20, "high", 0.9)
    ops.VOLUME_CLOCK_VPIN(small["close"], small["volume"], 5, 5e5)
    ops.MUTUAL_INFO_LAG(small["close"], small["volume"], 30, 1)


def _timed(fn, *args, **kw) -> float:
    t0 = time.perf_counter()
    fn(*args, **kw)
    return time.perf_counter() - t0


BUDGET_CASES = [
    ("PRICE_GAP_FILL", 2.0),
    ("CHIP_ENTROPY", 2.0),
    ("CHIP_PEAK_LOC", 2.0),
    ("CHIP_WASS_DIST", 3.5),
    ("CHIP_BIMODAL_SCORE", 3.5),
    ("WICK_EFFICIENCY", 1.0),
    ("CROWD_SHARE", 1.5),
    ("VOLUME_CLOCK_VPIN", 1.0),
    ("MUTUAL_INFO_LAG", 2.0),
]


@pytest.mark.parametrize("name,budget", BUDGET_CASES)
def test_operator_within_budget(pnl, warm, name, budget):
    if os.environ.get("ALPHA_DSL_BOUNDARIES_PARALLEL", "1").strip().lower() in ("0", "false", "off"):
        pytest.skip("并行边界层被环境变量禁用")
    from alphaagent.dsl.core import accel as _accel

    if not _accel.accel_available().get("numba"):
        pytest.skip("numba 不可用，回落旧路径")
    if not _accel._boundaries_parallel_enabled():
        pytest.skip("并行层被禁用")

    o, h, l, c, v, f = (pnl[k] for k in ("open", "high", "low", "close", "volume", "float_cap"))
    dispatch = {
        "PRICE_GAP_FILL": lambda: ops.PRICE_GAP_FILL(o, h, l, c),
        "CHIP_ENTROPY": lambda: ops.CHIP_ENTROPY(c, l, h, v, 60, f),
        "CHIP_PEAK_LOC": lambda: ops.CHIP_PEAK_LOC(c, l, h, v, 60, f),
        "CHIP_WASS_DIST": lambda: ops.CHIP_WASS_DIST(c, l, h, v, 60, f, 32, 10),
        "CHIP_BIMODAL_SCORE": lambda: ops.CHIP_BIMODAL_SCORE(c, l, h, v, 60, f, 32, "simple"),
        "WICK_EFFICIENCY": lambda: ops.WICK_EFFICIENCY(o, h, l, c, 3),
        "CROWD_SHARE": lambda: ops.CROWD_SHARE(c, v, 20, "high", 0.9),
        "VOLUME_CLOCK_VPIN": lambda: ops.VOLUME_CLOCK_VPIN(c, v, 5, 5e5),
        "MUTUAL_INFO_LAG": lambda: ops.MUTUAL_INFO_LAG(c, v, 30, 1),
    }
    elapsed = _timed(dispatch[name])
    assert elapsed < budget, f"{name} 耗时 {elapsed:.2f}s 超预算 {budget}s —— 并行层可能退化，检查 _boundaries_fast / accel 边界内核"


def test_fast_path_wired(pnl):
    """快路径必须处于激活状态：_boundaries_fast 能产出归组边界（numba 可用时）。"""
    from alphaagent.dsl.core import accel as _accel

    if not _accel.accel_available().get("numba"):
        pytest.skip("numba 不可用")
    fast = ops._boundaries_fast(pnl["close"])
    assert fast is not None, "_boundaries_fast 返回 None，并行快路径未生效"
    order, bounds, inv, arrays = fast
    assert order.shape[0] == len(pnl["close"])
    assert bounds[0] == 0 and bounds[-1] == len(pnl["close"])
    assert (np.diff(bounds) > 0).all(), "品种区间边界必须严格递增"
