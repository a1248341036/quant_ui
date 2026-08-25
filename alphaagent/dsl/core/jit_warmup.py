"""Numba JIT 预热模块：在 Agent 启动前触发慢算子的首次编译。

问题：VOLUME_CLOCK_VPIN / TS_EFFICIENCY_RATIO / TS_PERMUTATION_ENTROPY 等
算子包含复杂的 @njit 函数，首次调用时 JIT 编译需要数十秒到数分钟，
导致 Agent 运行时超时（180s timeout）。

解决方案：在 Agent 启动前用极小 dummy 数据触发编译，编译结果通过
cache=True 持久化到磁盘，后续真实调用直接加载缓存。
"""
from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def warmup_numba_jit() -> dict[str, float]:
    """预编译所有已知慢 JIT 算子，返回 {算子名: 编译耗时秒}。

    用极小的 dummy 数据（50 根 bar）触发编译，数据量不影响编译时间
    （numba 按函数 signature 编译，与输入长度无关）。
    """
    from alphaagent.dsl.core import accel as _accel

    results: dict[str, float] = {}

    # --- 1. VOLUME_CLOCK_VPIN (最慢，之前 >2 分钟) ---
    try:
        t0 = time.perf_counter()
        n = 60
        rng = np.random.default_rng(42)
        price = (100.0 + rng.standard_normal(n).cumsum()).astype(np.float32)
        volume = (1e6 + rng.random(n) * 1e6).astype(np.float32)
        _accel.volume_clock_vpin_fixed(price, volume, window=10, bucket_size=5e5, classification="tick")
        dt = time.perf_counter() - t0
        results["volume_clock_vpin"] = dt
        logger.info("JIT warmup: volume_clock_vpin compiled in %.1fs", dt)
    except Exception as e:
        logger.warning("JIT warmup: volume_clock_vpin failed: %s", e)
        results["volume_clock_vpin"] = -1.0

    # --- 2. roll_efficiency_ratio (TS_EFFICIENCY_RATIO 的内核) ---
    try:
        t0 = time.perf_counter()
        vals = np.random.default_rng(42).standard_normal(60).astype(np.float32)
        _accel.roll_efficiency_ratio_fixed(vals, 20)
        dt = time.perf_counter() - t0
        results["roll_efficiency_ratio"] = dt
        logger.info("JIT warmup: roll_efficiency_ratio compiled in %.1fs", dt)
    except Exception as e:
        logger.warning("JIT warmup: roll_efficiency_ratio failed: %s", e)
        results["roll_efficiency_ratio"] = -1.0

    # --- 3. roll_permutation_entropy (TS_PERMUTATION_ENTROPY 的内核) ---
    try:
        t0 = time.perf_counter()
        vals = np.random.default_rng(42).standard_normal(60).astype(np.float32)
        _accel.roll_permutation_entropy_fixed(vals, 30, order=3)
        dt = time.perf_counter() - t0
        results["roll_permutation_entropy"] = dt
        logger.info("JIT warmup: roll_permutation_entropy compiled in %.1fs", dt)
    except Exception as e:
        logger.warning("JIT warmup: roll_permutation_entropy failed: %s", e)
        results["roll_permutation_entropy"] = -1.0

    # --- 4. chip_daily (CHIP_PEAK_LOC, CHIP_ENTROPY, CHIP_MASS_ASYM 等) ---
    try:
        t0 = time.perf_counter()
        n = 60
        rng = np.random.default_rng(42)
        close = (100.0 + rng.standard_normal(n).cumsum()).astype(np.float32)
        low = close - rng.random(n) * 2.0
        high = close + rng.random(n) * 2.0
        volume = (1e6 + rng.random(n) * 1e6).astype(np.float32)
        aux = (1e8 + rng.random(n) * 1e8).astype(np.float32)
        _accel.roll_chip_metric_fixed(
            close, volume, low, high, aux, 40, 30, "peak_loc", "cyq"
        )
        _accel.roll_chip_metric_fixed(
            close, volume, low, high, aux, 40, 30, "entropy", "cyq"
        )
        _accel.roll_chip_metric_fixed(
            close, volume, low, high, aux, 40, 30, "mass_asym", "cyq"
        )
        dt = time.perf_counter() - t0
        results["chip_daily"] = dt
        logger.info("JIT warmup: chip_daily compiled in %.1fs", dt)
    except Exception as e:
        logger.warning("JIT warmup: chip_daily failed: %s", e)
        results["chip_daily"] = -1.0

    # --- 5. roll_corr (TS_CORR 的内核) ---
    try:
        t0 = time.perf_counter()
        vals = np.random.default_rng(42).standard_normal(60).astype(np.float32)
        other = np.random.default_rng(99).standard_normal(60).astype(np.float32)
        _accel.roll_corr_fixed(vals, other, 20)
        dt = time.perf_counter() - t0
        results["roll_corr"] = dt
        logger.info("JIT warmup: roll_corr compiled in %.1fs", dt)
    except Exception as e:
        logger.warning("JIT warmup: roll_corr failed: %s", e)
        results["roll_corr"] = -1.0

    # --- 6. roll_rankcorr (TS_RANK_CORR 的内核) ---
    try:
        t0 = time.perf_counter()
        vals = np.random.default_rng(42).standard_normal(60).astype(np.float32)
        other = np.random.default_rng(99).standard_normal(60).astype(np.float32)
        _accel.roll_rankcorr_fixed(vals, other, 20)
        dt = time.perf_counter() - t0
        results["roll_rankcorr"] = dt
        logger.info("JIT warmup: roll_rankcorr compiled in %.1fs", dt)
    except Exception as e:
        logger.warning("JIT warmup: roll_rankcorr failed: %s", e)
        results["roll_rankcorr"] = -1.0

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()
    results = warmup_numba_jit()
    total = time.perf_counter() - t0
    print(f"\n=== JIT warmup complete in {total:.1f}s ===")
    for name, dt in results.items():
        status = "OK" if dt >= 0 else "FAILED"
        print(f"  {name}: {dt:.1f}s ({status})")
