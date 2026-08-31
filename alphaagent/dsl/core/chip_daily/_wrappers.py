"""Python 包装函数：参数校验 + 类型转换 → 调用 njit kernels。"""
from __future__ import annotations

import numpy as np

from ._ids import (
    CHIP_OP,
    chip_bimodal_impl_id,
    chip_method_id,
    chip_peak_sharpness_impl_id,
    chip_wass_implementation_id,
)
from ._kernels import (
    roll_chip_bimodal_daily_numba,
    roll_chip_metric_daily_numba,
    roll_chip_peak_sharpness_daily_numba,
    roll_chip_wass_dist_daily_numba,
)


def roll_chip_metric_daily(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    op: str,
    method: str,
) -> np.ndarray:
    op_id = CHIP_OP.get(op)
    if op_id is None:
        raise ValueError(f"Unknown chip op: {op}")
    mid = chip_method_id(method)
    arrays = [
        close.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        low.astype(np.float32, copy=False),
        high.astype(np.float32, copy=False),
        aux.astype(np.float32, copy=False),
    ]
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise ValueError("chip daily arrays must have the same length")
    return roll_chip_metric_daily_numba(
        arrays[0], arrays[1], arrays[2], arrays[3], arrays[4],
        int(window), int(nbins), int(op_id), int(mid),
    )


def roll_chip_peak_sharpness_daily(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    implementation: str,
    method: str,
) -> np.ndarray:
    impl = chip_peak_sharpness_impl_id(implementation)
    mid = chip_method_id(method)
    return roll_chip_peak_sharpness_daily_numba(
        close.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        low.astype(np.float32, copy=False),
        high.astype(np.float32, copy=False),
        aux.astype(np.float32, copy=False),
        int(window), int(nbins), int(impl), int(mid),
    )


def roll_chip_bimodal_daily(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    implementation: str,
    method: str,
    *,
    lambda_scale: float = 1.0,
) -> np.ndarray:
    impl = chip_bimodal_impl_id(implementation)
    mid = chip_method_id(method)
    return roll_chip_bimodal_daily_numba(
        close.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        low.astype(np.float32, copy=False),
        high.astype(np.float32, copy=False),
        aux.astype(np.float32, copy=False),
        int(window), int(nbins), int(impl), int(mid), float(lambda_scale),
    )


def roll_chip_wass_dist_daily(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    rho: np.ndarray,
    nbins: int,
    implementation: str,
    method: str,
) -> np.ndarray:
    impl = chip_wass_implementation_id(implementation)
    mid = chip_method_id(method)
    return roll_chip_wass_dist_daily_numba(
        close.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        low.astype(np.float32, copy=False),
        high.astype(np.float32, copy=False),
        aux.astype(np.float32, copy=False),
        wa.astype(np.int64, copy=False),
        wb.astype(np.int64, copy=False),
        rho.astype(np.int64, copy=False),
        int(nbins), int(impl), int(mid),
    )
