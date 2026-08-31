"""日频筹码分布内核：uniform / cyq / triangular 三种构建方式。"""
from __future__ import annotations

from ._ids import (
    CHIP_OP,
    _CHIP_METHOD,
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
from ._wrappers import (
    roll_chip_bimodal_daily,
    roll_chip_metric_daily,
    roll_chip_peak_sharpness_daily,
    roll_chip_wass_dist_daily,
)

__all__ = [
    "CHIP_OP",
    "_CHIP_METHOD",
    "chip_bimodal_impl_id",
    "chip_method_id",
    "chip_peak_sharpness_impl_id",
    "chip_wass_implementation_id",
    "roll_chip_bimodal_daily",
    "roll_chip_bimodal_daily_numba",
    "roll_chip_metric_daily",
    "roll_chip_metric_daily_numba",
    "roll_chip_peak_sharpness_daily",
    "roll_chip_peak_sharpness_daily_numba",
    "roll_chip_wass_dist_daily",
    "roll_chip_wass_dist_daily_numba",
]
