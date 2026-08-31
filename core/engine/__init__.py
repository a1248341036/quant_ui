"""engine 包：回测引擎四阶段管线（向后兼容 re-export）。

原 core.engine.py 已拆分为：
- factors:         因子构建（build_factor_frames, build_composite_factor 等）
- config:          BacktestConfig + run_backtest / run_backtest_config 门面
- prepare:         阶段 1 准备
- factor_matrix:   阶段 2 因子矩阵
- simulate:        阶段 3 主循环模拟
- result:          阶段 4 收尾 + latest_signals
"""
from .config import BacktestConfig, run_backtest, run_backtest_config
from .factors import (
    build_factor_frames,
    build_composite_factor,
    _inject_pred_factor,
    _ensure_ma_cross_factor,
    _selection_count,
    _compute_atr,
    _compute_adx,
)
from .prepare import _prepare_backtest, _load_st_mask_for
from .factor_matrix import _build_factor_matrix
from .simulate import _simulate
from .result import _finalize_result, latest_signals

__all__ = [
    "BacktestConfig",
    "run_backtest",
    "run_backtest_config",
    "build_factor_frames",
    "build_composite_factor",
    "latest_signals",
]
