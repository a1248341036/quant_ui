"""事件驱动回测引擎 — 向后兼容 re-export 层。

原 core/event_engine.py 已拆分为:
- context: Order, Bar, EventStrategy, Context
- runner: _load_st_mask_for, run_event_backtest
- strategies: GoldenCrossStrategy, RiskParityStrategy, LongShortMomentumStrategy
"""

from .context import Order, Bar, EventStrategy, Context
from .runner import _load_st_mask_for, run_event_backtest
from .strategies import (
    GoldenCrossStrategy,
    RiskParityStrategy,
    LongShortMomentumStrategy,
)

__all__ = [
    "Order",
    "Bar",
    "EventStrategy",
    "Context",
    "run_event_backtest",
    "GoldenCrossStrategy",
    "RiskParityStrategy",
    "LongShortMomentumStrategy",
]
