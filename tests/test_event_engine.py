from __future__ import annotations

import pytest

import pandas as pd

from core.event_engine import Context, EventStrategy, GoldenCrossStrategy, run_event_backtest
from conftest import CODES, END, START


def test_run_event_backtest_golden_cross(panel):
    res = run_event_backtest(
        panel=panel, codes=CODES, strategy_class=GoldenCrossStrategy,
        start=START, end=END, capital=1_000_000,
    )

    assert {"nav", "bench", "drawdown", "metrics", "trades", "holdings"} <= set(res)
    nav = res["nav"]
    assert len(nav) > 0
    assert nav.notna().all()
    assert nav.iloc[0] == pytest.approx(1.0, abs=0.05)
    assert res["metrics"]["总收益"] is not None


def test_run_event_backtest_custom_strategy(panel):
    class BuyFirstStrategy(EventStrategy):
        """第一个交易日买入前 2 只，其余时间持有。"""

        def __init__(self) -> None:
            self._done = False

        def on_bar(self, ctx: Context, bar) -> None:
            if self._done:
                return
            self._done = True
            for code in ctx.codes[:2]:
                if ctx.is_tradable(code):
                    ctx.order_target_pct(code, 0.4)

    res = run_event_backtest(
        panel=panel, codes=CODES, strategy_class=BuyFirstStrategy,
        start=START, end=END, capital=1_000_000,
    )

    assert len(res["nav"]) > 0
    assert len(res["trades_detail"]) > 0
    assert res["trades_detail"][0]["side"] == "buy"


def test_run_event_backtest_short_interval_raises(panel):
    from core.event_engine import Bar

    class NoopStrategy(EventStrategy):
        def on_bar(self, ctx: Context, bar: Bar) -> None:
            return None

    # 少于 5 个交易日（因子尚未形成，am20 全 NaN 会丢列），命中 T<5 保护
    short = panel[panel["date"] < pd.Timestamp("2024-01-06")]
    with pytest.raises(ValueError):
        run_event_backtest(
            panel=short, codes=CODES, strategy_class=NoopStrategy,
            start="2024-01-02", end="2024-01-10", capital=1_000_000,
        )
