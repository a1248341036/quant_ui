from __future__ import annotations

import pytest

from core.engine import latest_signals, run_backtest
from conftest import CODES, END, START


def test_run_backtest_smoke(panel):
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
    )

    assert {"nav", "bench", "drawdown", "metrics", "trades", "holdings"} <= set(res)
    nav = res["nav"]
    assert len(nav) > 0
    assert nav.notna().all()
    assert nav.iloc[0] == pytest.approx(1.0, abs=0.05)
    assert res["metrics"]["总收益"] is not None
    assert {"总收益", "年化收益", "年化波动", "夏普", "最大回撤"} <= set(res["metrics"])


def test_run_backtest_weekly_and_composite(panel):
    res = run_backtest(
        panel=panel, codes=CODES, factor="composite", ascending=False,
        start=START, end=END, capital=500_000, top_n=3, freq="weekly",
        factor_weights={"mom20": 1.0, "vol20": -0.5},
    )
    assert len(res["nav"]) > 0
    assert res["nav"].notna().all()


def test_run_backtest_unknown_codes_raises(panel):
    with pytest.raises(ValueError):
        run_backtest(
            panel=panel, codes=["999999"], factor="mom20", ascending=False,
            start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
        )


def test_latest_signals_smoke(panel):
    signals, last_date = latest_signals(panel, CODES, "mom20", False, top_n=2)

    assert last_date is not None
    assert len(signals) == 2
    assert {"code", "score", "close", "turnover"} <= set(signals.columns)
