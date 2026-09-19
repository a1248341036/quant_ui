"""P1-3 engine_gate buffer zone + no-trade band 门禁。

硬保证：
- buffer_ratio=0.0 + no_trade_band=0.0（默认）时，run_backtest 输出与不传参数一致
  （向后兼容，核心引擎回归测试）；
- buffer_ratio=0.5 时，run_backtest 正常运行不报错，trades 换手降低或持平；
- no_trade_band=0.15 时，run_backtest 正常运行，trades 换手降低或持平；
- EngineGateCriteria 默认 buffer_ratio=0.5 + no_trade_band=0.15（spec §4.3）；
- run_engine_gate 透传 buffer_ratio/no_trade_band 到 run_backtest。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import CODES, END, START
from core.engine import run_backtest
from alphaagent.factor.mining.delivery.delivery_criteria import EngineGateCriteria


# ── 向后兼容：默认参数 ──


def test_run_backtest_default_params_no_buffer_band(panel):
    """buffer_ratio=0.0 + no_trade_band=0.0 与不传参数输出一致。"""
    res_ref = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
    )
    res_def = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
        buffer_ratio=0.0, no_trade_band=0.0,
    )
    # nav 逐位一致
    pd.testing.assert_series_equal(res_ref["nav"], res_def["nav"])
    # trades 数量一致
    assert len(res_ref["trades"]) == len(res_def["trades"])


# ── buffer_ratio 透传 + 不报错 ──


def test_run_backtest_buffer_ratio_runs(panel):
    """buffer_ratio=0.5 时 run_backtest 正常运行。"""
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
        buffer_ratio=0.5,
    )
    assert res["nav"].notna().all()
    assert len(res["trades"]) > 0


def test_run_backtest_no_trade_band_runs(panel):
    """no_trade_band=0.15 时 run_backtest 正常运行。"""
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
        no_trade_band=0.15,
    )
    assert res["nav"].notna().all()


def test_run_backtest_buffer_plus_band_runs(panel):
    """buffer_ratio=0.5 + no_trade_band=0.15 组合正常运行。"""
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
        buffer_ratio=0.5, no_trade_band=0.15,
    )
    assert res["nav"].notna().all()


# ── buffer/band 降低换手 ──


def test_buffer_band_reduces_turnover(panel):
    """buffer+band 组合的换手 <= 原始换手（4 只股票 top_n=2 效果有限，但不应增加）。"""
    res_raw = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
    )
    res_buf = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
        buffer_ratio=0.5, no_trade_band=0.15,
    )
    raw_turn = float(res_raw["trades"]["turnover"].sum())
    buf_turn = float(res_buf["trades"]["turnover"].sum())
    # buffer/band 不应增加换手（允许持平，因为 4 只股票 buffer 区很小）
    assert buf_turn <= raw_turn + 1e-9, \
        f"buffer/band should not increase turnover: buf={buf_turn} > raw={raw_turn}"


# ── no_trade_band=1.0 时换手降为 0（首日建仓除外）──


def test_no_trade_band_large_zero_turnover(panel):
    """no_trade_band=1.0 时建仓后所有调仓都在 band 内，换手降为 0。"""
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
        no_trade_band=1.0,
    )
    trades = res["trades"]
    # 找到第一次建仓（num_hold>0 且 turnover>0），之后所有调仓换手应为 0
    build_idx = None
    for i, row in trades.iterrows():
        if row["num_hold"] > 0 and row["turnover"] > 0:
            build_idx = i
            break
    if build_idx is not None and build_idx < len(trades) - 1:
        subsequent_turn = float(trades["turnover"].iloc[build_idx + 1:].sum())
        assert subsequent_turn == pytest.approx(0.0, abs=1e-9), \
            f"band=1.0 should zero turnover after first build: {subsequent_turn}"


# ── EngineGateCriteria 默认值 ──


def test_engine_gate_criteria_defaults():
    """EngineGateCriteria 默认 buffer_ratio=0.5 + no_trade_band=0.15（spec §4.3）。"""
    eg = EngineGateCriteria()
    assert eg.buffer_ratio == 0.5
    assert eg.no_trade_band == 0.15


def test_engine_gate_criteria_dict_includes_new_fields():
    """engine_gate_dict 包含 buffer_ratio/no_trade_band。"""
    from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria
    dc = DeliveryCriteria()
    eg_dict = dc.engine_gate_dict()
    assert "buffer_ratio" in eg_dict
    assert "no_trade_band" in eg_dict
    assert eg_dict["buffer_ratio"] == 0.5
    assert eg_dict["no_trade_band"] == 0.15


# ── run_engine_gate 透传 ──


def test_run_engine_gate_passes_buffer_band(panel):
    """run_engine_gate 透传 buffer_ratio/no_trade_band 到 run_backtest（不报错）。"""
    from alphaagent.factor.mining.delivery.engine_gate import run_engine_gate

    # run_engine_gate 要求 MultiIndex (datetime, instrument) panel
    mi_panel = panel.copy()
    mi_panel["datetime"] = pd.to_datetime(mi_panel["date"].values)
    mi_panel["instrument"] = mi_panel["code"]
    mi_panel = mi_panel.drop(columns=["date", "code"]).set_index(["datetime", "instrument"])
    # amount 千元 → 元（与 alpha_panel_to_engine_frame stock 口径一致）
    mi_panel["amount"] = mi_panel["amount"] * 1000.0
    # turnover 百分数 → 比例
    mi_panel["turnover_rate"] = mi_panel["turnover"] / 100.0

    # 用 close 的 20 日动量作 factor_values（与 panel 行序对齐）
    close = mi_panel["close"]
    mom20 = close.groupby(level="instrument").transform(lambda s: s.pct_change(20))
    factor_values = mom20.to_numpy(dtype=np.float64)

    res = run_engine_gate(
        mi_panel,
        factor_values,
        val_start=START,
        val_end=END,
        direction=1,
        policy={
            "enabled": True,
            "selection_mode": "top_n",
            "top_n": 2,
            "freq": "monthly",
            "capital": 1_000_000,
            "min_excess_annual": -1.0,  # 放宽门槛避免误拒
            "min_excess_sharpe": -1.0,
            "max_drawdown": 1.0,
            "min_daily_overlap": 0.0,
            "min_invested_ratio": 0.0,
        },
    )
    # 应正常运行（不因 buffer/band 报错）
    assert "passed" in res
    assert "metrics" in res
