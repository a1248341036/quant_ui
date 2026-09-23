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
    # 执行换手硬门：裸构造为 None（由 DeliveryCriteria 派生填充）
    assert eg.max_avg_daily_turnover is None


def test_engine_gate_turnover_gate_derived():
    """engine_gate 执行换手硬门：spec 层 None=未配置，消费点 fallback 分档值。

    不在构造路径派生（物化的值会破坏 freq 覆盖后的按档 fallback）；
    真实取值链：to_prompt_text gate_max_to / run_engine_gate 缺键回落，
    均走 DeliveryCriteria.turnover_gate_limit（按 engine_gate.freq 分档）。
    """
    from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria
    from alphaagent.factor.mining.research_spec import default_research_spec

    # 未显式配置 → None（裸默认 / from_spec / DEFAULT spec 三层一致）
    assert DeliveryCriteria.defaults().engine_gate.max_avg_daily_turnover is None
    assert (
        DeliveryCriteria.from_spec(default_research_spec("technical"))
        .engine_gate.max_avg_daily_turnover is None
    )
    assert (
        default_research_spec("technical")["delivery_policy"]["production"]
        ["engine_gate"].get("max_avg_daily_turnover") is None
    )
    # engine_gate_dict 不物化 None 键（保持 spec 干净、freq 联动）
    assert "max_avg_daily_turnover" not in DeliveryCriteria.defaults().engine_gate_dict()
    # 显式配置保留
    spec2 = default_research_spec("technical")
    spec2["delivery_policy"]["production"]["engine_gate"]["max_avg_daily_turnover"] = 0.42
    assert DeliveryCriteria.from_spec(spec2).engine_gate.max_avg_daily_turnover == 0.42
    # 消费点 fallback：prompt 渲染按 freq 分档（weekly → 0.65）
    dc = DeliveryCriteria.from_spec(default_research_spec("technical"))
    text = dc.to_prompt_text()
    assert "日均执行换手 <= 0.65" in text


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
    # conftest panel amount 已是元（rng.uniform(5e7, 2e8)），与 main 的
    # alpha_panel_to_engine_frame stock 口径一致（amount_mult=1.0），不再 ×1000。
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


def test_run_engine_gate_does_not_mutate_policy(panel):
    """审计 F-040：缺键回落写回必须走副本，不得污染调用方传入的 policy。"""
    from alphaagent.factor.mining.delivery.engine_gate import run_engine_gate

    mi_panel = panel.copy()
    mi_panel["datetime"] = pd.to_datetime(mi_panel["date"].values)
    mi_panel["instrument"] = mi_panel["code"]
    mi_panel = mi_panel.drop(columns=["date", "code"]).set_index(["datetime", "instrument"])
    mi_panel["turnover_rate"] = mi_panel["turnover"] / 100.0
    close = mi_panel["close"]
    mom20 = close.groupby(level="instrument").transform(lambda s: s.pct_change(20))
    factor_values = mom20.to_numpy(dtype=np.float64)

    policy = {
        "enabled": True, "selection_mode": "top_n", "top_n": 2,
        "freq": "monthly", "capital": 1_000_000,
        "min_excess_annual": -1.0, "min_excess_sharpe": -1.0,
        "max_drawdown": 1.0, "min_daily_overlap": 0.0, "min_invested_ratio": 0.0,
    }
    before = dict(policy)
    run_engine_gate(
        mi_panel, factor_values, val_start=START, val_end=END, direction=1, policy=policy,
    )
    # 调用方 dict 不被就地写入回落默认键（buffer_ratio/no_trade_band/max_avg_daily_turnover）
    assert policy == before, f"policy 被就地修改: {set(policy) - set(before)}"


def test_run_engine_gate_high_turnover_gate(panel):
    """执行换手硬门（2026-09-22）：diag avg_daily_turnover > 门槛 → high_turnover。"""
    from alphaagent.factor.mining.delivery.engine_gate import run_engine_gate

    mi_panel = panel.copy()
    mi_panel["datetime"] = pd.to_datetime(mi_panel["date"].values)
    mi_panel["instrument"] = mi_panel["code"]
    mi_panel = mi_panel.drop(columns=["date", "code"]).set_index(["datetime", "instrument"])
    mi_panel["turnover_rate"] = mi_panel["turnover"] / 100.0
    close = mi_panel["close"]
    mom20 = close.groupby(level="instrument").transform(lambda s: s.pct_change(20))
    factor_values = mom20.to_numpy(dtype=np.float64)

    base_policy = {
        "enabled": True, "selection_mode": "top_n", "top_n": 2,
        "freq": "monthly", "capital": 1_000_000,
        "min_excess_annual": -1.0, "min_excess_sharpe": -1.0,
        "max_drawdown": 1.0, "min_daily_overlap": 0.0, "min_invested_ratio": 0.0,
    }

    # 门槛 0.0：只要有可得的执行换手就必超 → high_turnover
    res_block = run_engine_gate(
        mi_panel, factor_values, val_start=START, val_end=END, direction=1,
        policy={**base_policy, "max_avg_daily_turnover": 0.0},
    )
    assert "high_turnover" in res_block["fail_reasons"], res_block["fail_reasons"]
    assert res_block["thresholds"]["max_avg_daily_turnover"] == 0.0

    # 门槛 9.9：不因换手被拒
    res_pass = run_engine_gate(
        mi_panel, factor_values, val_start=START, val_end=END, direction=1,
        policy={**base_policy, "max_avg_daily_turnover": 9.9},
    )
    assert "high_turnover" not in res_pass["fail_reasons"], res_pass["fail_reasons"]

    # 缺键：回落 defaults 分档（weekly → 0.65），thresholds 记录
    res_default = run_engine_gate(
        mi_panel, factor_values, val_start=START, val_end=END, direction=1,
        policy=dict(base_policy),
    )
    assert res_default["thresholds"]["max_avg_daily_turnover"] == 0.65


# ── buffer zone 严格降换手（review 2026-09-19：原 test_buffer_band_reduces_turnover
#    用 <= 无法检测 selection.py:build_targets 的 buffer bug——chosen⊂buf_zone 恒 True，
#    kept=top-N∩老持仓，没把 top-(N+1~M)∩老持仓拉回。修复后补严格 < 测试。）──


def test_build_targets_buffer_pulls_back_old_in_buf_zone():
    """build_targets buffer zone 单元测试：老持仓跌出 top-N 但在 top-(N+M) 内应被拉回。

    构造确定性场景：
    - 5 只票，top_n=2，buffer_ratio=0.5 → M=1，buf_zone=top-3
    - 期1: scores=[5,4,3,2,1] → top-2={0,1}
    - 期2: scores=[5,3,4,2,1] → rank: 0>2>1>3>4, top-2={0,2}, buf_zone={0,2,1}
      老持仓 1 跌到 rank3，在 buf_zone 内 → 应被拉回，chosen 应含 1
    - 无 buffer 时 chosen=[0,2]（1 被踢出）
    - 有 buffer 时 chosen=[0,1]（1 被拉回，0 也在 top-2 补足）
    """
    from core.selection import PortfolioBuilder, SelectionPolicy

    builder = PortfolioBuilder(codes=[f"c{i}" for i in range(5)])
    policy = SelectionPolicy(
        top_n=2, ascending=False, min_positions=2, max_positions=2,
    )

    # 期1: 建仓 top-2={0,1}
    cand1 = np.array([0, 1, 2, 3, 4])
    sc1 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    chosen1, _ = builder.build_targets(policy, cand1, sc1)
    assert chosen1 == [0, 1], f"期1 top-2 应为 [0,1], got {chosen1}"

    # 期2: 老持仓={0,1}, 票1跌到rank3但在buf_zone top-3内
    cand2 = np.array([0, 1, 2, 3, 4])
    sc2 = np.array([5.0, 3.0, 4.0, 2.0, 1.0])  # rank: 0>2>1>3>4
    chosen2_no_buf, _ = builder.build_targets(policy, cand2, sc2)
    assert chosen2_no_buf == [0, 2], f"期2 无buffer 应为 [0,2], got {chosen2_no_buf}"

    chosen2_buf, _ = builder.build_targets(
        policy, cand2, sc2, buffer_keep={0, 1}, buffer_ratio=0.5,
    )
    # 老持仓 1 在 buf_zone 内应被拉回，0 也在 top-2 内保留
    assert 1 in chosen2_buf, f"buffer 应拉回老持仓 1, got {chosen2_buf}"
    assert 0 in chosen2_buf, f"top-2 的 0 应保留, got {chosen2_buf}"
    assert 2 not in chosen2_buf, f"buffer 拉回 1 后 2 应被挤出, got {chosen2_buf}"
    assert len(chosen2_buf) == 2, f"应保持 top_n=2, got {chosen2_buf}"


def test_buffer_only_strictly_reduces_engine_turnover():
    """buffer_ratio=0.5 + no_trade_band=0.0 时，引擎回测换手应严格 < 原始换手。

    review 2026-09-19：原 test_buffer_band_reduces_turnover 断言 <=（允许持平），
    无法检测 buffer bug（bug 下 buffer 与无 buffer 换手完全一致）。
    本测试用 8 只票 + 构造因子确保老持仓有票跌出 top-N 但在 top-(N+M) 内，
    断言严格 <。
    """
    # 8 只票，top_n=2，buffer_ratio=0.5 → M=1，buf_zone=top-3
    # 构造因子：让 top-2 成员在月度调仓时显著轮换，老持仓常跌到 rank3
    rng = np.random.default_rng(7)
    n_days = 120
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes_local = [f"S{i:03d}" for i in range(8)]
    frames = []
    for c in codes_local:
        base = 10.0 + int(c[-1])
        close = base * np.cumprod(1.0 + rng.normal(0, 0.02, n_days))
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n_days)))
        amount = rng.uniform(1e8, 5e8, n_days)
        turnover = rng.uniform(0.5, 5.0, n_days)
        volume = amount / close
        frames.append(pd.DataFrame({
            "date": dates, "code": c, "open": open_, "high": high, "low": low,
            "close": close, "turnover": turnover, "amount": amount,
            "turn20": pd.Series(turnover).rolling(20, min_periods=15).mean().values,
            "am20": pd.Series(amount).rolling(20, min_periods=15).mean().values,
            "volume": volume,
        }))
    panel_local = pd.concat(frames, ignore_index=True)

    # 用 20 日动量作因子（月度调仓时头部常轮换）
    res_raw = run_backtest(
        panel=panel_local, codes=codes_local, factor="mom20", ascending=False,
        start=dates[0].date().isoformat(), end=dates[-1].date().isoformat(),
        capital=1_000_000, top_n=2, freq="monthly",
    )
    res_buf = run_backtest(
        panel=panel_local, codes=codes_local, factor="mom20", ascending=False,
        start=dates[0].date().isoformat(), end=dates[-1].date().isoformat(),
        capital=1_000_000, top_n=2, freq="monthly",
        buffer_ratio=0.5, no_trade_band=0.0,  # 只开 buffer，不开 band
    )
    raw_turn = float(res_raw["trades"]["turnover"].sum())
    buf_turn = float(res_buf["trades"]["turnover"].sum())
    # 严格 < ：buffer zone 应至少在一个调仓日拉回老持仓，降低换手
    assert buf_turn < raw_turn - 1e-9, (
        f"buffer only should strictly reduce turnover: "
        f"buf={buf_turn} >= raw={raw_turn}"
    )
