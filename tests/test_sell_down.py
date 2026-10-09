"""Sell-down-to-target（执行层双向调仓）单测。

覆盖：
- 股票 executor 部分减仓：整手向下取整数学、卖出价/费用链
  （slippage/spread/impact/sell_cost 复用现有卖出路径）、容差带内不动、
  差额不足一手不动；
- 双向调仓的买入段：已持仓按"目标市值−当前市值"增量补足（不是整额
  预算），超额但在容差带内的标的买 0（不做负买入，Lead 点名边界）；
- 开关关 = 恒等：默认不传参与显式 False 逐位一致，退出名单清仓等旧
  路径不受影响；
- 拒单：停牌/跌停复用现有卖出拒单模板；
- 卖出现金同日可被买入段使用（新进入标的用释放现金建仓）；
- 与 vol targeting 组合：vol_scale < 1 时仓位真实下降；
- 基金申赎 executor 对称支持：按金额差额部分赎回（0.01 份精度向下）、
  增量申购、开关关恒等、差额不足最小份额精度不动；
- 配置默认值 + run_backtest 门面签名 + 引擎端到端（开关关恒等、
  vol targeting 组合下出现 status="rebalanced" 减仓成交且净值改变）。
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from core.engine.config import BacktestConfig, run_backtest
from core.execution import FundNavExecutionAdapter, StockExecutionAdapter
from core.selection import SelectionPolicy
from conftest import CODES, END, START


# ── 构造辅助 ──────────────────────────────────────────────────────────


def _stock_adapter(codes, opens, *, valid_open=None, limit_down=None,
                   limit_up=None, slippage_bps=0.0, spread_bps=0.0,
                   buy_cost=0.0, sell_cost=0.0, impact_coef=0.0,
                   lot_size=100):
    """两日（信号日+执行日）最小股票 executor：高流动性、无冲击。"""
    opens = np.asarray(opens, dtype=float)
    T, K = opens.shape
    if valid_open is None:
        valid_open = np.isfinite(opens) & (opens > 0)
    return StockExecutionAdapter(
        codes=codes, open_mat=opens, valid_open=valid_open,
        am20_mat=np.full((T, K), 1e9), turnover_mat=np.full((T, K), 1.0),
        limit_up=limit_up, limit_down=limit_down,
        dates=pd.bdate_range("2024-06-03", periods=T),
        buy_cost=buy_cost, sell_cost=sell_cost, lot_size=lot_size,
        slippage_bps=slippage_bps, max_participation=0.0,
        spread_bps=spread_bps, min_commission=0.0, impact_coef=impact_coef,
    )


def _fund_adapter(codes, opens, **kw):
    """最小基金申赎 executor（净值=open 矩阵，无涨跌停/流动性约束）。"""
    opens = np.asarray(opens, dtype=float)
    T, K = opens.shape
    return FundNavExecutionAdapter(
        codes=codes, open_mat=opens, valid_open=np.isfinite(opens) & (opens > 0),
        am20_mat=np.full((T, K), 1e9), turnover_mat=np.full((T, K), 1.0),
        limit_up=None, limit_down=None,
        dates=pd.bdate_range("2024-06-03", periods=T),
        buy_cost=0.0, sell_cost=0.0, lot_size=100,
        slippage_bps=0.0, max_participation=0.0, **kw)


def _exec(adapter, cash, positions, targets, chosen_list, pv,
          amount_threshold=0.0, sig=0, exec_idx=1, **kw):
    return adapter.execute_targets(
        cash, positions, targets, chosen_list, pv, amount_threshold,
        sig, exec_idx, **kw)


# ── 股票 executor：部分减仓数学 ───────────────────────────────────────


def test_sell_down_lot_floor_math_and_cost_chain():
    # 持仓 50_000 股 @10（权重 0.5）→ 目标 0.25：差额 250_000 元 = 25_000 股
    # = 250 手整，全卖；卖出价走 slippage/spread/impact/sell_cost 链。
    ad = _stock_adapter(["000001"], [[10.0], [10.0]],
                        slippage_bps=10.0, spread_bps=2.0, sell_cost=0.001,
                        impact_coef=0.1)
    res = _exec(ad, 100_000.0, {0: 50_000}, {0: 0.25}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.01)
    # 整手向下：25_000 股恰为整手
    assert res.positions[0] == 25_000
    # 价格链：10*(1-10bp-2bp)*(1-impact)，impact=0.1*0.02*sqrt(25万/1e9)
    px = 10.0 * (1 - 10.0 / 1e4 - 2.0 / 1e4)
    px *= 1 - 0.1 * 0.02 * np.sqrt((25_000 * px) / 1e9)
    amount = 25_000 * px
    fee = amount * 0.001
    assert res.cash == pytest.approx(100_000.0 + amount - fee)
    assert res.sell_amount == pytest.approx(amount)
    assert res.sold_codes == ["000001"]
    (trade,) = res.trades_detail
    assert trade["side"] == "sell" and trade["status"] == "rebalanced"
    assert trade["reason"] == "sell_down"
    assert trade["shares"] == pytest.approx(25_000.0)
    assert trade["price"] == pytest.approx(px)
    assert trade["fee"] == pytest.approx(fee)
    # 买入段增量预算：目标 250_000 − 剩余持仓 250_000 = 0 → 不买回
    assert res.bought_codes == []
    assert len(res.trades_detail) == 1
    assert res.rejections == []


def test_excess_below_one_lot_not_sold():
    # 差额 500 元 = 50 股 < 1 手 → 整手向下取整后不动
    ad = _stock_adapter(["000001"], [[10.0], [10.0]])
    res = _exec(ad, 100_000.0, {0: 50_000}, {0: 0.4995}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.0)
    assert res.positions[0] == 50_000
    assert res.cash == 100_000.0
    assert res.trades_detail == []


def test_band_tolerance_boundary_inclusive():
    # 超额权重恰为 0.10：band=0.10 → 容差含边界不卖；band 略小 → 卖
    ad = _stock_adapter(["000001"], [[10.0], [10.0]])
    res_in = _exec(ad, 0.0, {0: 50_000}, {0: 0.40}, [0], 1_000_000.0,
                   sell_down_to_target=True, weight_band=0.10)
    assert res_in.positions[0] == 50_000
    assert res_in.trades_detail == []
    res_out = _exec(ad, 0.0, {0: 50_000}, {0: 0.40}, [0], 1_000_000.0,
                    sell_down_to_target=True, weight_band=0.0999)
    assert res_out.positions[0] == 40_000  # 差额 100_000 元 = 10_000 股整手


def test_negative_budget_buys_zero_when_above_target_within_band():
    # Lead 点名边界：当前市值高于目标市值但差额在容差带内 → 不卖出，
    # 买入段增量预算为负 → 买 0（既不买入也不产生负金额成交）。
    ad = _stock_adapter(["000001"], [[10.0], [10.0]])
    res = _exec(ad, 100_000.0, {0: 50_000}, {0: 0.42}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.15)
    assert res.positions[0] == 50_000          # 0.08 超额 ≤ 0.15 容差 → 不卖
    assert res.cash == 100_000.0               # 无卖出进账
    assert res.trades_detail == []             # 无任何成交
    assert res.rejections == []                # 增量预算 ≤ 0 → 静默买 0
    assert res.bought_codes == []


def test_top_up_toward_target_when_below_target():
    # 低于目标：增量预算 = 目标市值 − 当前市值，受现金与整手限制补足
    ad = _stock_adapter(["000001"], [[10.0], [10.0]])
    res = _exec(ad, 100_000.0, {0: 10_000}, {0: 0.25}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.05)
    # 不卖（当前 0.1 < 目标 0.25）；预算 150_000 → 想买 150 手但现金只够 100 手
    assert res.positions[0] == 20_000
    assert res.cash == 0.0
    assert res.bought_codes == ["000001"]


def test_sell_cash_funds_same_day_buy_of_new_name():
    # 卖出释放的现金同日供买入段：新进入标的用释放现金建仓
    ad = _stock_adapter(["000001", "000002"], [[10.0, 20.0], [10.0, 20.0]])
    res = _exec(ad, 0.0, {0: 50_000}, {0: 0.2, 1: 0.2}, [0, 1], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.05)
    # k=0：卖 30_000 股 → 剩 20_000 股（0.2）；现金 300_000
    # k=0 买入段：预算 200_000 − 持仓 200_000 = 0 → 不买回
    # k=1 新建仓：预算 200_000 @20 → 10_000 股
    assert res.positions[0] == 20_000
    assert res.positions[1] == 10_000
    assert res.cash == pytest.approx(100_000.0)
    assert res.sold_codes == ["000001"]
    assert res.bought_codes == ["000002"]


def test_suspended_position_rejects_partial_sell():
    opens = np.array([[10.0], [np.nan]])
    ad = _stock_adapter(["000001"], opens)
    res = _exec(ad, 100_000.0, {0: 50_000}, {0: 0.25}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.01)
    assert res.positions[0] == 50_000
    sell_rej = [r for r in res.rejections if r["side"] == "sell"]
    assert sell_rej and sell_rej[0]["status"] == "rejected"
    assert sell_rej[0]["reason"] == "停牌/无开盘价"
    assert res.trades_detail == []


def test_limit_down_position_rejects_partial_sell():
    ad = _stock_adapter(["000001"], [[10.0], [10.0]],
                        limit_down=np.array([[False], [True]]))
    res = _exec(ad, 100_000.0, {0: 50_000}, {0: 0.25}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.01)
    assert res.positions[0] == 50_000
    sell_rej = [r for r in res.rejections if r["side"] == "sell"]
    assert sell_rej and sell_rej[0]["reason"] == "跌停卖不出"
    assert res.trades_detail == []


# ── 开关关 = 恒等 ─────────────────────────────────────────────────────


def test_flag_off_identity_and_legacy_full_budget_buy():
    # 开关关：名单内持仓不减仓；买入段维持旧口径（整额目标预算、受现金
    # 限制）——用精确数值锚定旧行为，保证默认路径逐位一致。
    ad = _stock_adapter(["000001"], [[10.0], [10.0]])
    res_default = _exec(ad, 100_000.0, {0: 50_000}, {0: 0.25}, [0], 1_000_000.0)
    res_explicit = _exec(ad, 100_000.0, {0: 50_000}, {0: 0.25}, [0], 1_000_000.0,
                         sell_down_to_target=False)
    for res in (res_default, res_explicit):
        assert res.trades_detail == [{
            "date": ad.dates[1], "signal_date": ad.dates[0],
            "code": "000001", "side": "buy", "shares": 10_000.0,
            "price": 10.0, "fee": 0.0, "amount": 100_000.0, "status": "filled",
        }]
        assert res.positions == {0: 60_000}   # 旧口径：现金整额买回同一持仓
        assert res.cash == 0.0
        assert res.sell_amount == 0.0
        assert not any(t.get("status") == "rebalanced" for t in res.trades_detail)


def test_exit_list_liquidation_unchanged_when_flag_on():
    # 退出名单清仓路径与开关无关：开关开时清仓行为不变
    ad = _stock_adapter(["000001"], [[10.0], [10.0]])
    res = _exec(ad, 100_000.0, {0: 50_000}, {}, [], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.01)
    assert res.positions == {}
    assert res.cash == pytest.approx(600_000.0)
    (trade,) = res.trades_detail
    assert trade["side"] == "sell" and trade["status"] == "filled"


# ── 与 vol targeting 组合 ─────────────────────────────────────────────


def test_vol_targeting_scaled_target_really_reduces_position():
    # vol scale = clip(0.20/0.40, lo, hi) = 0.5：等权 0.5 的名单缩到 0.25。
    # 开关关：仓位降不下来（结构性封顶）；开关开：仓位真实降到目标。
    policy = SelectionPolicy(vol_target_annual=0.20)
    scale = policy.vol_scale_for(0.40)
    assert scale == pytest.approx(0.5)
    scaled_target = 0.5 * scale
    ad = _stock_adapter(["000001"], [[10.0], [10.0]])
    off = _exec(ad, 0.0, {0: 50_000}, {0: scaled_target}, [0], 1_000_000.0)
    assert off.positions[0] == 50_000          # 开关关：仍 0.5 权重
    on = _exec(ad, 0.0, {0: 50_000}, {0: scaled_target}, [0], 1_000_000.0,
               sell_down_to_target=True, weight_band=0.05)
    assert on.positions[0] == 25_000           # 0.5 → 0.25，风险承诺真降
    assert on.cash == pytest.approx(250_000.0)
    assert any(t.get("status") == "rebalanced" for t in on.trades_detail)
    # 买入段不再把释放的现金买回同一持仓
    assert on.bought_codes == []


# ── 基金申赎 executor：按金额对称支持 ─────────────────────────────────


def test_fund_partial_redeem_math():
    # 持有 500_000 份 @净值1（权重 0.5）→ 目标 0.2：赎回差额 300_000 份。
    # 持有 1 天 → 赎回费 1.5%（阶梯表首档）。
    ad = _fund_adapter(["000001"], [[1.0], [1.0]])
    ad._lots[0] = [(500_000.0, 0)]
    res = _exec(ad, 0.0, {0: 500_000.0}, {0: 0.2}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.05)
    assert res.positions[0] == pytest.approx(200_000.0)
    assert res.cash == pytest.approx(300_000.0 * (1 - 0.015))
    assert res.sell_amount == pytest.approx(300_000.0)
    (trade,) = res.trades_detail
    assert trade["side"] == "sell" and trade["status"] == "rebalanced"
    assert trade["shares"] == pytest.approx(300_000.0)
    assert trade["fee"] == pytest.approx(4_500.0)
    # 增量申购：目标 200_000 − 持仓 200_000 = 0 → 不买回
    assert res.bought_codes == []


def test_fund_flag_off_identity():
    ad = _fund_adapter(["000001"], [[1.0], [1.0]])
    ad._lots[0] = [(500_000.0, 0)]
    res = _exec(ad, 0.0, {0: 500_000.0}, {0: 0.2}, [0], 1_000_000.0)
    assert res.positions == {0: 500_000.0}     # 名单内不减仓
    assert res.cash == 0.0
    assert res.trades_detail == []
    # 旧口径：整额预算受现金限制 → 现金不足拒单
    assert any(r["reason"] == "现金不足/预算过小" for r in res.rejections)
    assert not any(t.get("status") == "rebalanced" for t in res.trades_detail)


def test_fund_excess_below_min_precision_not_redeemed():
    # 差额 0.009 元 < 0.01 份精度 → 向下取整后 0 份 → 不动
    ad = _fund_adapter(["000001"], [[1.0], [1.0]])
    ad._lots[0] = [(200_000.009, 0)]
    res = _exec(ad, 0.0, {0: 200_000.009}, {0: 0.2}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.0)
    assert res.positions[0] == pytest.approx(200_000.009)
    assert res.trades_detail == []
    assert res.cash == 0.0


def test_fund_no_nav_rejects_partial_redeem():
    opens = np.array([[1.0], [np.nan]])
    ad = _fund_adapter(["000001"], opens)
    ad._lots[0] = [(500_000.0, 0)]
    res = _exec(ad, 0.0, {0: 500_000.0}, {0: 0.2}, [0], 1_000_000.0,
                sell_down_to_target=True, weight_band=0.05)
    assert res.positions == {0: 500_000.0}
    sell_rej = [r for r in res.rejections if r["side"] == "sell"]
    assert sell_rej and sell_rej[0]["reason"] == "无净值"


# ── 配置字段与门面透传 ────────────────────────────────────────────────


def test_config_defaults_off():
    cfg = BacktestConfig(panel=None, codes=[], factor="x", ascending=False,
                         start="2024-01-01", end="2024-03-01",
                         capital=1e6, top_n=2)
    assert cfg.sell_down_to_target is False    # 默认关
    assert cfg.weight_band == 0.15


def test_run_backtest_facade_accepts_sell_down_params():
    params = inspect.signature(run_backtest).parameters
    assert params["sell_down_to_target"].default is False
    assert params["weight_band"].default == 0.15


# ── 引擎端到端 ────────────────────────────────────────────────────────


def test_run_backtest_flag_off_bitwise_identity(panel):
    common = dict(panel=panel, codes=CODES, factor="mom20", ascending=False,
                  start=START, end=END, capital=1_000_000, top_n=2,
                  freq="monthly", warmup_days=400)
    base = run_backtest(**common)
    explicit_off = run_backtest(**common, sell_down_to_target=False,
                                weight_band=0.15)
    assert np.array_equal(base["nav"].to_numpy(), explicit_off["nav"].to_numpy())


def test_run_backtest_sell_down_with_vol_targeting(panel):
    # vol targeting（scale=lo=0.3，每只目标权重 0.15）+ sell-down：
    # 日频调仓 + band=0（任何向上漂移即修剪）确定性触发名单内减仓，
    # 出现 status="rebalanced" 成交，且与开关关的净值不同。
    common = dict(panel=panel, codes=CODES, factor="mom20", ascending=False,
                  start=START, end=END, capital=1_000_000, top_n=2,
                  freq="daily", warmup_days=400, vol_target_annual=1e-4)
    off = run_backtest(**common)
    on = run_backtest(**common, sell_down_to_target=True, weight_band=0.0)
    rebalanced = [t for t in on["trades_detail"] if t.get("status") == "rebalanced"]
    assert rebalanced                          # 出现部分减仓成交
    assert all(t["side"] == "sell" for t in rebalanced)
    assert not np.allclose(off["nav"].to_numpy(), on["nav"].to_numpy())
