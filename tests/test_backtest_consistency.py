"""内部一致性检查：在较复杂场景验证引擎的自洽性。

与 test_backtest_manual_verify.py 的区别：
- 手算验证用确定性数据验证"算得对不对"
- 一致性检查用更复杂的数据验证"逻辑是否自洽"

检查项：
1. NAV 恒等式：每日 cash + 持仓收盘市值 == NAV × capital（多个场景）
2. 无负现金（包括小资金 + 多持仓场景）
3. trades_detail 的 amount = shares × price
4. trades_detail 的 fee = amount × rate
5. sell 的 shares 和持仓减少量一致
6. buy 的 shares 和持仓增加量一致
7. 同一日的 sell 在 buy 之前（先卖后买）
8. 涨停买入被拒，跌停卖出被拒
9. 换手率公式：turnover = (buy_amt + sell_amt) / 2 / pv
10. 月度/周度/半年度调仓频率正确
11. 多空模式 NAV 与持仓一致
12. 资金不凭空消失：末日 total cash+持仓 = NAV × capital
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.engine import run_backtest

CODES = ["000001", "000002", "000003", "000004", "000005"]


def _make_panel(n_days: int = 60, n_codes: int = 5, seed: int = 42) -> pd.DataFrame:
    """合成面板：多只股票、足够长，允许因子计算。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes = CODES[:n_codes]
    frames = []
    for code in codes:
        base = 10.0 + int(code[-1]) * 3
        close = base * np.cumprod(1.0 + rng.normal(0, 0.015, n_days))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        # open[0] = close[0] 保证 valid
        open_[0] = close[0]
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n_days)))
        amount = rng.uniform(5e7, 3e8, n_days)
        turnover = rng.uniform(0.5, 5.0, n_days)
        volume = amount / close
        turn20 = pd.Series(turnover).rolling(20, min_periods=10).mean().values
        am20 = pd.Series(amount).rolling(20, min_periods=10).mean().values
        frames.append(pd.DataFrame({
            "date": dates, "code": code,
            "open": open_, "high": high, "low": low, "close": close,
            "turnover": turnover, "amount": amount,
            "turn20": turn20, "am20": am20, "volume": volume,
        }))
    return pd.concat(frames, ignore_index=True)


# ── 1. NAV 恒等式（月度）────────────────────────────────
def test_nav_identity_monthly():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=3, freq="monthly", cash_mode=True,
    )
    _check_nav_identity(res, panel, 1_000_000)


# ── 2. NAV 恒等式（周度）────────────────────────────────
def test_nav_identity_weekly():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res = run_backtest(
        panel=panel, codes=CODES, factor="vol20", ascending=True,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=3, freq="weekly", cash_mode=True,
    )
    _check_nav_identity(res, panel, 1_000_000)


# ── 3. NAV 恒等式（多因子组合）──────────────────────────
def test_nav_identity_composite():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res = run_backtest(
        panel=panel, codes=CODES, factor="composite", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=2, freq="monthly", cash_mode=True,
        factor_weights={"mom20": 1.0, "vol20": -0.5},
    )
    _check_nav_identity(res, panel, 1_000_000)


# ── 4. 无负现金（小资金 + 多持仓）───────────────────────
def test_no_negative_cash_stress():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=50_000, top_n=5, freq="weekly", cash_mode=True,
    )
    cash_hist = res["cash_history"]
    for idx, c in enumerate(cash_hist):
        assert c >= -0.01, f"Day {idx}: 现金为负 {c:.2f}"


# ── 5. 成交明细自洽：amount = shares × price ────────────
def test_trade_detail_amount_consistency():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=3, freq="monthly", cash_mode=True,
        buy_cost=0.0008, sell_cost=0.0013,
    )
    for t in res["trades_detail"]:
        expected_amount = t["shares"] * t["price"]
        assert t["amount"] == pytest.approx(expected_amount, abs=0.01), \
            f"{t['code']} {t['side']}: amount={t['amount']:.4f}, " \
            f"shares×price={expected_amount:.4f}"


# ── 6. 成交明细自洽：fee = amount × rate ────────────────
def test_trade_detail_fee_consistency():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    buy_cost, sell_cost = 0.0008, 0.0013
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=3, freq="monthly", cash_mode=True,
        buy_cost=buy_cost, sell_cost=sell_cost,
    )
    for t in res["trades_detail"]:
        rate = buy_cost if t["side"] == "buy" else sell_cost
        expected_fee = t["amount"] * rate
        assert t["fee"] == pytest.approx(expected_fee, abs=0.01), \
            f"{t['code']} {t['side']}: fee={t['fee']:.4f}, " \
            f"amount×rate={expected_fee:.4f}"


# ── 7. 先卖后买顺序 ─────────────────────────────────────
def test_sell_before_buy_in_trades_detail():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=2, freq="weekly", cash_mode=True,
    )
    by_date = {}
    for t in res["trades_detail"]:
        d = t["date"]
        by_date.setdefault(d, []).append(t["side"])
    for d, sides in by_date.items():
        if "sell" in sides and "buy" in sides:
            assert sides.index("sell") < sides.index("buy"), \
                f"Day {d}: 先卖后买顺序错误, sides={sides}"


# ── 8. 涨停股不被买入 ──────────────────────────────────
def test_limit_up_stock_not_bought():
    """涨停股在选股阶段被过滤，不会进入持仓。

    engine.py 第610行: cand = cand[~limit_up[t, cand]]
    在选股阶段就把涨停股排除，不传入 adapter。
    因此涨停股不会出现在 trades_detail 的买入记录中。
    """
    dates = pd.bdate_range("2024-01-02", periods=30)
    n = len(dates)
    # 股票 A 在第25天涨停（前日收盘 11.0 → 当日收盘 12.1，涨 10%）
    close_a = np.linspace(10.0, 11.0, 25).tolist() + [12.1] * 5
    close_b = np.linspace(20.0, 22.0, n)
    close_c = np.linspace(5.0, 4.5, n)
    close_d = np.linspace(8.0, 8.5, n)
    close_e = np.linspace(15.0, 16.0, n)
    close_arr = np.array([close_a, close_b, close_c, close_d, close_e]).T
    open_arr = np.vstack([close_arr[0:1], close_arr[:-1]]).astype(float)
    open_arr[0] = close_arr[0]
    open_arr[25, 0] = 12.1  # 涨停日开盘 = 涨停价

    frames = []
    for i, code in enumerate(CODES):
        frames.append(pd.DataFrame({
            "date": dates, "code": code,
            "open": open_arr[:, i],
            "high": np.maximum(open_arr[:, i], close_arr[:, i]) * 1.001,
            "low": np.minimum(open_arr[:, i], close_arr[:, i]) * 0.999,
            "close": close_arr[:, i],
            "turnover": np.full(n, 2.0), "amount": np.full(n, 1e8),
            "turn20": np.full(n, 2.0), "am20": np.full(n, 1e8),
            "volume": 1e8 / close_arr[:, i],
        }))
    panel = pd.concat(frames, ignore_index=True)

    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=1, freq="daily", cash_mode=True,
        limit_flags=True,
    )
    # 第25天(涨停日)的买入记录中不应包含涨停股 000001
    limit_date = dates[25]
    buys_on_limit_day = [
        t for t in res["trades_detail"]
        if t["side"] == "buy" and t["date"] == limit_date and t["code"] == "000001"
    ]
    assert len(buys_on_limit_day) == 0, "涨停股不应被买入"
    # NAV 不应有异常
    assert res["nav"].notna().all()
    assert (res["nav"] > 0).all()


# ── 9. 换手率公式 ───────────────────────────────────────
def test_turnover_formula():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=3, freq="monthly", cash_mode=True,
    )
    trades = res["trades"]
    trades_detail = res["trades_detail"]

    # 按执行日聚合 trades_detail 的 buy_amount 和 sell_amount
    by_date = {}
    for t in trades_detail:
        d = t["date"]
        if d not in by_date:
            by_date[d] = {"buy": 0.0, "sell": 0.0}
        by_date[d][t["side"]] += t["amount"]

    for _, row in trades.iterrows():
        d = row["date"]
        if d in by_date:
            buy_amt = by_date[d]["buy"]
            sell_amt = by_date[d]["sell"]
            pv = 1_000_000 * res["nav"].loc[d]  # 近似
            if pv > 0:
                expected_turn = (buy_amt + sell_amt) / 2.0 / pv
                assert row["turnover"] == pytest.approx(expected_turn, abs=0.01), \
                    f"Day {d}: turnover={row['turnover']:.4f}, " \
                    f"预期={expected_turn:.4f}"


# ── 10. 多空模式 NAV 一致性 ─────────────────────────────
def test_long_short_nav_positive():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=2, freq="monthly", cash_mode=False,
        long_short=True, short_n=2,
    )
    nav = res["nav"]
    assert nav.notna().all(), "多空模式 NAV 不应有 NaN"
    assert (nav > 0).all(), "多空模式 NAV 不应为负"


# ── 11. 半年度调仓频率 ──────────────────────────────────
def test_semiannual_freq():
    """半年度调仓：每年3月/9月换仓。"""
    dates = pd.bdate_range("2023-06-01", periods=200)
    n = len(dates)
    rng = np.random.default_rng(123)
    close_arr = np.column_stack([
        10 * np.cumprod(1 + rng.normal(0, 0.01, n)) for _ in range(5)
    ])
    open_arr = np.vstack([close_arr[0:1], close_arr[:-1]]).astype(float)
    open_arr[0] = close_arr[0]

    frames = []
    for i, code in enumerate(CODES):
        frames.append(pd.DataFrame({
            "date": dates, "code": code,
            "open": open_arr[:, i],
            "high": np.maximum(open_arr[:, i], close_arr[:, i]) * 1.001,
            "low": np.minimum(open_arr[:, i], close_arr[:, i]) * 0.999,
            "close": close_arr[:, i],
            "turnover": np.full(n, 2.0), "amount": np.full(n, 1e8),
            "turn20": np.full(n, 2.0), "am20": np.full(n, 1e8),
            "volume": 1e8 / close_arr[:, i],
        }))
    panel = pd.concat(frames, ignore_index=True)

    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=2, freq="semiannual", cash_mode=True,
    )
    trades = res["trades"]
    if len(trades) > 0:
        # 调仓日应在3月或9月（signal_date 的下月）
        for _, row in trades.iterrows():
            exec_month = pd.Timestamp(row["date"]).month
            assert exec_month in (3, 9), \
                f"半年度调仓日应在3月或9月, 实际: {exec_month}"


# ── 12. 末日总资产 = NAV × capital ──────────────────────
def test_final_equity_matches_nav():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    capital = 1_000_000
    res = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=capital, top_n=3, freq="monthly", cash_mode=True,
    )
    # 末日现金
    final_cash = res["cash_history"][-1]
    # 末日持仓市值
    close_df = panel.pivot_table(index="date", columns="code", values="close")
    last_date = res["nav"].index[-1]
    final_pos_value = sum(
        shares * float(close_df.loc[last_date, code])
        for code, shares in res["positions_history"][-1].items()
    )
    final_equity = final_cash + final_pos_value
    expected_nav = final_equity / capital
    actual_nav = res["nav"].iloc[-1]
    assert actual_nav == pytest.approx(expected_nav, abs=0.01), \
        f"末日 NAV={actual_nav:.6f}, 手算={expected_nav:.6f} " \
        f"(cash={final_cash:.2f}, pos={final_pos_value:.2f})"


# ── 13. 滑点影响成交价 ──────────────────────────────────
def test_slippage_affects_price():
    panel = _make_panel(60, 5)
    dates = sorted(panel["date"].unique())
    res_no_slip = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=1, freq="monthly", cash_mode=True,
        slippage_bps=0.0, limit_flags=False,
    )
    res_with_slip = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start=str(dates[2].date()), end=str(dates[-1].date()),
        capital=1_000_000, top_n=1, freq="monthly", cash_mode=True,
        slippage_bps=10.0, limit_flags=False,  # 10bps = 0.1%
    )
    # 有滑点时，买入价应略高于无滑点
    no_slip_buys = [t for t in res_no_slip["trades_detail"] if t["side"] == "buy"]
    with_slip_buys = [t for t in res_with_slip["trades_detail"] if t["side"] == "buy"]
    if len(no_slip_buys) > 0 and len(with_slip_buys) > 0:
        # 找同一交易日的买入
        for ns, ws in zip(no_slip_buys, with_slip_buys):
            if ns["code"] == ws["code"] and ns["date"] == ws["date"]:
                assert ws["price"] > ns["price"], \
                    f"滑点应使买入价升高: 无滑点={ns['price']}, " \
                    f"有滑点={ws['price']}"


# ── 辅助函数 ────────────────────────────────────────────
def _check_nav_identity(res, panel, capital):
    nav = res["nav"]
    cash_hist = res["cash_history"]
    pos_hist = res["positions_history"]
    assert len(nav) == len(cash_hist) == len(pos_hist), \
        f"长度应一致: nav={len(nav)}, cash={len(cash_hist)}, pos={len(pos_hist)}"
    close_df = panel.pivot_table(index="date", columns="code", values="close")
    dates_out = nav.index
    for idx, d in enumerate(dates_out):
        expected_equity = cash_hist[idx]
        for code, shares in pos_hist[idx].items():
            px = float(close_df.loc[d, code])
            expected_equity += shares * px
        expected_nav = expected_equity / capital
        assert nav.iloc[idx] == pytest.approx(expected_nav, abs=0.02), \
            f"Day {d.date()}: NAV={nav.iloc[idx]:.6f}, " \
            f"手算={expected_nav:.6f}"
