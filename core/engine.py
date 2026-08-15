from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_metrics, drawdown_series


def build_factor_frames(close: pd.DataFrame, am20: pd.DataFrame,
                        turn20: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mom20 = close.pct_change(20, fill_method=None)
    mom60 = close.pct_change(60, fill_method=None)
    vol20 = close.pct_change(fill_method=None).rolling(20).std().reindex_like(am20)
    composite = am20.rank(axis=1) + vol20.rank(axis=1)
    return {
        "mom20": mom20,
        "mom60": mom60,
        "vol20": vol20,
        "am20": am20,
        "turn20": turn20,
        "composite": composite,
    }


def run_backtest(
    panel: pd.DataFrame,
    codes: list[str],
    factor: str,
    ascending: bool,
    start: str,
    end: str,
    capital: float,
    top_n: int,
    freq: str = "monthly",
    buy_cost: float = 0.0008,
    sell_cost: float = 0.0013,
    amount_q: float = 0.3,
    affordable: bool = True,
    lot_size: int = 100,
    warmup_days: int | None = None,
) -> dict:
    """事件驱动回测：T+1、一手 100 股、费用、可承载性过滤。

    策略只需提供「每个信号日的因子得分」，引擎负责月度/周度调仓、
    停牌继承、买卖成本与净值计算。

    warmup_days: 因子预热天数。短窗口回测时（如只看近半年），动量/波动类
    因子在窗口起点没有足够历史，会用 start 前 warmup_days 个自然日的数据
    计算因子，但净值仍从 start 开始输出。None 表示不预热（窗口即计算区间）。
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    calc_start = (start_ts - pd.Timedelta(days=warmup_days)
                  if warmup_days and warmup_days > 0 else start_ts)
    sub = panel[panel["code"].isin(codes)].copy()
    sub = sub[(sub["date"] >= calc_start) & (sub["date"] <= end_ts)]
    if sub.empty:
        raise ValueError("所选区间/股票池内没有数据")

    cal = pd.DatetimeIndex(sorted(sub["date"].unique()))

    def pivot(col: str) -> pd.DataFrame:
        # pivot_table 会丢弃全 NaN 行（如 am20 前 20 个交易日），需统一回交易日历
        return sub.pivot_table(index="date", columns="code", values=col,
                               aggfunc="last").reindex(cal).sort_index()

    close = pivot("close")
    open_ = pivot("open")
    turnover = pivot("turnover")
    am20 = pivot("am20")
    turn20 = pivot("turn20")
    codes_used = close.columns.tolist()
    factors = build_factor_frames(close, am20, turn20)
    fmat = factors[factor].values

    close_mat = close.values
    open_mat = open_.values
    turn_mat = turnover.values
    am20_mat = am20.values
    valid_close = ~np.isnan(close_mat)
    valid_open = ~np.isnan(open_mat)

    dates = close.index
    T, K = close.shape
    if T < 5:
        raise ValueError("数据区间太短")

    open_ff = open_.ffill()
    o2o = open_ff.pct_change().values  # t>=1
    o2o = np.nan_to_num(o2o, nan=0.0)

    if freq == "monthly":
        signal_idx = [i - 1 for i in range(1, T) if dates[i].month != dates[i - 1].month]
    else:
        signal_idx = [i for i, d in enumerate(dates) if d.weekday() == 4]
    exec_dates = [i + 1 for i in signal_idx if i + 1 < T]
    start_idx = int(np.argmax(dates >= start_ts)) if (dates >= start_ts).any() else 0
    # 预热模式：窗口起点不继承预热段持仓，也不在窗口起点当天调仓，
    # 与"全量算因子、窗口起点从零开始"的旧脚本语义一致。
    exec_set = ({i for i in exec_dates if i != start_idx}
                if warmup_days else set(exec_dates))

    nav = np.ones(T)
    bench = np.ones(T)
    hold = np.zeros(K)
    holdings_history = []
    trades: list[dict] = []
    last_chosen = []

    for t in range(1, T):
        prev = t - 1
        if t == start_idx and warmup_days:
            # 窗口起点：持仓清零、净值归 1，只输出预热段之后的净值
            hold = np.zeros(K)
            nav[t] = 1.0
            bench[t] = 1.0
            holdings_history.append(hold.copy())
            continue
        rr = o2o[t]
        if hold.sum() > 0:
            raw = float((rr * hold).sum() / hold.sum())
            hold = hold * (1.0 + rr) / (1.0 + raw)
        else:
            raw = 0.0

        elig = valid_close[prev] & valid_open[prev] & valid_open[t]
        bench_ret = float(np.nanmean(np.where(elig, rr, np.nan))) if elig.any() else 0.0

        if t in exec_set:
            sig = prev
            am_vals = am20_mat[sig]
            finite = am_vals[~np.isnan(am_vals)]
            am_thr = np.nanquantile(am_vals, amount_q) if finite.size else np.nan
            valid = (valid_close[sig] & valid_open[t]
                     & ~np.isnan(fmat[sig])
                     & ~np.isnan(turn_mat[sig])
                     & ~np.isnan(am20_mat[sig])
                     & (am20_mat[sig] >= am_thr)
                     & (turn_mat[sig] > 0))
            cand = np.where(valid)[0]
            cant_sell = np.where((hold > 0) & ~valid_open[t])[0]
            new_hold = np.zeros(K)
            if len(cand) > 0:
                scores = fmat[sig, cand]
                if affordable:
                    per_budget = capital / top_n
                    prices = close_mat[sig, cand]
                    afford = (prices * lot_size) <= per_budget
                    if afford.any():
                        cand = cand[afford]
                        scores = scores[afford]
                    else:
                        cand = np.array([], dtype=int)
                        scores = np.array([])
                if len(cand) > 0:
                    order = np.argsort(scores, kind="mergesort")
                    if ascending:
                        chosen = order[:top_n]
                    else:
                        chosen = order[-top_n:][::-1]
                    chosen = cand[chosen]
                    remain = 1.0 - hold[cant_sell].sum() if len(cant_sell) else 1.0
                    new_hold[chosen] = remain / len(chosen) if len(chosen) else 0.0
                    last_chosen = [int(c) for c in chosen]
            new_hold[cant_sell] = hold[cant_sell]

            buy = float(np.maximum(new_hold - hold, 0).sum())
            sell = float(np.maximum(hold - new_hold, 0).sum())
            turn = float(np.abs(new_hold - hold).sum() / 2.0)
            raw = raw - buy * buy_cost - sell * sell_cost
            sold_codes = [codes_used[k] for k in np.where((hold > 0) & (new_hold <= hold))[0] if hold[k] > 0]
            bought_codes = [codes_used[k] for k in np.where(new_hold > hold)[0]]
            trades.append({
                "date": dates[t],
                "signal_date": dates[sig],
                "num_hold": int((new_hold > 0).sum()),
                "turnover": turn,
                "bought": ",".join(bought_codes[:12]),
                "sold": ",".join(sold_codes[:12]),
            })
            hold = new_hold

        nav[t] = nav[t - 1] * (1.0 + raw)
        bench[t] = bench[t - 1] * (1.0 + bench_ret)
        holdings_history.append(hold.copy())

    exec_in_out = sorted(exec_set)
    last_signal_date = dates[exec_in_out[-1] - 1] if exec_in_out else None
    if start_idx > 0:
        # 预热段只用于因子计算，净值/成交从 start 开始输出
        nav = nav[start_idx:]
        bench = bench[start_idx:]
        dates_out = dates[start_idx:]
        trades = [t for t in trades if t["date"] >= start_ts]
        holdings_history = holdings_history[start_idx - 1:] if start_idx > 0 else holdings_history
    else:
        dates_out = dates

    nav_s = pd.Series(nav, index=dates_out, name="nav")
    bench_s = pd.Series(bench, index=dates_out, name="bench")
    trades_df = pd.DataFrame(trades)

    last_hold = pd.Series(hold, index=codes_used)
    last_holdings = last_hold[last_hold > 0].sort_values(ascending=False)
    last_price = close.iloc[-1]
    holdings_df = pd.DataFrame({
        "code": last_holdings.index,
        "weight": last_holdings.values,
        "price": [last_price.get(c, np.nan) for c in last_holdings.index],
    })
    holdings_df["market_value"] = holdings_df["weight"] * nav_s.iloc[-1] * capital
    holdings_df["weight_pct"] = holdings_df["weight"] * 100

    return {
        "nav": nav_s,
        "bench": bench_s,
        "drawdown": drawdown_series(nav_s),
        "metrics": compute_metrics(nav_s),
        "bench_metrics": compute_metrics(bench_s),
        "trades": trades_df,
        "holdings": holdings_df,
        "last_signal_date": last_signal_date,
        "capital": capital,
        "dates": dates_out,
        "last_chosen": last_chosen,
    }


def latest_signals(panel: pd.DataFrame, codes: list[str], factor: str,
                   ascending: bool, top_n: int = 10) -> pd.DataFrame:
    sub = panel[panel["code"].isin(codes)].copy()

    cal = pd.DatetimeIndex(sorted(sub["date"].unique()))

    def pivot(col: str) -> pd.DataFrame:
        return sub.pivot_table(index="date", columns="code", values=col,
                               aggfunc="last").reindex(cal).sort_index()

    close = pivot("close")
    am20 = pivot("am20")
    turn20 = pivot("turn20")
    turnover = pivot("turnover")
    factors = build_factor_frames(close, am20, turn20)
    last_date = close.index[-1]
    row = factors[factor].iloc[-1]
    am_row = am20.iloc[-1]
    turn_row = turnover.iloc[-1]
    close_row = close.iloc[-1]

    cand = row.dropna()
    valid = (am_row[cand.index].notna() & (turn_row[cand.index] > 0)
             & close_row[cand.index].notna())
    cand = cand[valid].sort_values(ascending=ascending)
    top = cand.tail(top_n) if not ascending else cand.head(top_n)

    out = pd.DataFrame({
        "code": top.index,
        "score": top.values,
        "close": [close_row.get(c, np.nan) for c in top.index],
        "turnover": [turn_row.get(c, np.nan) for c in top.index],
    }).reset_index(drop=True)
    return out, last_date
