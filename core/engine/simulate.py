"""阶段 3 主循环：现金/整手模型与旧权重连续模型（含多空）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..execution import ETFExecutionAdapter, FundNavExecutionAdapter, StockExecutionAdapter
from ..selection import PortfolioBuilder


def _simulate(cfg, prep: dict, fctx: dict) -> dict:
    """阶段 3 主循环:现金/整手模型与旧权重连续模型(含多空)。"""
    codes_used = prep["codes_used"]
    profile = prep["profile"]
    close = prep["close"]
    open_ = prep["open_"]
    turnover = prep["turnover"]
    am20 = prep["am20"]
    high = prep["high"]
    low = prep["low"]
    dates = prep["dates"]
    close_mat = prep["close_mat"]
    close_fill_mat = prep["close_fill_mat"]
    open_mat = prep["open_mat"]
    turn_mat = prep["turn_mat"]
    am20_mat = prep["am20_mat"]
    valid_close = prep["valid_close"]
    valid_open = prep["valid_open"]
    o2o = prep["o2o"]
    T = prep["T"]
    K = prep["K"]
    start_idx = prep["start_idx"]
    exec_set = prep["exec_set"]
    limit_up = prep["limit_up"]
    limit_down = prep["limit_down"]
    use_cash = prep["use_cash"]
    stop_mat = prep["stop_mat"]
    adx_mat = prep["adx_mat"]
    selection_policy = prep["selection_policy"]
    portfolio_builder = prep["portfolio_builder"]
    capital = prep["capital"]
    top_n = prep["top_n"]
    warmup_days = prep["warmup_days"]
    amount_q = prep["amount_q"]
    affordable = prep["affordable"]
    lot_size = prep["lot_size"]
    buy_cost = prep["buy_cost"]
    sell_cost = prep["sell_cost"]
    slippage_bps = prep["slippage_bps"]
    max_participation = prep["max_participation"]
    spread_bps = prep["spread_bps"]
    min_commission = prep["min_commission"]
    impact_coef = prep["impact_coef"]
    impact_vol = prep["impact_vol"]
    max_weight = prep["max_weight"]
    share_classes = prep["share_classes"]
    long_short = prep["long_short"]
    short_n = prep["short_n"]
    short_cost_rate = prep["short_cost_rate"]
    selection_mode = prep["selection_mode"]
    selection_pct = prep["selection_pct"]
    min_positions = prep["min_positions"]
    max_positions = prep["max_positions"]
    ascending = cfg.ascending
    adx_filter = prep["adx_filter"]
    fmat = fctx["fmat"]

    nav = np.ones(T)
    bench = np.ones(T)
    hold = np.zeros(K)
    holdings_history = []
    trades: list[dict] = []
    trades_detail: list[dict] = []
    cash_history: list[float] = []
    positions_history: list[dict[str, float]] = []
    rejections: list[dict] = []
    last_chosen = []

    def _record_holdings(day: int, cash_: float, pos: dict[int, float]) -> float:
        eq = float(cash_)
        for k, sh in pos.items():
            if np.isfinite(close_fill_mat[day, k]):
                eq += sh * float(close_fill_mat[day, k])
        w = np.zeros(K)
        for k, sh in pos.items():
            px = float(close_fill_mat[day, k]) if np.isfinite(close_fill_mat[day, k]) else 0.0
            w[k] = sh * px / eq if eq > 0 else 0.0
        holdings_history.append(w)
        cash_history.append(float(cash_))
        positions_history.append({codes_used[k]: float(sh)
                                  for k, sh in pos.items() if abs(sh) > 1e-9})
        return eq

    if use_cash:
        cash = float(capital)
        positions: dict[int, float] = {}
        if profile.asset_type == "etf":
            adapter_cls = ETFExecutionAdapter
        elif profile.asset_type == "fund_nav":
            adapter_cls = FundNavExecutionAdapter
        else:
            adapter_cls = StockExecutionAdapter
        adapter_kwargs = dict(
            codes=codes_used, open_mat=open_mat, valid_open=valid_open,
            am20_mat=am20_mat, turnover_mat=turn_mat,
            limit_up=limit_up, limit_down=limit_down, dates=dates,
            buy_cost=buy_cost, sell_cost=sell_cost, lot_size=lot_size,
            slippage_bps=slippage_bps,
            max_participation=max_participation,
            spread_bps=spread_bps,
            min_commission=min_commission,
            impact_coef=impact_coef, impact_vol=impact_vol,
        )
        if adapter_cls is FundNavExecutionAdapter:
            # 传入 A/C 类信息，供申购费率判断
            if share_classes:
                adapter_kwargs["share_classes"] = share_classes
        execution_adapter = adapter_cls(**adapter_kwargs)

        def _portfolio_value(day: int) -> float:
            v = cash
            for k, sh in positions.items():
                if np.isfinite(close_fill_mat[day, k]):
                    v += sh * float(close_fill_mat[day, k])
            return v

        for t in range(1, T):
            prev = t - 1
            if use_cash and stop_mat is not None:
                # Chandelier/ATR 追踪止损：收盘跌破止损线，次日开盘卖出
                stopped = execution_adapter.execute_stop_losses(
                    cash, positions, close_mat, valid_close, stop_mat, prev, t,
                )
                cash, positions = stopped.cash, stopped.positions
                trades_detail.extend(stopped.trades_detail)
            if t == start_idx:
                # 窗口起点：清空预热段持仓，与模拟盘/旧口径一致从零开始
                positions.clear()
                cash = float(capital)
                bench[t] = 1.0
            elig = valid_close[prev] & valid_open[prev] & valid_open[t]
            bench_ret = float(np.nanmean(np.where(elig, o2o[t], np.nan))) if elig.any() else 0.0

            if t in exec_set:
                sig = prev
                am_vals = am20_mat[sig]
                finite = am_vals[~np.isnan(am_vals)]
                am_thr = np.nanquantile(am_vals, amount_q) if finite.size else np.nan
                adx_ok = (adx_mat[sig] >= adx_filter) if adx_mat is not None else None
                valid = (valid_close[sig] & valid_open[t]
                         & ~np.isnan(fmat[sig])
                         & ~np.isnan(turn_mat[sig])
                         & ~np.isnan(am20_mat[sig])
                         & (am20_mat[sig] >= am_thr)
                         & (turn_mat[sig] > 0))
                if adx_ok is not None:
                    valid = valid & adx_ok
                cand = np.where(valid)[0]
                if limit_up is not None and len(cand):
                    cand = cand[~limit_up[t, cand]]
                targets: dict[int, float] = {}
                chosen_list: list[int] = []
                if len(cand) > 0:
                    scores = fmat[sig, cand]
                    market_adx = (float(np.nanmedian(adx_mat[sig]))
                                  if (selection_policy.regime_adx is not None
                                      and adx_mat is not None) else None)
                    chosen_list, sel_targets = portfolio_builder.build_targets(
                        selection_policy, cand, scores, market_adx)
                    targets.update(sel_targets)
                    if chosen_list:
                        last_chosen = [codes_used[k] for k in chosen_list]

                pv = _portfolio_value(sig)
                signal_d = dates[sig].date().isoformat()
                exec_d = dates[t].date().isoformat()
                bought_codes: list[str] = []
                sold_codes: list[str] = []
                buy_amt = 0.0
                sell_amt = 0.0

                executed = execution_adapter.execute_targets(
                    cash, positions, targets, chosen_list, pv, am_thr, sig, t,
                    max_weight=max_weight,
                )
                cash, positions = executed.cash, executed.positions
                buy_amt, sell_amt = executed.buy_amount, executed.sell_amount
                bought_codes = executed.bought_codes
                sold_codes = executed.sold_codes
                trades_detail.extend(executed.trades_detail)
                rejections.extend(executed.rejections)

                turn = (buy_amt + sell_amt) / 2.0 / pv if pv else 0.0
                trades.append({
                    "date": dates[t],
                    "signal_date": dates[sig],
                    "num_hold": int(sum(1 for v in positions.values() if v > 0)),
                    "turnover": float(turn),
                    "bought": ",".join(bought_codes[:12]),
                    "sold": ",".join(sold_codes[:12]),
                })

            eq = _record_holdings(t, cash, positions)
            nav[t] = eq / capital
            if t != start_idx:
                bench[t] = bench[t - 1] * (1.0 + bench_ret)

        hold = np.zeros(K)
        final_eq = float(cash)
        for k, sh in positions.items():
            px = float(close_fill_mat[-1, k]) if np.isfinite(close_fill_mat[-1, k]) else 0.0
            final_eq += sh * px
        for k, sh in positions.items():
            px = float(close_fill_mat[-1, k]) if np.isfinite(close_fill_mat[-1, k]) else 0.0
            hold[k] = sh * px / final_eq if final_eq > 0 else 0.0
    else:
        for t in range(1, T):
            prev = t - 1
            if t == start_idx and warmup_days:
                # 窗口起点：持仓清零、净值归 1，只输出预热段之后的净值
                hold = np.zeros(K)
                nav[t] = 1.0
                bench[t] = 1.0
                holdings_history.append(hold.copy())
                cash_history.append(float(capital))
                positions_history.append({})
                continue
            rr = o2o[t]
            if long_short:
                # 多空：long/short 两腿按各自的加权收益分别再平衡，
                # 保持多头名义=1、空头名义=1、净敞口=0，避免共用分母导致敞口漂移。
                long_mask = hold > 0
                short_mask = hold < 0
                long_gross = float(hold[long_mask].sum()) if long_mask.any() else 0.0
                short_notional = float(-hold[short_mask].sum()) if short_mask.any() else 0.0
                long_ret = (float((rr[long_mask] * hold[long_mask]).sum()) / long_gross
                            if long_gross > 0 else 0.0)
                # 空头腿：标的加权收益 u_short，空头腿 P&L = -u_short。
                # 再平衡时多头按 (1+long_ret) 缩放，空头按 (1+u_short) 缩放，
                # 保持两腿名义各 1、净敞口为 0。
                short_base = -hold[short_mask] if short_mask.any() else np.zeros(0)
                u_short = (float((rr[short_mask] * short_base).sum()) / short_notional
                           if short_notional > 0 else 0.0)
                raw = long_gross * long_ret - short_notional * u_short
                new_hold = np.zeros(K)
                if long_gross > 0 and (1.0 + long_ret) > 0:
                    new_hold[long_mask] = (hold[long_mask] * (1.0 + rr[long_mask])
                                           / (1.0 + long_ret))
                if short_notional > 0 and (1.0 + u_short) > 0:
                    new_hold[short_mask] = (hold[short_mask] * (1.0 + rr[short_mask])
                                            / (1.0 + u_short))
                hold = new_hold
                raw -= short_notional * short_cost_rate / 252.0
            else:
                gross = float(np.abs(hold).sum())
                if gross > 0:
                    raw = float((rr * hold).sum()) / hold.sum()
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
                adx_ok = (adx_mat[sig] >= adx_filter) if adx_mat is not None else None
                valid = (valid_close[sig] & valid_open[t]
                         & ~np.isnan(fmat[sig])
                         & ~np.isnan(turn_mat[sig])
                         & ~np.isnan(am20_mat[sig])
                         & (am20_mat[sig] >= am_thr)
                         & (turn_mat[sig] > 0))
                if adx_ok is not None:
                    valid = valid & adx_ok
                cand = np.where(valid)[0]
                cant_sell = np.where((hold != 0) & ~valid_open[t])[0]
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
                        gated = selection_policy.gate(cand, scores)
                        if len(gated) > 0:
                            scores = scores[np.isin(cand, gated)]
                            cand = gated
                    if len(cand) > 0:
                        long_n = PortfolioBuilder.selection_count(
                            len(cand), top_n, selection_mode, selection_pct,
                            min_positions, max_positions)
                        short_count = short_n or long_n
                        min_cand = long_n + short_count if long_short else long_n
                        ordered = portfolio_builder.rank_select(
                            cand, scores, ascending, min_cand, "top_n", 0.10,
                            1, None, limit_count=min_cand,
                        )
                        ordered_sel = np.asarray(ordered, dtype=int)
                        if len(ordered_sel) >= min_cand:
                            chosen = ordered_sel[:long_n]
                            long_stuck = float(np.maximum(hold[cant_sell], 0).sum()) if len(cant_sell) else 0.0
                            remain_long = 1.0 - long_stuck
                            new_hold[chosen] = remain_long / len(chosen)
                            last_chosen = [codes_used[int(c)] for c in chosen]
                            if long_short:
                                shorts = ordered_sel[-short_count:]
                                short_stuck = float(np.maximum(-hold[cant_sell], 0).sum()) if len(cant_sell) else 0.0
                                remain_short = 1.0 - short_stuck
                                new_hold[shorts] = -remain_short / len(shorts)
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
                    "num_hold": int((new_hold != 0).sum() if long_short
                                    else (new_hold > 0).sum()),
                    "turnover": turn,
                    "bought": ",".join(bought_codes[:12]),
                    "sold": ",".join(sold_codes[:12]),
                })
                hold = new_hold

            nav[t] = nav[t - 1] * (1.0 + raw)
            bench[t] = bench[t - 1] * (1.0 + bench_ret)
            holdings_history.append(hold.copy())
            cash_history.append(float(capital * nav[t]))
            positions_history.append({codes_used[k]: float(v)
                                      for k, v in enumerate(hold)
                                      if abs(v) > 1e-9})

    return {
        "nav": nav, "bench": bench, "trades": trades,
        "trades_detail": trades_detail, "holdings_history": holdings_history,
        "cash_history": cash_history, "positions_history": positions_history,
        "rejections": rejections, "last_chosen": last_chosen, "hold": hold,
    }
