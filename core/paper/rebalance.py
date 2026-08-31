"""调仓 / 撮合 / 日结主流程 / 事件策略重放 / 因子策略重放 / 组合展示。"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .storage import (
    DEFAULT_RISK, _load_module, _ensure_columns,
)
from .account import (
    list_accounts, get_account, _clear_account_state, _update_account_start,
)
from .details import (
    account_orders, account_trades, account_positions, account_equity,
    _add_order, _add_trade, _set_position, _add_snapshot, _add_event,
    _update_account_dates, _name_map,
)
from ..limit import build_limit_flags
from .. import trading_config


def _rebalance_due(acc: dict, exec_ts: pd.Timestamp) -> bool:
    last = acc.get("last_rebalance_date")
    if not last:
        return True
    last_ts = pd.Timestamp(last)
    freq = str(acc.get("freq", "monthly"))
    if freq == "daily":
        return exec_ts.date() != last_ts.date()
    if freq == "weekly":
        return exec_ts.isocalendar()[:2] != last_ts.isocalendar()[:2]
    if freq == "semiannual":
        return ((exec_ts.year, exec_ts.month) != (last_ts.year, last_ts.month)
                and exec_ts.month in (3, 9))
    return (exec_ts.year, exec_ts.month) != (last_ts.year, last_ts.month)


def _compute_targets(acc: dict, panel: pd.DataFrame, codes: list[str],
                     exec_ts: pd.Timestamp):
    """信号日收盘生成目标权重 {code: pct}。返回 (targets, signal_date, error)。"""
    try:
        from ..engine import latest_signals
        sub = panel[panel["date"] < exec_ts].copy()
        if sub.empty:
            return None, None, "信号日无数据"
        risk = {**DEFAULT_RISK, **(acc.get("risk_config") or {})}
        sig, sig_date = latest_signals(
            sub, codes, acc["factor"], acc["ascending"],
            top_n=acc["top_n"],
            long_short=False,  # 日级模拟盘先做纯多头
            adx_filter=risk.get("adx_filter"),
        )
        if sig is None or len(sig) == 0:
            return None, sig_date, "信号为空"
        if "side" in sig.columns:
            sig = sig[sig["side"] != "空"]
        targets: dict[str, float] = {}
        n = int(min(len(sig), acc["top_n"]))
        if n <= 0:
            return None, sig_date, "目标持仓为空"
        for code in sig["code"].head(n):
            targets[str(code).zfill(6)] = 1.0 / n
        return targets, sig_date, None
    except Exception as exc:
        return None, None, f"信号计算失败: {type(exc).__name__}: {exc}"


def _execute_rebalance(
    acc: dict,
    panel: pd.DataFrame,
    exec_ts: pd.Timestamp,
    signal_date: pd.Timestamp,
    targets: dict[str, float],
    positions: dict[str, dict],
    cash: float,
) -> dict:
    """在 exec_ts 开盘执行目标权重，返回 {positions, cash, orders, events}。"""
    risk = {**DEFAULT_RISK, **(acc.get("risk_config") or {})}
    buy_cost = float(risk.get("buy_cost", trading_config.BUY_COST))
    sell_cost = float(risk.get("sell_cost", trading_config.SELL_COST))
    lot = int(risk.get("lot_size", 100))
    amount_q = float(risk.get("amount_q", 0.2))
    max_weight = float(risk.get("max_weight", 0.5))

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    if exec_ts not in dates:
        return {"positions": positions, "cash": cash, "orders": [], "events": []}
    t = dates.get_loc(exec_ts)
    sig = t - 1
    if t == 0:
        return {"positions": positions, "cash": cash, "orders": [], "events": []}

    close = panel.pivot_table(index="date", columns="code", values="close",
                              aggfunc="last", observed=True).reindex(dates).sort_index()
    open_ = panel.pivot_table(index="date", columns="code", values="open",
                              aggfunc="last", observed=True).reindex(dates).sort_index()
    am20 = panel.pivot_table(index="date", columns="code", values="am20",
                             aggfunc="last", observed=True).reindex(dates).sort_index()
    turn = panel.pivot_table(index="date", columns="code", values="turnover",
                             aggfunc="last", observed=True).reindex(dates).sort_index()
    limit_up, limit_down, _, _ = build_limit_flags(close, open_)

    close_row = close.iloc[sig]
    open_row = open_.iloc[t]
    am20_row = am20.iloc[sig]
    turn_row = turn.iloc[sig]
    limit_up_row = limit_up[t]
    limit_down_row = limit_down[t]

    am_vals = am20_row.values
    finite = am_vals[np.isfinite(am_vals)]
    am_thr = np.nanquantile(finite, amount_q) if finite.size else np.nan

    pv = cash
    for code, pos in positions.items():
        px = close_row.get(code, np.nan) if code in close.columns else np.nan
        if np.isfinite(px):
            pv += pos["shares"] * float(px)

    orders: list[dict] = []
    events: list[str] = []
    signal_d = signal_date.date().isoformat()
    exec_d = exec_ts.date().isoformat()

    # ---- 第一步：卖出（先） ----
    for code, pos in list(positions.items()):
        code = str(code).zfill(6)
        if targets.get(code, 0.0) > 0:
            continue
        if code not in close.columns or not np.isfinite(open_row.get(code, np.nan)):
            _add_order(acc["id"], {"code": code, "side": "sell", "target_pct": 0.0,
                                   "signal_date": signal_d, "exec_date": exec_d,
                                   "shares": float(pos["shares"]),
                                   "status": "rejected", "reject_reason": "停牌/无开盘价"})
            orders.append({"code": code, "side": "sell", "status": "rejected",
                           "reason": "停牌/无开盘价"})
            continue
        if limit_down_row[close.columns.get_loc(code)]:
            _add_order(acc["id"], {"code": code, "side": "sell", "target_pct": 0.0,
                                   "signal_date": signal_d, "exec_date": exec_d,
                                   "shares": float(pos["shares"]),
                                   "status": "rejected", "reject_reason": "跌停卖不出"})
            orders.append({"code": code, "side": "sell", "status": "rejected",
                           "reason": "跌停卖不出"})
            continue
        px = float(open_row[code])
        shares = float(pos["shares"])
        amt = shares * px
        fee = amt * sell_cost
        cash += amt - fee
        oid = _add_order(acc["id"], {"code": code, "side": "sell", "target_pct": 0.0,
                                     "signal_date": signal_d, "exec_date": exec_d,
                                     "shares": shares,
                                     "status": "filled", "fill_price": px, "fee": fee})
        _add_trade(acc["id"], {"order_id": oid or None, "exec_date": exec_d,
                               "code": code, "side": "sell", "shares": shares,
                               "price": px, "fee": fee})
        _set_position(acc["id"], code, 0.0, pos["avg_cost"], exec_d)
        positions.pop(code, None)
        orders.append({"code": code, "side": "sell", "status": "filled"})

    # ---- 第二步：买入（后） ----
    for code, pct in targets.items():
        code = str(code).zfill(6)
        pct = max(min(float(pct), max_weight), 0.0)
        if pct <= 0 or code not in close.columns:
            continue
        k = close.columns.get_loc(code)
        if not np.isfinite(open_row.get(code, np.nan)):
            _add_order(acc["id"], {"code": code, "side": "buy", "target_pct": pct,
                                   "signal_date": signal_d, "exec_date": exec_d,
                                   "shares": 0,
                                   "status": "rejected", "reject_reason": "停牌/无开盘价"})
            orders.append({"code": code, "side": "buy", "status": "rejected",
                           "reason": "停牌/无开盘价"})
            continue
        if limit_up_row[k]:
            _add_order(acc["id"], {"code": code, "side": "buy", "target_pct": pct,
                                   "signal_date": signal_d, "exec_date": exec_d,
                                   "shares": 0,
                                   "status": "rejected", "reject_reason": "涨停买不进"})
            orders.append({"code": code, "side": "buy", "status": "rejected",
                           "reason": "涨停买不进"})
            continue
        am = am20_row.get(code, np.nan)
        if not np.isfinite(am) or (np.isfinite(am_thr) and am < am_thr):
            _add_order(acc["id"], {"code": code, "side": "buy", "target_pct": pct,
                                   "signal_date": signal_d, "exec_date": exec_d,
                                   "shares": 0,
                                   "status": "rejected", "reject_reason": "流动性不足(am20分位)"})
            orders.append({"code": code, "side": "buy", "status": "rejected",
                           "reason": "流动性不足(am20分位)"})
            continue
        tv = turn_row.get(code, np.nan)
        if not np.isfinite(tv) or tv <= 0:
            _add_order(acc["id"], {"code": code, "side": "buy", "target_pct": pct,
                                   "signal_date": signal_d, "exec_date": exec_d,
                                   "shares": 0,
                                   "status": "rejected", "reject_reason": "无成交量"})
            orders.append({"code": code, "side": "buy", "status": "rejected",
                           "reason": "无成交量"})
            continue
        px = float(open_row[code])
        budget = pv * pct
        gross = px * (1.0 + buy_cost)
        want_lots = int(budget // gross // lot)
        cash_lots = int(cash // gross // lot)
        lots = min(want_lots, cash_lots)
        if lots <= 0:
            _add_order(acc["id"], {"code": code, "side": "buy", "target_pct": pct,
                                   "signal_date": signal_d, "exec_date": exec_d,
                                   "shares": 0,
                                   "status": "rejected", "reject_reason": "现金不足/预算过小"})
            orders.append({"code": code, "side": "buy", "status": "rejected",
                           "reason": "现金不足/预算过小"})
            continue
        shares = lots * lot
        fee = shares * px * buy_cost
        cost = shares * px + fee
        cash -= cost
        old = positions.get(code)
        old_sh = old["shares"] if old else 0.0
        old_cost = old["avg_cost"] if old else 0.0
        new_sh = old_sh + shares
        new_cost = (old_sh * old_cost + cost) / new_sh if new_sh > 0 else 0.0
        oid = _add_order(acc["id"], {"code": code, "side": "buy", "target_pct": pct,
                                     "signal_date": signal_d, "exec_date": exec_d,
                                     "shares": shares,
                                     "status": "filled", "fill_price": px, "fee": fee})
        _add_trade(acc["id"], {"order_id": oid or None, "exec_date": exec_d,
                               "code": code, "side": "buy", "shares": shares,
                               "price": px, "fee": fee})
        _set_position(acc["id"], code, new_sh, new_cost, exec_d)
        positions[code] = {"account_id": acc["id"], "code": code, "shares": new_sh,
                           "avg_cost": new_cost, "updated_date": exec_d}
        orders.append({"code": code, "side": "buy", "status": "filled"})

    return {"positions": positions, "cash": cash, "orders": orders, "events": events}


# ---------- 日结主流程 ----------

def run_paper_trade(
    panel: pd.DataFrame,
    codes_by_universe: dict[str, list[str]],
    account_id: int | None = None,
    account_ids: list[int] | None = None,
    exec_date: str | None = None,
    dry_run: bool = False,
) -> dict:
    """对启用账户执行一次日级模拟盘。

    panel: 全市场日线面板；codes_by_universe: {universe: codes}。
    exec_date: 目标交易日（默认面板最新交易日）。
    dry_run: 只输出将要执行的订单，不落库。
    """
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    if len(dates) == 0:
        raise ValueError("panel 无交易日")
    exec_ts = pd.Timestamp(exec_date) if exec_date else dates[-1]
    if exec_ts not in dates:
        prev = dates[dates <= exec_ts]
        if len(prev) == 0:
            raise ValueError(f"exec_date {exec_date} 早于数据范围")
        exec_ts = prev[-1]

    accounts = list_accounts()
    if account_id is not None:
        accounts = [a for a in accounts if a["id"] == int(account_id)]
    elif account_ids is not None:
        ids = set(int(x) for x in account_ids)
        accounts = [a for a in accounts if a["id"] in ids]
    if not accounts:
        raise ValueError("未找到可执行账户")

    out: list[dict] = []
    for acc in accounts:
        out.append(_run_one(acc, panel, codes_by_universe, exec_ts, dry_run))
    return {"run_date": exec_ts.date().isoformat(), "accounts": out}


# ---------- 事件策略账户（重放模式） ----------

def _event_bt_start(panel: pd.DataFrame, start_date: str, warmup_days: int) -> str:
    """事件回测从 start_date 前一交易日启动，保证 start_date 当天产生成交。

    event_engine 把 start 参数当天视为预热重置日（跳过交易），
    因此把回测起点前移一个交易日，账户起始日即为正式交易首日。
    """
    start_ts = pd.Timestamp(start_date)
    if warmup_days and warmup_days > 0:
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        prev = dates[dates < start_ts]
        if len(prev):
            return prev[-1].date().isoformat()
    return start_date


def _replay_event_positions(
    aid: int, trades_detail: list[dict], exec_dates: set[str],
    note: str = "事件策略",
) -> dict[str, dict]:
    """按逐笔成交回放持仓（含 avg_cost），并落订单/成交明细。"""
    positions: dict[str, dict] = {}
    for t in trades_detail:
        exec_d = pd.Timestamp(t["date"]).date().isoformat()
        if exec_d not in exec_dates:
            continue
        code = str(t["code"]).zfill(6)
        side = str(t["side"])
        shares = float(t["shares"])
        price = float(t["price"])
        fee = float(t.get("fee", 0.0))
        signal_d = pd.Timestamp(t["signal_date"]).date().isoformat()
        oid = _add_order(aid, {
            "code": code, "side": side, "target_pct": None,
            "signal_date": signal_d, "exec_date": exec_d,
            "shares": shares, "status": "filled",
            "fill_price": price, "fee": fee,
        })
        _add_trade(aid, {
            "order_id": oid or None, "exec_date": exec_d, "code": code,
            "side": side, "shares": shares, "price": price, "fee": fee,
            "note": note,
        })
        old = positions.get(code)
        old_sh = old["shares"] if old else 0.0
        old_cost = old["avg_cost"] if old else 0.0
        if side == "buy":
            new_sh = old_sh + shares
            cost = shares * price + fee
            new_cost = (old_sh * old_cost + cost) / new_sh if new_sh > 0 else 0.0
            positions[code] = {"account_id": aid, "code": code,
                               "shares": new_sh, "avg_cost": new_cost,
                               "updated_date": exec_d}
        elif side == "sell":
            new_sh = old_sh - shares
            if new_sh < -1e-9:
                raise ValueError(f"卖出 {code} 超过持仓，事件策略疑似做空")
            if abs(new_sh) <= 1e-9:
                positions.pop(code, None)
            else:
                positions[code] = {"account_id": aid, "code": code,
                                   "shares": new_sh,
                                   "avg_cost": old_cost if old else 0.0,
                                   "updated_date": exec_d}
    return positions


def _run_one_event(
    acc: dict,
    panel: pd.DataFrame,
    codes_by_universe: dict[str, list[str]],
    exec_ts: pd.Timestamp,
    dry_run: bool,
    base: dict,
) -> dict:
    aid = acc["id"]
    codes = codes_by_universe.get(acc["universe"])
    if not codes:
        return {**base, "processed": "error", "message": f"未知股票池: {acc['universe']}"}

    last = acc.get("last_processed_date")
    if last and pd.Timestamp(last).date() >= exec_ts.date():
        return {**base, "processed": "already", "message": "该交易日已处理"}

    module_path = acc.get("module")
    event_name = acc.get("event_strategy")
    if not module_path or not event_name:
        return {**base, "processed": "error",
                "message": "事件策略账户缺少 module/event_strategy 配置"}
    module_path = str(module_path)
    if not os.path.exists(module_path):
        return {**base, "processed": "error",
                "message": f"代码模块不存在: {module_path}"}
    try:
        mod = _load_module(module_path)
    except Exception as exc:
        return {**base, "processed": "error",
                "message": f"加载代码模块失败: {type(exc).__name__}: {exc}"}
    event_strategies = getattr(mod, "EVENT_STRATEGIES", None) or {}
    strategy_class = event_strategies.get(event_name)
    if strategy_class is None:
        return {**base, "processed": "error",
                "message": f"模块中没有事件策略: {event_name}"}

    risk = {**DEFAULT_RISK, **(acc.get("risk_config") or {})}
    warmup_days = int(risk.get("warmup_days", 400) or 400)
    start_date = acc.get("start_date")
    if not start_date:
        start_date = exec_ts.date().isoformat()
    start_ts = pd.Timestamp(start_date).date()
    if start_ts > exec_ts.date():
        start_ts = exec_ts.date()
        start_date = start_ts.isoformat()

    try:
        from ..event_engine import run_event_backtest
        from ..assets import ETF_PROFILE, STOCK_PROFILE
        is_etf = acc.get("universe") == "ETF"
        res = run_event_backtest(
            panel=panel,
            codes=codes,
            strategy_class=strategy_class,
            start=_event_bt_start(panel, start_date, warmup_days),
            end=exec_ts.date().isoformat(),
            capital=float(acc["capital"]),
            buy_cost=float(risk.get("buy_cost", trading_config.BUY_COST)),
            sell_cost=float(risk.get("sell_cost", trading_config.SELL_COST)),
            lot_size=int(risk.get("lot_size", trading_config.LOT_SIZE)),
            warmup_days=warmup_days,
            amount_q=float(risk.get("amount_q", trading_config.AMOUNT_Q)),
            limit_flags=not is_etf,
            slippage_bps=float(risk.get("slippage_bps", trading_config.SLIPPAGE_BPS) or trading_config.SLIPPAGE_BPS),
            max_participation=float(risk.get("max_participation", 0.0) or 0.0),
            short_rate=0.0,
            execution_profile=ETF_PROFILE if is_etf else STOCK_PROFILE,
        )
    except Exception as exc:
        return {**base, "processed": "error",
                "message": f"事件回放失败: {type(exc).__name__}: {exc}"}

    trades = [t for t in res.get("trades_detail", [])
              if pd.Timestamp(t["date"]).date() >= start_ts]

    # 模拟盘先限制多头：任何时点出现空头持仓都暂不支持落库
    short_codes: list[str] = []
    for pos_hist in (res.get("positions_history") or []):
        for c, sh in (pos_hist or {}).items():
            if float(sh) < -1e-9:
                short_codes.append(str(c))
    if short_codes:
        return {**base, "processed": "error",
                "message": f"事件策略出现空头持仓 {short_codes[:8]}，模拟盘暂不支持空头"}

    if dry_run:
        exec_dates = {t["date"] for t in trades}
        return {**base, "processed": "ok",
                "rebalanced": bool(trades),
                "orders": len(trades),
                "filled": len(trades),
                "rejected": 0,
                "positions": len({str(t["code"]).zfill(6) for t in trades}),
                "start_date": start_date,
                "message": f"dry-run：事件回放预览（成交 {len(trades)} 笔，"
                           f"涉及执行日 {len(exec_dates)} 天），未落库"}

    # 重放落库：清空旧状态后整体重写，保证状态一致
    _clear_account_state(aid)
    if start_date != acc.get("start_date"):
        _update_account_start(aid, start_date)
        acc["start_date"] = start_date

    exec_dates = {pd.Timestamp(t["date"]).date().isoformat() for t in trades}
    positions = _replay_event_positions(aid, trades, exec_dates)

    capital = float(acc["capital"])
    n_snaps = 0
    dates_out = res.get("dates")
    if dates_out is None:
        dates_out = []
    cash_hist = res.get("cash_history") or []
    nav_s = res.get("nav")
    for i, dt in enumerate(dates_out):
        d = pd.Timestamp(dt).date()
        if d < start_ts:
            continue
        nav_v = float(nav_s.iloc[i]) if nav_s is not None else 1.0
        equity = nav_v * capital
        cash = float(cash_hist[i]) if i < len(cash_hist) else equity
        _add_snapshot(aid, {
            "date": d.isoformat(), "cash": cash,
            "market_value": max(0.0, equity - cash),
            "equity": equity, "pnl": equity - capital,
            "pnl_pct": (equity / capital - 1.0) if capital else 0.0,
        })
        n_snaps += 1

    for code, pos in positions.items():
        _set_position(aid, code, pos["shares"], pos["avg_cost"],
                      pos["updated_date"])

    _add_event(aid, exec_ts.date().isoformat(), "info",
               f"事件策略回放完成：成交 {len(trades)} 笔，"
               f"持仓 {len(positions)} 只，估值 {n_snaps} 条")
    _update_account_dates(aid, exec_ts.date().isoformat(),
                          exec_ts.date().isoformat())

    mv_display = 0.0
    for code, pos in positions.items():
        mv_display += pos["shares"] * _last_close(panel, exec_ts, code)
    return {
        **base, "processed": "ok", "rebalanced": bool(trades),
        "signal_date": None,
        "orders": len(trades), "filled": len(trades), "rejected": 0,
        "cash": round(float(cash_hist[-1]) if cash_hist else capital, 2),
        "market_value": round(mv_display, 2),
        "message": "OK",
    }


def _run_one_factor(
    acc: dict,
    panel: pd.DataFrame,
    codes_by_universe: dict[str, list[str]],
    exec_ts: pd.Timestamp,
    dry_run: bool,
    base: dict,
) -> dict:
    """因子账户：重放 core.engine.run_backtest(cash_mode=True) 的完整结果。

    每次执行都从账户起始日到 exec_date 重算一遍并整体重写状态，
    保证模拟盘与回测的成交/持仓/估值/拒单口径完全一致。
    """
    aid = acc["id"]
    last = acc.get("last_processed_date")
    last_ts = pd.Timestamp(last) if last else None
    if last_ts is not None and last_ts.date() >= exec_ts.date():
        return {**base, "processed": "already", "message": "该交易日已处理"}

    codes = codes_by_universe.get(acc["universe"])
    if not codes:
        return {**base, "processed": "error", "message": f"未知股票池: {acc['universe']}"}

    risk = {**DEFAULT_RISK, **(acc.get("risk_config") or {})}
    warmup_days = int(risk.get("warmup_days", 400) or 400)
    start_date = acc.get("start_date")
    if not start_date:
        dates_all = pd.DatetimeIndex(sorted(panel["date"].unique()))
        start_date = dates_all[0].date().isoformat()
    start_ts = pd.Timestamp(start_date).date()
    if start_ts > exec_ts.date():
        start_ts = exec_ts.date()
        start_date = start_ts.isoformat()

    try:
        from ..engine import run_backtest
        from ..assets import ETF_PROFILE, STOCK_PROFILE
        is_etf = acc.get("universe") == "ETF"
        res = run_backtest(
            panel=panel,
            codes=codes,
            factor=acc["factor"],
            ascending=bool(acc["ascending"]),
            start=start_date,
            end=exec_ts.date().isoformat(),
            capital=float(acc["capital"]),
            top_n=int(acc["top_n"]),
            freq=str(acc.get("freq", "monthly")),
            buy_cost=float(risk.get("buy_cost", trading_config.BUY_COST)),
            sell_cost=float(risk.get("sell_cost", trading_config.SELL_COST)),
            lot_size=int(risk.get("lot_size", trading_config.LOT_SIZE)),
            amount_q=float(risk.get("amount_q", trading_config.AMOUNT_Q)),
            warmup_days=warmup_days,
            cash_mode=True,
            limit_flags=bool(risk.get("limit_flags", True)) and not is_etf,
            slippage_bps=float(risk.get("slippage_bps", trading_config.SLIPPAGE_BPS) or trading_config.SLIPPAGE_BPS),
            max_participation=float(risk.get("max_participation", 0.0) or 0.0),
            spread_bps=(float(risk["spread_bps"]) if risk.get("spread_bps") is not None
                        else None),
            min_commission=(float(risk["min_commission"]) if risk.get("min_commission") is not None
                            else None),
            max_weight=(float(risk["max_weight"])
                        if risk.get("max_weight") is not None else None),
            adx_filter=(float(risk["adx_filter"])
                        if risk.get("adx_filter") is not None else None),
            chandelier_mult=float(risk.get("chandelier_mult", 0.0) or 0.0),
            chandelier_period=int(risk.get("chandelier_period", 22)),
            regime_adx=(float(risk["regime_adx"])
                        if risk.get("regime_adx") is not None else None),
            regime_scale=float(risk.get("regime_scale", 0.5) or 0.5),
            execution_profile=ETF_PROFILE if is_etf else STOCK_PROFILE,
        )
    except Exception as exc:
        return {**base, "processed": "error",
                "message": f"因子回放失败: {type(exc).__name__}: {exc}"}

    trades = [t for t in res.get("trades_detail", [])
              if pd.Timestamp(t["date"]).date() >= start_ts]
    rejects = [r for r in res.get("rejections", [])
               if pd.Timestamp(r["date"]).date() >= start_ts]

    if dry_run:
        exec_d = exec_ts.date().isoformat()
        day_trades = [t for t in trades
                      if pd.Timestamp(t["date"]).date().isoformat() == exec_d]
        day_rejects = [r for r in rejects if str(r.get("date")) == exec_d]
        chosen = res.get("last_chosen") or []
        preview_targets = {str(c).zfill(6): 1.0 / len(chosen)
                           for c in chosen} if chosen else {}
        return {**base, "processed": "ok",
                "rebalanced": bool(day_trades or day_rejects),
                "orders": len(day_trades) + len(day_rejects),
                "filled": len(day_trades),
                "rejected": len(day_rejects),
                "positions": len({str(t["code"]).zfill(6) for t in day_trades
                                  if t["side"] == "buy"}),
                "targets": {k: round(v, 4) for k, v in preview_targets.items()},
                "start_date": start_date,
                "message": f"dry-run：因子回放预览（当日成交 {len(day_trades)} 笔，"
                           f"拒单 {len(day_rejects)} 笔；最近信号 {len(chosen)} 只），未落库"}

    # 重放落库：清空旧状态后整体重写，保证状态与回测引擎一致
    _clear_account_state(aid)
    if start_date != acc.get("start_date"):
        _update_account_start(aid, start_date)
        acc["start_date"] = start_date

    exec_dates = {pd.Timestamp(t["date"]).date().isoformat() for t in trades}
    positions = _replay_event_positions(aid, trades, exec_dates, note="因子策略")

    # 拒单也写入订单表，保持与引擎口径可见
    for r in rejects:
        rdate = pd.Timestamp(r["date"]).date().isoformat()
        signal_d = (pd.Timestamp(r["signal_date"]).date().isoformat()
                    if r.get("signal_date") else rdate)
        _add_order(aid, {
            "code": str(r.get("code", "")).zfill(6),
            "side": r.get("side", "buy"),
            "target_pct": None, "signal_date": signal_d, "exec_date": rdate,
            "shares": 0, "status": "rejected",
            "reject_reason": r.get("reason", ""),
        })

    capital = float(acc["capital"])
    n_snaps = 0
    dates_out = res.get("dates")
    if dates_out is None:
        dates_out = []
    cash_hist = res.get("cash_history") or []
    nav_s = res.get("nav")
    for i, dt in enumerate(dates_out):
        d = pd.Timestamp(dt).date()
        if d < start_ts:
            continue
        nav_v = float(nav_s.iloc[i]) if nav_s is not None else 1.0
        equity = nav_v * capital
        cash = float(cash_hist[i]) if i < len(cash_hist) else equity
        _add_snapshot(aid, {
            "date": d.isoformat(), "cash": cash,
            "market_value": max(0.0, equity - cash),
            "equity": equity, "pnl": equity - capital,
            "pnl_pct": (equity / capital - 1.0) if capital else 0.0,
        })
        n_snaps += 1

    for code, pos in positions.items():
        _set_position(aid, code, pos["shares"], pos["avg_cost"],
                      pos["updated_date"])

    n_filled = len(trades)
    n_rejected = len(rejects)
    _add_event(aid, exec_ts.date().isoformat(), "info",
               f"因子策略回放完成：成交 {n_filled} 笔，拒单 {n_rejected} 笔，"
               f"持仓 {len(positions)} 只，估值 {n_snaps} 条")
    _update_account_dates(aid, exec_ts.date().isoformat(),
                          exec_ts.date().isoformat())

    mv_display = 0.0
    for code, pos in positions.items():
        mv_display += pos["shares"] * _last_close(panel, exec_ts, code)
    return {
        **base, "processed": "ok", "rebalanced": bool(trades),
        "signal_date": None,
        "orders": n_filled + n_rejected, "filled": n_filled,
        "rejected": n_rejected,
        "cash": round(float(cash_hist[-1]) if cash_hist else capital, 2),
        "market_value": round(mv_display, 2),
        "message": "OK",
    }


def _run_one(
    acc: dict,
    panel: pd.DataFrame,
    codes_by_universe: dict[str, list[str]],
    exec_ts: pd.Timestamp,
    dry_run: bool,
) -> dict:
    aid = acc["id"]
    base = {"account_id": aid, "name": acc["name"], "status": acc["status"]}
    if acc["status"] != "active":
        return {**base, "processed": "skipped", "message": "账户已暂停"}
    if str(acc.get("strategy_type") or "factor") == "event":
        return _run_one_event(acc, panel, codes_by_universe, exec_ts,
                              dry_run, base)
    return _run_one_factor(acc, panel, codes_by_universe, exec_ts,
                           dry_run, base)


def _last_close(panel: pd.DataFrame, exec_ts: pd.Timestamp, code: str) -> float:
    sub = panel[(panel["code"] == code) & (panel["date"] <= exec_ts)]
    if sub.empty:
        return 0.0
    return float(sub["close"].iloc[-1])


# ---------- 组合展示 ----------

def enrich_positions(account_id: int, panel: pd.DataFrame) -> list[dict]:
    """持仓 + 最新收盘价 + 市值 + 盈亏。"""
    rows = account_positions(account_id)
    nm = _name_map()
    last = panel[panel["date"] == panel["date"].max()]
    px_map = dict(zip(last["code"].astype(str).str.zfill(6), last["close"]))
    out = []
    for r in rows:
        code = str(r["code"]).zfill(6)
        px = float(px_map.get(code, np.nan))
        mv = r["shares"] * px if np.isfinite(px) else 0.0
        cost = r["shares"] * r["avg_cost"]
        out.append({
            **r, "code": code, "name": nm.get(code, ""),
            "price": px if np.isfinite(px) else None,
            "market_value": mv, "cost": cost,
            "pnl": mv - cost if np.isfinite(px) else None,
        })
    return out


def account_summary(account_id: int, panel: pd.DataFrame) -> dict | None:
    acc = get_account(account_id)
    if acc is None:
        return None
    snaps = account_equity(account_id)
    eq = snaps[-1] if snaps else None
    pos = enrich_positions(account_id, panel)
    return {
        "account": acc,
        "latest": eq,
        "positions": pos,
        "n_orders": len(account_orders(account_id)),
        "n_trades": len(account_trades(account_id)),
    }
