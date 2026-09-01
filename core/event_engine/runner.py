from __future__ import annotations

import numpy as np
import pandas as pd

from ..limit import build_limit_flags
from ..metrics import compute_excess_metrics, compute_metrics, drawdown_series
from ..assets import AssetExecutionProfile, STOCK_PROFILE
from .. import trading_config

from .context import Context, Bar, EventStrategy


class BacktestAborted(Exception):
    """外部取消: 由进度回调抛出, 引擎循环立即终止。"""


def _load_st_mask_for(
    cal: pd.DatetimeIndex,
    codes_used: list[str],
    calc_start: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame | None:
    """读取事件回测窗口内逐日 ST 标记，失败时返回 None（降级板块近似）。"""
    try:
        from ..cne_reader import load_st_mask, CneUnavailable
        mask = load_st_mask(
            codes_used,
            start=calc_start.date().isoformat(),
            end=end_ts.date().isoformat(),
        )
        if mask is None or mask.empty:
            return None
        return mask.reindex(index=cal, columns=codes_used)
    except (CneUnavailable, Exception):  # noqa: BLE001 - 降级不打断回测
        return None


def run_event_backtest(
    panel: pd.DataFrame,
    codes: list[str],
    strategy_class: type[EventStrategy],
    start: str,
    end: str,
    capital: float,
    buy_cost: float = trading_config.BUY_COST,
    sell_cost: float = trading_config.SELL_COST,
    lot_size: int = 100,
    warmup_days: int | None = None,
    amount_q: float = 0.3,
    limit_flags: bool = True,
    slippage_bps: float = trading_config.SLIPPAGE_BPS,
    max_participation: float = trading_config.MAX_PARTICIPATION,
    short_rate: float = 0.0,
    min_commission: float = 0.0,
    buy_tax: float = 0.0,
    sell_tax: float = 0.0,
    execution_profile: AssetExecutionProfile | None = None,
    progress=None,
) -> dict:
    """事件驱动回测。

    slippage_bps: 固定滑点（基点）。买入价=开盘×(1+bps/1e4)，卖出反向。
    max_participation: 流动性约束，单笔买入金额 <= 20日均成交额 × 该比例。
        0 表示不限。
    short_rate: 空头年化融券费率（占空头市值比例/年），每日按 short_rate/252 扣。
    progress(done, total, date, nav): 每个交易日收盘估值后回调一次
        (done=t, total=T-1, 含窗口前预热段); 抛异常(如 BacktestAborted)即终止。
    """
    profile = execution_profile or STOCK_PROFILE
    if profile.asset_type == "etf":
        if buy_cost == STOCK_PROFILE.buy_cost:
            buy_cost = profile.buy_cost
        if sell_cost == STOCK_PROFILE.sell_cost:
            sell_cost = profile.sell_cost
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    calc_start = (start_ts - pd.Timedelta(days=warmup_days)
                  if warmup_days and warmup_days > 0 else start_ts)
    sub = panel[panel["code"].isin(codes)].copy()
    sub = sub[(sub["date"] >= calc_start) & (sub["date"] <= end_ts)]
    if sub.empty:
        raise ValueError("所选区间/股票池内没有数据")

    cal = pd.DatetimeIndex(sorted(sub["date"].unique()))

    for req_col in ("close", "open"):
        if req_col not in sub.columns:
            raise ValueError(f"事件回测需要列: {req_col}")

    # 排序/去重/建索引一次共享（这里逐数值列都 pivot，列数多，groupby-last 开销更显著）；
    # (date, code) 唯一时 unstack 与 pivot_table(aggfunc="last") 逐位等价，
    # 重复键保留最后一条，与 aggfunc="last" 语义一致。
    dup = sub.duplicated(["date", "code"], keep="last")
    if dup.any():
        sub = sub[~dup]
    sub = sub.sort_values(["date", "code"], kind="stable").set_index(["date", "code"])

    def pivot(col: str) -> pd.DataFrame:
        # 全 NaN 列丢弃（pivot_table 语义，codes_used 依赖这一点），
        # 全 NaN 行由 reindex(cal) 恢复交易日历
        return sub[col].unstack().dropna(axis=1, how="all").reindex(cal).sort_index()

    def pivot_aligned(col: str) -> pd.DataFrame:
        # 其余字段列集必须与 close 对齐: 某代码 close 有效但该字段全 NaN
        # (如窗口尾部次新股 am20 不足 min_periods) 时, 各自 dropna 会造成
        # 矩阵列数不一致 -> 下标越界; 统一 reindex 到 cols_ref, 无数据即 NaN。
        return sub[col].unstack().reindex(index=cal, columns=cols_ref).sort_index()

    close_df = pivot("close")
    cols_ref = close_df.columns.tolist()
    codes_used = cols_ref
    # 除日期/代码外，全部数值列转成矩阵，供 ctx.history 多字段查询
    field_mats: dict[str, np.ndarray] = {}
    for col in sub.columns:
        if col in ("date", "code"):
            continue
        try:
            arr = (pivot(col) if col == "close" else pivot_aligned(col)
                   ).to_numpy(dtype=np.float64)
        except Exception:
            continue
        if np.isfinite(arr).any():
            field_mats[col] = arr
    missing = [f for f in ("close", "open", "turnover", "am20") if f not in field_mats]
    if missing:
        raise ValueError(f"事件回测缺少字段: {missing}")

    close_mat = field_mats["close"]
    open_mat = field_mats["open"]
    turn_mat = field_mats["turnover"]
    am20_mat = field_mats["am20"]
    df_close = pd.DataFrame(close_mat, index=cal, columns=cols_ref)
    df_open = pd.DataFrame(open_mat, index=cal, columns=cols_ref)
    valid_close = ~np.isnan(close_mat)
    valid_open = ~np.isnan(open_mat)

    dates = close_df.index
    T, K = close_mat.shape
    if T < 5:
        raise ValueError("数据区间太短")

    open_ff = df_open.ffill()
    o2o = np.nan_to_num(open_ff.pct_change().values, nan=0.0)
    start_idx = int(np.argmax(dates >= start_ts)) if (dates >= start_ts).any() else 0

    limit_up = limit_down = None
    if limit_flags:
        st_mask = _load_st_mask_for(cal, codes_used, calc_start, end_ts)
        limit_up, limit_down, _, _ = build_limit_flags(df_close, df_open, st_mask=st_mask)

    ctx = Context(
        codes=codes_used, close_mat=close_mat, open_mat=open_mat,
        valid_close=valid_close, valid_open=valid_open,
        turnover_mat=turn_mat, am20_mat=am20_mat,
        limit_up=limit_up, limit_down=limit_down,
        dates=dates, capital=capital, buy_cost=buy_cost,
        sell_cost=sell_cost, lot_size=lot_size, amount_q=amount_q,
        slippage=slippage_bps / 1e4, max_participation=max_participation,
        min_commission=min_commission,
        buy_tax=buy_tax, sell_tax=sell_tax,
        field_mats=field_mats,
    )
    strategy = strategy_class()
    strategy.init(ctx)

    nav = np.ones(T)
    bench = np.ones(T)
    holdings_history: list[dict[str, float]] = []
    weight_history: list[dict[str, float]] = []
    cash_history: list[float] = []
    trades: list[dict] = []
    trades_detail: list[dict] = []
    last_chosen: list[str] = []

    for t in range(1, T):
        prev = t - 1
        ctx.t = t
        ctx.sig = prev
        ctx.orders = []

        if t == start_idx and warmup_days:
            # 预热段只用于积累策略状态，窗口起点持仓清零、净值归 1
            ctx.reset()
            nav[t] = 1.0
            bench[t] = 1.0
            holdings_history.append({})
            weight_history.append({})
            cash_history.append(ctx.cash)
            if progress is not None:
                progress(t, T - 1, dates[t], 1.0)
            continue

        # 基准：股票池等权（开盘到开盘收益）
        elig = valid_close[prev] & valid_open[prev] & valid_open[t]
        bench_ret = float(np.nanmean(np.where(elig, o2o[t], np.nan))) if elig.any() else 0.0

        # 构造信号日截面
        bar = Bar(
            date=dates[prev],
            exec_date=dates[t],
            close={c: float(close_mat[prev, k]) for k, c in enumerate(codes_used)
                   if valid_close[prev, k]},
            open={c: float(open_mat[t, k]) for k, c in enumerate(codes_used)
                  if valid_open[t, k]},
            turnover={c: float(turn_mat[prev, k]) for k, c in enumerate(codes_used)
                      if np.isfinite(turn_mat[prev, k])},
            am20={c: float(am20_mat[prev, k]) for k, c in enumerate(codes_used)
                  if np.isfinite(am20_mat[prev, k])},
            tradable={c for c in codes_used if ctx.is_tradable(c)},
        )

        # 估值基准更新为信号日收盘后，再让策略下单
        ctx.mark_to_market(prev, close_mat[prev])
        try:
            strategy.on_bar(ctx, bar)
        except Exception as exc:
            raise RuntimeError(f"策略在 {dates[prev].date()} 出错: {exc}") from exc

        stats = ctx.execute()
        hold_count = len(ctx.positions)
        if ctx.fills:
            for f in ctx.fills:
                trades_detail.append({
                    "date": dates[t], "signal_date": dates[prev],
                    **f,
                })
        if stats["trades"]:
            buys = [c for c, _, s in stats["trades"] if s == "buy"]
            sells = [c for c, _, s in stats["trades"] if s == "sell"]
            total_val = ctx.portfolio_value or 1.0
            turn = (stats["buy"] + stats["sell"]) / 2.0 / total_val
            trades.append({
                "date": dates[t],
                "signal_date": dates[prev],
                "num_hold": hold_count,
                "turnover": float(turn),
                "bought": ",".join(buys[:12]),
                "sold": ",".join(sells[:12]),
            })
            last_chosen = buys

        # 执行日收盘估值
        ctx.mark_to_market(t, close_mat[t])
        if short_rate > 0:
            short_val = sum(max(0.0, -sh * ctx._last_close.get(c, 0.0))
                            for c, sh in ctx.positions.items())
            ctx.cash -= short_val * short_rate / 252.0
        nav[t] = ctx.portfolio_value / capital
        bench[t] = bench[t - 1] * (1.0 + bench_ret)
        if progress is not None:
            progress(t, T - 1, dates[t], float(nav[t]))
        holdings_history.append(dict(ctx.positions))
        cash_history.append(ctx.cash)
        pv = ctx.portfolio_value or 1.0
        weight_history.append({c: sh * ctx._last_close.get(c, 0.0) / pv
                               for c, sh in ctx.positions.items()})

    if start_idx > 0:
        nav = nav[start_idx:]
        bench = bench[start_idx:]
        dates_out = dates[start_idx:]
        trades = [t for t in trades if t["date"] >= start_ts]
        trades_detail = [t for t in trades_detail if t["date"] >= start_ts]
        holdings_history = holdings_history[start_idx - 1:] if start_idx > 0 else holdings_history
        weight_history = weight_history[start_idx - 1:] if start_idx > 0 else weight_history
        cash_history = cash_history[start_idx - 1:] if start_idx > 0 else cash_history
    else:
        dates_out = dates

    nav_s = pd.Series(nav, index=dates_out, name="nav")
    bench_s = pd.Series(bench, index=dates_out, name="bench")
    trades_df = pd.DataFrame(trades)

    last_hold = ctx.positions
    if last_hold:
        last_close_vals = [ctx._last_close.get(c, np.nan) for c in last_hold]
        weights = [sh * px / nav_s.iloc[-1] / capital for sh, px in
                   zip(last_hold.values(), last_close_vals)]
        holdings_df = pd.DataFrame({
            "code": list(last_hold.keys()),
            "weight": weights,
            "price": last_close_vals,
            "direction": ["空" if sh < 0 else "多" for sh in last_hold.values()],
        })
        holdings_df["market_value"] = holdings_df["weight"] * nav_s.iloc[-1] * capital
        holdings_df["weight_pct"] = holdings_df["weight"] * 100
        holdings_df = holdings_df.sort_values("weight", ascending=False).reset_index(drop=True)
    else:
        holdings_df = pd.DataFrame(columns=["code", "weight", "price",
                                            "market_value", "weight_pct"])

    metrics = compute_metrics(nav_s)
    metrics.update(compute_excess_metrics(nav_s, bench_s))

    return {
        "nav": nav_s,
        "bench": bench_s,
        "drawdown": drawdown_series(nav_s),
        "metrics": metrics,
        "bench_metrics": compute_metrics(bench_s),
        "trades": trades_df,
        "trades_detail": trades_detail,
        "holdings": holdings_df,
        "last_signal_date": dates_out[-2] if len(dates_out) >= 2 else None,
        "capital": capital,
        "dates": dates_out,
        "weight_history": weight_history,
        "positions_history": holdings_history,
        "cash_history": cash_history,
        "last_chosen": last_chosen,
        "factor_quality": None,
    }
