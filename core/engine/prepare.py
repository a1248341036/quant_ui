"""阶段 1 准备：面板切片/pivot、ADX/止损矩阵、估值矩阵、o2o 与调仓计划。"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .factors import _compute_adx, _compute_atr
from ..assets import STOCK_PROFILE
from ..limit import build_limit_flags
from ..selection import PortfolioBuilder, SelectionPolicy


def _load_st_mask_for(
    cal: pd.DatetimeIndex,
    codes_used: list[str],
    calc_start: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame | None:
    """读取回测窗口内逐日 ST 标记（6 位 code 宽表），失败时返回 None。

    返回与 build_limit_flags 契约一致的 st_mask（index=交易日, columns=code,
    True=ST）。CNE 读取不可用时（未启用/无数据/失败）静默降级为 None，
    由 limit 层按板块比例近似——保持旧行为。
    """
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


def _prepare_backtest(cfg) -> dict:
    """阶段 1 准备:面板切片/pivot、ADX/止损矩阵、估值矩阵、o2o 与调仓计划。

    产出后续阶段所需全部矩阵/计划/构建器;因子矩阵构建在阶段 2。
    """
    panel = cfg.panel
    codes = cfg.codes
    factor = cfg.factor
    ascending = cfg.ascending
    start = cfg.start
    end = cfg.end
    capital = cfg.capital
    top_n = cfg.top_n
    freq = cfg.freq
    buy_cost = cfg.buy_cost
    sell_cost = cfg.sell_cost
    amount_q = cfg.amount_q
    affordable = cfg.affordable
    lot_size = cfg.lot_size
    warmup_days = cfg.warmup_days
    cash_mode = cfg.cash_mode
    limit_flags = cfg.limit_flags
    slippage_bps = cfg.slippage_bps
    max_participation = cfg.max_participation
    max_weight = cfg.max_weight
    industry_map = cfg.industry_map
    industry_cap = cfg.industry_cap
    factor_builder = cfg.factor_builder
    external_scores = cfg.external_scores
    factor_weights = cfg.factor_weights
    factor_directions = cfg.factor_directions
    analyze = cfg.analyze
    long_short = cfg.long_short
    short_n = cfg.short_n
    short_cost_rate = cfg.short_cost_rate
    industry_neutral = cfg.industry_neutral
    use_financial = cfg.use_financial
    risk_neutral = cfg.risk_neutral
    adx_filter = cfg.adx_filter
    chandelier_mult = cfg.chandelier_mult
    chandelier_period = cfg.chandelier_period
    regime_adx = cfg.regime_adx
    regime_scale = cfg.regime_scale
    selection_mode = cfg.selection_mode
    selection_pct = cfg.selection_pct
    min_positions = cfg.min_positions
    max_positions = cfg.max_positions
    min_score = cfg.min_score
    execution_profile = cfg.execution_profile
    share_classes = cfg.share_classes
    spread_bps = cfg.spread_bps
    min_commission = cfg.min_commission
    impact_coef = cfg.impact_coef
    impact_vol = cfg.impact_vol
    profile = execution_profile or STOCK_PROFILE
    spread_bps = profile.spread_bps if spread_bps is None else spread_bps
    min_commission = profile.min_commission if min_commission is None else min_commission
    if profile.asset_type in ("etf", "fund_nav"):
        # ETF/基金入口使用各自默认费率；显式传入非股票默认费率时保留调用方配置。
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

    # 排序/去重/建索引只做一次，8 列 pivot 共享（原先每列重复做是回测的主要耗时）。
    # 日频面板 (date, code) 唯一时 set_index+unstack 与 pivot_table(aggfunc="last")
    # 逐位等价；出现重复键时保留最后一条，与 aggfunc="last" 语义一致。
    dup = sub.duplicated(["date", "code"], keep="last")
    if dup.any():
        sub = sub[~dup]
    sub = sub.sort_values(["date", "code"], kind="stable").set_index(["date", "code"])

    def pivot(col: str) -> pd.DataFrame:
        # unstack 会保留全 NaN 行/列（pivot_table 会丢弃），dropna+reindex 恢复其语义：
        # 全 NaN 列（如新上市代码的 am20）丢弃后由下方统一对齐回 close 列；
        # 全 NaN 行（如 am20 前 20 个交易日）经 reindex(cal) 回到交易日历。
        out = sub[col].unstack()
        out = out.dropna(axis=1, how="all")
        return out.reindex(cal).sort_index()

    close = pivot("close")
    open_ = pivot("open")
    turnover = pivot("turnover")
    am20 = pivot("am20")
    turn20 = pivot("turn20")
    high = pivot("high") if "high" in sub.columns else None
    low = pivot("low") if "low" in sub.columns else None
    volume_w = pivot("volume") if "volume" in sub.columns else None
    # 新上市/新发代码可能在部分因子矩阵中无列（如 am20 全 NaN 被 pivot 丢弃），
    # 统一对齐到 close 列，缺失列补 NaN 后由 valid 掩码过滤为不可交易。
    open_ = open_.reindex(columns=close.columns)
    turnover = turnover.reindex(columns=close.columns)
    am20 = am20.reindex(columns=close.columns)
    turn20 = turn20.reindex(columns=close.columns)
    if high is not None:
        high = high.reindex(columns=close.columns)
    if low is not None:
        low = low.reindex(columns=close.columns)
    codes_used = close.columns.tolist()

    adx_mat = None
    if adx_filter is not None and adx_filter > 0:
        if high is None or low is None:
            print("警告: 面板无 high/low，ADX 过滤被跳过", file=sys.stderr)
        else:
            adx_mat = _compute_adx(high, low, close).values

    stop_mat = None
    if chandelier_mult > 0:
        if high is None or low is None:
            print("警告: 面板无 high/low，Chandelier 出场被跳过", file=sys.stderr)
        else:
            stop_mat = (high.rolling(chandelier_period).max()
                        - chandelier_mult * _compute_atr(high, low, close, chandelier_period)).values

    close_mat = close.values
    # 停牌日 close 为 NaN，持仓按最后一笔有效收盘价（每股 ffill）继续估值，
    # 避免停牌期间市值被当成 0、复牌日净值跳变。
    close_fill_mat = close.ffill().values
    open_mat = open_.values
    turn_mat = turnover.values
    am20_mat = am20.values
    valid_close = ~np.isnan(close_mat)
    valid_open = ~np.isnan(open_mat) & (open_mat > 0)

    dates = close.index
    T, K = close.shape
    if T < 5:
        raise ValueError("数据区间太短")

    open_ff = open_.where(open_ > 0).ffill()
    o2o = open_ff.pct_change().values  # t>=1
    # 停牌/缺失日 open 为 0 时 pct_change 会产生 inf，与 NaN 一起归零，避免污染等权基准
    o2o = np.nan_to_num(o2o, nan=0.0, posinf=0.0, neginf=0.0)

    start_idx = int(np.argmax(dates >= start_ts)) if (dates >= start_ts).any() else 0
    if freq == "daily":
        signal_idx = list(range(max(1, start_idx)))
    elif freq == "monthly":
        signal_idx = [i - 1 for i in range(1, T) if dates[i].month != dates[i - 1].month]
    elif freq == "semiannual":
        # 每年 3 月/9 月换仓：信号取前一交易日收盘，次日开盘成交
        signal_idx = [i - 1 for i in range(1, T)
                      if dates[i].month != dates[i - 1].month
                      and dates[i].month in (3, 9)]
    else:
        signal_idx = [i for i, d in enumerate(dates) if d.weekday() == 4]
    exec_dates = [i + 1 for i in signal_idx if i + 1 < T]
    # 模拟盘不支持空头；空头回测继续走权重模型
    use_cash = bool(cash_mode) and not long_short
    if freq == "daily":
        # 每日收盘信号 -> 次日开盘成交，现金/权重模式统一每日调仓
        exec_dates = list(range(max(1, start_idx), T))
    if use_cash:
        # 与模拟盘一致：窗口起点即首次调仓日（立即建仓）
        exec_dates = sorted(set(exec_dates) | {start_idx})
    limit_up = limit_down = None
    if use_cash and limit_flags:
        st_mask = _load_st_mask_for(cal, codes_used, calc_start, end_ts)
        limit_up, limit_down, _, _ = build_limit_flags(close, open_, st_mask=st_mask)
    # 预热模式：窗口起点不继承预热段持仓，也不在窗口起点当天调仓，
    # 与"全量算因子、窗口起点从零开始"的旧脚本语义一致。
    exec_set = ({i for i in exec_dates if i != start_idx}
                if (warmup_days and not use_cash) else set(exec_dates))

    portfolio_builder = PortfolioBuilder(codes_used, industry_map, industry_cap)
    # 选股策略对象：把散装选股参数收拢，选股逻辑统一走 build_targets
    selection_policy = SelectionPolicy(
        count_mode=selection_mode, top_n=top_n, pct=selection_pct,
        min_positions=min_positions, max_positions=max_positions,
        ascending=ascending, min_score=min_score,
        industry_cap=industry_cap,
        regime_adx=regime_adx, regime_scale=regime_scale,
    )

    return {
        "profile": profile, "spread_bps": spread_bps,
        "min_commission": min_commission, "buy_cost": buy_cost,
        "sell_cost": sell_cost, "start_ts": start_ts, "end_ts": end_ts,
        "cal": cal, "close": close, "open_": open_, "turnover": turnover,
        "am20": am20, "turn20": turn20, "high": high, "low": low,
        "volume_w": volume_w, "codes_used": codes_used,
        "adx_mat": adx_mat, "stop_mat": stop_mat,
        "close_mat": close_mat, "close_fill_mat": close_fill_mat,
        "open_mat": open_mat, "turn_mat": turn_mat, "am20_mat": am20_mat,
        "valid_close": valid_close, "valid_open": valid_open,
        "dates": dates, "T": T, "K": K, "o2o": o2o, "start_idx": start_idx,
        "exec_dates": exec_dates, "use_cash": use_cash,
        "limit_up": limit_up, "limit_down": limit_down, "exec_set": exec_set,
        "portfolio_builder": portfolio_builder,
        "selection_policy": selection_policy,
        "capital": capital, "top_n": top_n, "freq": freq,
        "amount_q": amount_q, "affordable": affordable, "lot_size": lot_size,
        "warmup_days": warmup_days, "cash_mode": cash_mode,
        "limit_flags": limit_flags, "slippage_bps": slippage_bps,
        "max_participation": max_participation, "max_weight": max_weight,
        "industry_map": industry_map, "industry_cap": industry_cap,
        "factor_builder": factor_builder, "external_scores": external_scores,
        "factor_weights": factor_weights,
        "factor_directions": factor_directions, "analyze": analyze,
        "long_short": long_short, "short_n": short_n,
        "short_cost_rate": short_cost_rate,
        "industry_neutral": industry_neutral, "use_financial": use_financial,
        "risk_neutral": risk_neutral, "adx_filter": adx_filter,
        "chandelier_mult": chandelier_mult,
        "chandelier_period": chandelier_period,
        "regime_adx": regime_adx, "regime_scale": regime_scale,
        "selection_mode": selection_mode, "selection_pct": selection_pct,
        "min_positions": min_positions, "max_positions": max_positions,
        "min_score": min_score, "execution_profile": execution_profile,
        "share_classes": share_classes,
        "impact_coef": impact_coef, "impact_vol": impact_vol,
        "use_screener": cfg.use_screener,
        "screener_lookback": cfg.screener_lookback,
        "screener_min_ic": cfg.screener_min_ic,
        "screener_max_corr": cfg.screener_max_corr,
        "screener_factors": cfg.screener_factors,
        "signal_indices": sorted(set(i for i in
            (e - 1 for e in exec_dates if e > 0)
            if 0 <= i < T)),
    }
