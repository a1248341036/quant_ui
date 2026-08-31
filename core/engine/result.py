"""阶段 4 收尾 + latest_signals 信号计算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import (
    build_factor_frames,
    _inject_pred_factor,
    _ensure_ma_cross_factor,
    build_composite_factor,
    _compute_adx,
)
from ..metrics import compute_excess_metrics, compute_metrics, drawdown_series


def _finalize_result(cfg, prep: dict, fctx: dict, sim: dict) -> dict:
    """阶段 4 收尾:预热段切片、指标、持仓表、风险归因与结果字典。"""
    profile = prep["profile"]
    close = prep["close"]
    codes_used = prep["codes_used"]
    dates = prep["dates"]
    K = prep["K"]
    o2o = prep["o2o"]
    start_ts = prep["start_ts"]
    start_idx = prep["start_idx"]
    exec_set = prep["exec_set"]
    capital = prep["capital"]
    nav = sim["nav"]
    bench = sim["bench"]
    trades = sim["trades"]
    trades_detail = sim["trades_detail"]
    holdings_history = sim["holdings_history"]
    cash_history = sim["cash_history"]
    positions_history = sim["positions_history"]
    rejections = sim["rejections"]
    last_chosen = sim["last_chosen"]
    hold = sim["hold"]
    _X_risk = fctx.get("_X_risk")
    _risk_names = fctx.get("_risk_names") or []
    quality = fctx.get("quality")
    screener_log = fctx.get("screener_log", [])

    exec_in_out = sorted(e for e in exec_set if e > 0)
    last_signal_date = dates[exec_in_out[-1] - 1] if exec_in_out else None
    if start_idx > 0:
        # 预热段只用于因子计算，净值/成交从 start 开始输出
        nav = nav[start_idx:]
        bench = bench[start_idx:]
        # 兜底：基准必须从窗口起点归 1，避免前端 capital*bench 起点偏差
        if len(bench) and not np.isclose(bench[0], 1.0):
            bench = bench / bench[0]
        dates_out = dates[start_idx:]
        trades = [t for t in trades if t["date"] >= start_ts]
        holdings_history = holdings_history[start_idx - 1:] if start_idx > 0 else holdings_history
        cash_history = cash_history[start_idx - 1:] if start_idx > 0 else cash_history
        positions_history = positions_history[start_idx - 1:] if start_idx > 0 else positions_history
        trades_detail = [t for t in trades_detail if t["date"] >= start_ts]
        rejections = [r for r in rejections if r["date"] >= start_ts.date().isoformat()]
    else:
        dates_out = dates
        # start_idx == 0：nav 有 T 个元素（含初始 nav[0]=1.0），
        # 但 cash_history/positions_history 只有 T-1 个（循环从 t=1 开始记录）。
        # 补齐初始状态，使长度与 nav/weight_history 一致。
        cash_history = [float(capital)] + cash_history
        positions_history = [{}] + positions_history

    if quality is not None:
        from ..performance import slice_quality
        quality = slice_quality(quality, dates_out)

    # 每日市值权重历史（供 Brinson/风险归因等下游使用）
    wh_list = holdings_history if start_idx > 0 else [np.zeros(K)] + holdings_history
    weight_history = [{codes_used[k]: float(v) for k, v in enumerate(h)
                       if abs(v) > 1e-9} for h in wh_list]

    nav_s = pd.Series(nav, index=dates_out, name="nav")
    bench_s = pd.Series(bench, index=dates_out, name="bench")
    trades_df = pd.DataFrame(trades)

    last_hold = pd.Series(hold, index=codes_used)
    last_holdings = last_hold[last_hold != 0].sort_values(ascending=False)
    last_price = close.iloc[-1]
    holdings_df = pd.DataFrame({
        "code": last_holdings.index,
        "weight": last_holdings.values,
        "price": [last_price.get(c, np.nan) for c in last_holdings.index],
        "direction": ["空" if v < 0 else "多" for v in last_holdings.values],
    })
    holdings_df["market_value"] = holdings_df["weight"] * nav_s.iloc[-1] * capital
    holdings_df["weight_pct"] = holdings_df["weight"] * 100

    risk_attribution = None
    if _X_risk is not None:
        from ..risk_model import (covariance_from_exposures,
                                 portfolio_risk_attribution)
        _, factor_cov, spec_var = covariance_from_exposures(_X_risk, o2o)
        last_w = np.zeros(K)
        for c, v in last_holdings.items():
            last_w[codes_used.index(str(c))] = v
        w_norm = last_w / (np.abs(last_w).sum() or 1.0)
        risk_attribution = portfolio_risk_attribution(
            w_norm, _X_risk, factor_cov, spec_var, _risk_names)

    metrics = compute_metrics(nav_s)
    metrics.update(compute_excess_metrics(nav_s, bench_s))

    return {
        "nav": nav_s,
        "bench": bench_s,
        "drawdown": drawdown_series(nav_s),
        "metrics": metrics,
        "bench_metrics": compute_metrics(bench_s),
        "trades": trades_df,
        "holdings": holdings_df,
        "last_signal_date": last_signal_date,
        "capital": capital,
        "dates": dates_out,
        "last_chosen": last_chosen,
        "factor_quality": quality,
        "weight_history": weight_history,
        "trades_detail": trades_detail,
        "cash_history": cash_history,
        "positions_history": positions_history,
        "rejections": rejections,
        "risk_attribution": risk_attribution,
        "asset_type": profile.asset_type,
        "execution_profile": profile,
        "screener_log": screener_log,
    }


def latest_signals(panel: pd.DataFrame, codes: list[str], factor: str,
                   ascending: bool, top_n: int = 10,
                   factor_weights: dict[str, float] | None = None,
                   factor_directions: dict[str, bool] | None = None,
                   long_short: bool = False,
                   short_n: int | None = None,
                   use_financial: bool = False,
                   adx_filter: float | None = None,
                   asset_type: str = "stock") -> pd.DataFrame:
    sub = panel[panel["code"].isin(codes)].copy()

    cal = pd.DatetimeIndex(sorted(sub["date"].unique()))

    # 与 _prepare_backtest.pivot 相同的快速路径：排序/去重/建索引一次共享。
    dup = sub.duplicated(["date", "code"], keep="last")
    if dup.any():
        sub = sub[~dup]
    sub = sub.sort_values(["date", "code"], kind="stable").set_index(["date", "code"])

    def pivot(col: str) -> pd.DataFrame:
        # 全 NaN 列丢弃（pivot_table 语义），全 NaN 行由 reindex(cal) 恢复交易日历
        return sub[col].unstack().dropna(axis=1, how="all").reindex(cal).sort_index()

    close = pivot("close")
    am20 = pivot("am20")
    turn20 = pivot("turn20")
    turnover = pivot("turnover")
    high = pivot("high") if "high" in sub.columns else None
    low = pivot("low") if "low" in sub.columns else None
    volume_sig = pivot("volume") if "volume" in sub.columns else None
    adx_row = None
    if adx_filter is not None and high is not None and low is not None:
        high = high.reindex(columns=close.columns)
        low = low.reindex(columns=close.columns)
        adx_row = _compute_adx(high, low, close).iloc[-1]
    from ..financial import FINANCIAL_FACTORS, financial_factor_frames
    need_financial = use_financial or factor in FINANCIAL_FACTORS
    if factor_weights:
        need_financial = need_financial or any(
            n in FINANCIAL_FACTORS for n in factor_weights)
    financial_frames = None
    if need_financial:
        try:
            financial_frames = financial_factor_frames(
                close.columns.tolist(), close.index, close)
        except Exception:
            financial_frames = None
    factors = build_factor_frames(close, am20, turn20,
                                  financial=financial_frames,
                                  asset_type=asset_type,
                                  volume=volume_sig)
    _inject_pred_factor(factors, close, factor, factor_weights)
    _ensure_ma_cross_factor(factors, close, factor)
    last_date = close.index[-1]
    if factor_weights:
        combo = build_composite_factor(
            close, am20, turn20, factor_weights, factor_directions,
            factor_builder=(lambda c, a, t: build_factor_frames(
                c, a, t, financial=financial_frames,
                asset_type=asset_type)),
            extra_factors={"pred": factors["pred"]} if "pred" in factors else None)
        row = combo.iloc[-1]
        factor = "composite"
    else:
        row = factors[factor].iloc[-1]
    am_row = am20.iloc[-1]
    turn_row = turnover.iloc[-1]
    close_row = close.iloc[-1]

    cand = row.dropna()
    valid = (am_row[cand.index].notna() & (turn_row[cand.index] > 0)
             & close_row[cand.index].notna())
    if adx_row is not None:
        valid = valid & (adx_row[cand.index] >= adx_filter)
    cand = cand[valid].sort_values(ascending=ascending)
    top = cand.tail(top_n) if not ascending else cand.head(top_n)
    if long_short:
        short_count = short_n or top_n
        bottom = cand.head(short_count) if not ascending else cand.tail(short_count)
        bottom = bottom[~bottom.index.isin(top.index)]
        rows = []
        for c, s in top.items():
            rows.append({"code": c, "score": s, "side": "多"})
        for c, s in bottom.items():
            rows.append({"code": c, "score": s, "side": "空"})
        top_idx = pd.Index([r["code"] for r in rows])
        out = pd.DataFrame({
            "code": top_idx,
            "side": [r["side"] for r in rows],
            "score": [r["score"] for r in rows],
            "close": [close_row.get(c, np.nan) for c in top_idx],
            "turnover": [turn_row.get(c, np.nan) for c in top_idx],
        }).reset_index(drop=True)
        return out, last_date

    out = pd.DataFrame({
        "code": top.index,
        "score": top.values,
        "close": [close_row.get(c, np.nan) for c in top.index],
        "turnover": [turn_row.get(c, np.nan) for c in top.index],
    }).reset_index(drop=True)
    return out, last_date
