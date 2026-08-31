"""旧引擎认证门禁：submit 进正式库前，在验证集窗口跑完整约束回测。

数据全部在内存中变换：AlphaAgent panel (datetime, instrument) → 旧引擎长表
(date, code, open/high/low/close, turnover, am20, turn20)。不产生任何磁盘中间表。

单位换算（amount 千元→元、turnover_rate %→比例）与列契约统一由
core.panel_schema 负责，本模块不再重复实现。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core import trading_config
from core.panel_schema import alpha_panel_to_engine_frame as panel_to_engine_frame


def run_engine_gate(
    panel: pd.DataFrame,
    factor_values: np.ndarray,
    *,
    val_start: str,
    val_end: str,
    direction: int = 1,
    policy: dict[str, Any] | None = None,
    engine_frame: pd.DataFrame | None = None,
    asset_type: str = "stock",
) -> dict[str, Any]:
    """验证集窗口的旧引擎 TopN 回测门禁（完整 T+1/涨跌停/停牌/整手约束）。

    asset_type 决定单位换算（stock：amount 千元→元；etf：amount 已是元）与
    回测执行规则（ETF 用 ETF_PROFILE：免涨跌停、低佣金、价差/最低佣金单独设）。
    policy 来自 ResearchSpec delivery_policy.production.engine_gate：
    {enabled, selection_mode, selection_pct, top_n, freq, capital,
     slippage_bps, max_participation, min_am20_yuan, min_excess_annual,
     min_excess_sharpe, max_drawdown, min_daily_overlap, min_invested_ratio}
    数值由 delivery_criteria.EngineGateCriteria 提供（唯一真源），
    本函数对缺失键仅回落 trading_config，不回落到散落的局部硬编码。
    engine_frame 可传入缓存的 panel_to_engine_frame 输出，多频率复评时避免重复变换。
    """
    policy = policy or {}
    from core.assets import get_execution_profile
    from core.engine import run_backtest

    scores = pd.Series(np.asarray(factor_values, dtype=np.float64), index=panel.index) * float(direction)

    if engine_frame is None:
        engine_frame = panel_to_engine_frame(panel, asset_type=asset_type)

    # 流动性硬过滤（2026-08 审计）：Top2% 尾部会选中日均成交额仅数万元的
    # 僵尸微盘，参与率预算连一手都买不起 → 大面积拒单 + 现金闲置 80%+。
    # 低于 min_am20_yuan 的 (date, code) 打分置 NaN，交由引擎 eligible 过滤剔除。
    min_am20 = float(policy.get("min_am20_yuan", 0) or 0)
    if min_am20 > 0 and not engine_frame.empty:
        am_key = pd.MultiIndex.from_arrays(
            [pd.to_datetime(engine_frame["date"]), engine_frame["code"].astype(str)],
            names=["datetime", "instrument"],
        )
        am_long = pd.Series(engine_frame["am20"].to_numpy(dtype=np.float64), index=am_key)
        am_aligned = am_long.reindex(scores.index)
        scores = scores.mask(am_aligned < min_am20)

    wide = scores.unstack("instrument")
    codes = [str(c) for c in wide.columns]

    # 选股模式：支持固定 top_n 或动态百分比 top_pct。
    # 默认用动态百分比（selection_pct=0.02 ≈ A股 5000 只的 2% ≈ 100 只），
    # 停牌/涨跌停导致当曰候选池缩小时自动适配，避免固定 N 选到不可交易的尾部股票。
    selection_mode = str(policy.get("selection_mode", trading_config.SELECTION_MODE))
    selection_pct = float(policy.get("selection_pct", trading_config.GATE_SELECTION_PCT))
    top_n_fixed = int(policy.get("top_n", trading_config.GATE_TOP_N))

    try:
        bt_kwargs: dict[str, Any] = dict(
            codes=codes,
            factor="pred",
            ascending=False,
            start=str(val_start),
            end=str(val_end),
            capital=float(policy.get("capital", trading_config.GATE_CAPITAL)),
            top_n=top_n_fixed,
            freq=str(policy.get("freq", trading_config.GATE_FREQ)),
            warmup_days=25,
            external_scores=wide,
            slippage_bps=float(policy.get("slippage_bps", trading_config.GATE_SLIPPAGE_BPS)),
            max_participation=float(policy.get("max_participation", trading_config.GATE_MAX_PARTICIPATION)),
            selection_mode=selection_mode,
            selection_pct=selection_pct,
        )
        result = run_backtest(
            engine_frame,
            **bt_kwargs,
            execution_profile=get_execution_profile(asset_type),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "enabled": True,
            "passed": False,
            "fail_reasons": ["engine_backtest_error"],
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {},
        }

    m = result.get("metrics") or {}
    reasons: list[str] = []
    thresholds: dict[str, Any] = {}

    # 净值超额年化（vs 全市场等权基准，扣全费）：alpha 的核心裁决项
    excess = m.get("超额年化")
    if "min_excess_annual" in policy:
        thr = float(policy["min_excess_annual"])
        thresholds["min_excess_annual"] = thr
        if excess is None or not np.isfinite(float(excess)) or float(excess) < thr:
            reasons.append("excess_annual")

    # 超额夏普（active NAV 口径）：替代旧的绝对夏普门——样本含熊市时
    # 绝对收益/绝对夏普门会把所有因子拒之门外，与 alpha 无关。
    exc_sharpe = m.get("超额夏普")
    if policy.get("min_excess_sharpe") is not None:
        thr = float(policy["min_excess_sharpe"])
        thresholds["min_excess_sharpe"] = thr
        if exc_sharpe is None or not np.isfinite(float(exc_sharpe)) or float(exc_sharpe) < thr:
            reasons.append("excess_sharpe")

    # 兼容旧键：显式提供时仍按旧口径检查（默认 spec 已不含它们）
    annual = m.get("年化收益")
    if policy.get("min_annual_return") is not None:
        thr = float(policy["min_annual_return"])
        thresholds["min_annual_return"] = thr
        if annual is None or not np.isfinite(float(annual)) or float(annual) < thr:
            reasons.append("annual_return")
    sharpe = m.get("夏普")
    if policy.get("min_sharpe") is not None:
        thr = float(policy["min_sharpe"])
        thresholds["min_sharpe"] = thr
        if sharpe is None or not np.isfinite(float(sharpe)) or float(sharpe) < thr:
            reasons.append("sharpe")

    drawdown = m.get("最大回撤")
    if policy.get("max_drawdown") is not None:
        max_dd = float(policy["max_drawdown"])
        thresholds["max_drawdown"] = max_dd
        if drawdown is None or not np.isfinite(float(drawdown)) or abs(float(drawdown)) > max_dd:
            reasons.append("max_drawdown")

    from alphaagent.factor.metrics import topn_selection_overlap
    selection_pct_val = float(policy.get("selection_pct", trading_config.GATE_SELECTION_PCT)) if selection_mode == "top_pct" else None
    overlap = topn_selection_overlap(
        scores,
        top_n=top_n_fixed if selection_mode != "top_pct" else None,
        selection_pct=selection_pct_val,
        rebalance=str(policy.get("freq", "daily")),
    )
    min_overlap = float(policy.get("min_daily_overlap") or 0)
    thresholds["min_daily_overlap"] = min_overlap
    if min_overlap and (not np.isfinite(overlap) or overlap < min_overlap):
        reasons.append("tail_stability")

    # ── 现金拖累 / 执行诊断（2026-08 审计新增）──
    # 利用 run_backtest 现成的 cash_history/rejections/trades 输出定位
    # "理想毛超额为正但净值大幅落后"的执行侧原因：
    # - avg_invested_ratio 长期 < 1 ⇒ 资金没打满（流动性/参与率/现金模式限制）
    # - rejections 数量大 ⇒ 大量订单被拒（涨跌停/停牌/参与率）
    diag: dict[str, Any] = {
        "avg_invested_ratio": None,
        "min_invested_ratio": None,
        "days_below_half_invested": None,
        "n_rejections": 0,
        "top_rejection_reasons": {},
        "avg_num_hold": None,
        "avg_daily_turnover": None,
    }
    try:
        nav_bt = result.get("nav")
        cash_hist = result.get("cash_history") or []
        cap_bt = float(result.get("capital") or 0) or 1.0
        if nav_bt is not None and len(cash_hist):
            nav_vals = np.asarray(nav_bt, dtype=np.float64)[: len(cash_hist)]
            ratios = []
            for i, eq_ratio in enumerate(nav_vals):
                eq_value = float(eq_ratio) * cap_bt
                if eq_value > 0:
                    ratios.append(1.0 - float(cash_hist[i]) / eq_value)
            if ratios:
                arr_r = np.asarray(ratios, dtype=np.float64)
                diag["avg_invested_ratio"] = round(float(arr_r.mean()), 4)
                diag["min_invested_ratio"] = round(float(arr_r.min()), 4)
                diag["days_below_half_invested"] = int((arr_r < 0.5).sum())
        rej = result.get("rejections") or []
        diag["n_rejections"] = len(rej)
        reason_counts: dict[str, int] = {}
        for rj in rej:
            key = str(rj.get("reason") if isinstance(rj, dict) else rj)[:48]
            reason_counts[key] = reason_counts.get(key, 0) + 1
        diag["top_rejection_reasons"] = dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])[:5])
        tr = result.get("trades")
        if tr is not None and len(tr):
            if "num_hold" in tr.columns:
                diag["avg_num_hold"] = round(float(pd.to_numeric(tr["num_hold"], errors="coerce").mean()), 1)
            if "turnover" in tr.columns:
                diag["avg_daily_turnover"] = round(
                    float(pd.to_numeric(tr["turnover"], errors="coerce").mean()), 4)
    except Exception as exc:  # noqa: BLE001
        diag["error"] = f"{type(exc).__name__}: {exc}"

    # 执行可行性硬门：平均仓位利用率过低说明大量目标单被拒、现金闲置，
    # 此时的净值超额被踏空损耗污染，不可作为因子质量的裁决依据。
    min_inv = float(policy.get("min_invested_ratio") or 0)
    if min_inv > 0:
        thresholds["min_invested_ratio"] = min_inv
        avg_inv = diag.get("avg_invested_ratio")
        if avg_inv is None or not np.isfinite(float(avg_inv)) or float(avg_inv) < min_inv:
            reasons.append("execution_infeasible")

    return {
        "enabled": True,
        "passed": len(reasons) == 0,
        "fail_reasons": reasons,
        "thresholds": thresholds,
        "diagnostics": diag,
        "metrics": {
            "annual_return": annual,
            "excess_annual": excess,
            "sharpe": sharpe,
            "excess_sharpe": exc_sharpe,
            "calmar": m.get("卡玛"),
            "win_rate": m.get("胜率"),
            "total_return": m.get("总收益"),
            "max_drawdown": drawdown,
            "daily_overlap": overlap,
        },
        "selection_mode": selection_mode,
        "selection_pct": selection_pct if selection_mode == "top_pct" else None,
        "top_n": top_n_fixed,
        "freq": str(policy.get("freq", trading_config.GATE_FREQ)),
        "window": {"start": str(val_start), "end": str(val_end)},
    }
