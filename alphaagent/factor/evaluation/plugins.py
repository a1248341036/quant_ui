"""Built-in transforms and metrics for the plugin evaluation engine."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from alphaagent.factor import metrics_fast as _mf
from alphaagent.factor.evaluation.context import EvaluationContext
from alphaagent.factor.metrics import (
    _day_slices,
    coverage,
    cross_sectional_lag1_pearson_autocorr,
    cs_ic_summary,
    daily_long_short_series,
    decile_mean_label,
    factor_skew_kurtosis,
    monthly_ic_robustness,
    mls_fmb_summary,
    newey_west_mean_tstat,
)

Transform = Callable[[EvaluationContext, dict[str, Any]], None]
Metric = Callable[[EvaluationContext, dict[str, Any]], dict[str, Any]]

_TRANSFORMS: dict[str, Transform] = {}
_METRICS: dict[str, Metric] = {}


def transform(name: str) -> Callable[[Transform], Transform]:
    def register(func: Transform) -> Transform:
        _TRANSFORMS[name] = func
        return func
    return register


def metric(name: str) -> Callable[[Metric], Metric]:
    def register(func: Metric) -> Metric:
        _METRICS[name] = func
        return func
    return register


def get_transform(name: str) -> Transform:
    try:
        return _TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown_transform_plugin:{name}") from exc


def get_metric(name: str) -> Metric:
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ValueError(f"unknown_metric_plugin:{name}") from exc


def available_plugins() -> dict[str, list[str]]:
    return {"transforms": sorted(_TRANSFORMS), "metrics": sorted(_METRICS)}


def _date_groups(context: EvaluationContext):
    dates = context.panel.index.get_level_values("datetime")
    positions = pd.Series(np.arange(len(context.panel)), index=dates)
    yield from positions.groupby(level=0, sort=False)


@transform("cross_sectional_winsorize")
def cross_sectional_winsorize(context: EvaluationContext, params: dict[str, Any]) -> None:
    lower = float(params.get("lower_pct", 1.0))
    upper = float(params.get("upper_pct", 99.0))
    if not 0 <= lower < upper <= 100:
        raise ValueError("cross_sectional_winsorize_percentiles_invalid")
    out = context.factor.to_numpy(dtype=np.float64, copy=True)
    slices = _day_slices(context.panel.index)
    if slices is not None and _mf.HAS_NUMBA:
        _mf.winsorize_days(out, slices[0], lower, upper)
    else:
        for _, row in _date_groups(context):
            idx = row.to_numpy(dtype=np.int64, copy=False)
            values = out[idx]
            finite = np.isfinite(values)
            if finite.sum() >= 2:
                lo, hi = np.nanpercentile(values[finite], [lower, upper])
                values[finite] = np.clip(values[finite], lo, hi)
                out[idx] = values
    context.replace_factor(out, transform_name="cross_sectional_winsorize")


@transform("cross_sectional_zscore")
def cross_sectional_zscore(context: EvaluationContext, params: dict[str, Any]) -> None:
    _ = params
    out = context.factor.to_numpy(dtype=np.float64, copy=True)
    slices = _day_slices(context.panel.index)
    if slices is not None and _mf.HAS_NUMBA:
        _mf.zscore_days(out, slices[0])
    else:
        for _, row in _date_groups(context):
            idx = row.to_numpy(dtype=np.int64, copy=False)
            values = out[idx]
            finite = np.isfinite(values)
            if finite.sum() >= 2:
                std = float(np.std(values[finite], ddof=0))
                if std > 0 and np.isfinite(std):
                    values[finite] = (values[finite] - float(np.mean(values[finite]))) / std
                    out[idx] = values
    context.replace_factor(out, transform_name="cross_sectional_zscore")


@transform("size_residualize")
def size_residualize(context: EvaluationContext, params: dict[str, Any]) -> None:
    field = str(params.get("field", "float_cap"))
    if field not in context.panel.columns:
        # ETF 等无市值字段的面板：跳过残差化，保留原因子（与 size_neutral_decay
        # 缺字段返回 note 的容错语义一致），避免 evaluate 整条流水报错。
        context.replace_factor(
            context.factor.to_numpy(dtype=np.float64, copy=True),
            transform_name="size_residualize_skipped",
        )
        return
    log_scale = bool(params.get("log", True))
    out = context.factor.to_numpy(dtype=np.float64, copy=True)
    size = context.panel[field].to_numpy(dtype=np.float64, copy=False)
    if log_scale:
        size = np.where(size > 0, np.log(size), np.nan)
    for _, row in _date_groups(context):
        idx = row.to_numpy(dtype=np.int64, copy=False)
        y, x = out[idx], size[idx]
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 3:
            continue
        x_valid, y_valid = x[valid], y[valid]
        variance = float(np.var(x_valid))
        if not np.isfinite(variance) or variance <= 1e-15:
            continue
        beta = float(np.cov(x_valid, y_valid, ddof=0)[0, 1] / variance)
        alpha = float(np.mean(y_valid) - beta * np.mean(x_valid))
        y[valid] = y_valid - (alpha + beta * x_valid)
        out[idx] = y
    context.replace_factor(out, transform_name="size_residualize")


@transform("industry_neutralize")
def industry_neutralize(context: EvaluationContext, params: dict[str, Any]) -> None:
    field = str(params.get("field", "industry_sw_l1"))
    if field not in context.panel.columns:
        raise ValueError(f"industry_neutralize_missing_field:{field}")
    out = context.factor.to_numpy(dtype=np.float64, copy=True)
    group = context.panel[field].to_numpy(copy=False)
    for _, row in _date_groups(context):
        idx = row.to_numpy(dtype=np.int64, copy=False)
        values, groups = out[idx], group[idx]
        valid = np.isfinite(values) & pd.notna(groups)
        if valid.sum() < 2:
            continue
        for group_id in pd.unique(groups[valid]):
            mask = valid & (groups == group_id)
            if mask.sum() >= 2:
                values[mask] -= float(np.mean(values[mask]))
        out[idx] = values
    context.replace_factor(out, transform_name="industry_neutralize")


@metric("cross_sectional_core")
def cross_sectional_core(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    _ = params
    daily_ic = context.daily_ic()
    daily_rank_ic = context.daily_rank_ic()
    summary = cs_ic_summary(
        daily_ic, daily_rank_ic,
        holding_days=getattr(context, "label_holding_days", 1),
    )
    values = context.factor.to_numpy(dtype=np.float64, copy=False)
    skew, kurt = factor_skew_kurtosis(values)
    n_instruments = int(context.panel.index.get_level_values("instrument").nunique())
    return {
        "ic": summary["ic"],
        "icir": summary["icir"],
        "rank_ic": summary["rank_ic"],
        "n_days": summary["n_days"],
        "n_instruments": n_instruments,
        "factor_coverage": coverage(values),
        "factor_skewness": skew,
        "factor_kurtosis": kurt,
        "cs_pearson_autocorr": cross_sectional_lag1_pearson_autocorr(context.factor, min_pairs=min(30, max(n_instruments - 1, 2))),
        "decile_mean_label": decile_mean_label(values, context.label.to_numpy(dtype=np.float64, copy=False), n_deciles=10),
    }


@metric("size_neutral_decay")
def size_neutral_decay(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    """市值中性化 IC 衰减诊断（辅助信号，不参与规则门槛）。

    计算方式与提交入门的 winsorized_abs_ic_decay 同构：
      1. 对当前因子逐日按 log(float_cap) 截面回归取残差（市值中性化）
      2. 重算逐日 IC 均值，取绝对值
      3. decay = max(0, (|raw_ic| - |neutral_ic|) / |raw_ic|)

    decay 越大说明因子 alpha 越依赖小市值暴露，实盘风格反转时越危险；
    这个字段只用于人工/LLM 判断，不改变主口径 IC（主口径保持
    winsorize + zscore，不含市值残差化）。
    """
    field = str(params.get("field", "float_cap"))
    if field not in context.panel.columns:
        return {
            "size_neutral_ic": float("nan"),
            "size_neutral_abs_ic_decay": float("nan"),
            "note": f"panel 缺少市值字段 {field}",
        }
    from alphaagent.factor.metrics import (
        cross_sectional_ic,
        cross_sectional_size_neutralize_values,
    )

    raw_ic_series = context.daily_ic()
    raw_ic = float(raw_ic_series.mean()) if raw_ic_series.notna().any() else float("nan")

    neutral_values = cross_sectional_size_neutralize_values(
        context.factor.to_numpy(dtype=np.float64, copy=False),
        context.panel,
        market_cap_field=field,
    )
    neutral_factor = pd.Series(neutral_values, index=context.panel.index,
                               name=context.factor_name, dtype=np.float32)
    neutral_ic_series = cross_sectional_ic(neutral_factor, context.label, min_pairs=5)
    neutral_ic = float(neutral_ic_series.mean()) if neutral_ic_series.notna().any() else float("nan")

    raw_abs = abs(raw_ic)
    neutral_abs = abs(neutral_ic)
    if np.isfinite(raw_abs) and raw_abs > 0 and np.isfinite(neutral_abs):
        decay = float(max(0.0, (raw_abs - neutral_abs) / raw_abs))
    else:
        decay = float("nan")

    return {
        "size_neutral_ic": neutral_ic,
        "size_neutral_abs_ic_decay": decay,
        "note": "逐日截面 IC 对 log(float_cap) 回归取残差后重算；decay 越大市值暴露越重。",
    }


@metric("monthly_robustness")
def monthly_robustness(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    _ = params
    return monthly_ic_robustness(context.daily_ic())


def _shared_decile_means(context: EvaluationContext, n_deciles: int, min_stocks: int) -> dict[object, list[float]]:
    """同参数的逐日十分组 label 均值在同一评估内共享（mls_fmb 与 long_short_portfolio 各算一次太贵）。"""
    key = f"decile_means:{n_deciles}:{min_stocks}"
    if key not in context.cache:
        from alphaagent.factor.metrics import _compute_daily_decile_mean_labels
        context.cache[key] = _compute_daily_decile_mean_labels(
            context.factor, context.label, n_deciles=n_deciles, min_stocks=min_stocks,
        )
    return context.cache[key]


@metric("mls_fmb")
def mls_fmb(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    return mls_fmb_summary(
        context.factor,
        context.label,
        n_deciles=int(params.get("groups", 10)),
        min_stocks=int(params.get("min_stocks", 30)),
        nw_lags=params.get("nw_lags"),
        holding_days=getattr(context, "label_holding_days", 1),
    )


@metric("ic_series_diagnostics")
def ic_series_diagnostics(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    _ = params
    series = context.daily_ic()
    values = series[np.isfinite(series.to_numpy(dtype=float, copy=False))]
    if values.empty:
        return {"n_days": 0, "mean": float("nan"), "std": float("nan"), "skewness": float("nan"), "kurtosis": float("nan"), "top_decile_abs_contribution": float("nan")}
    absolute = np.abs(values.to_numpy(dtype=np.float64, copy=False))
    top_n = max(1, int(math.ceil(len(absolute) * 0.1)))
    contribution = float(np.sort(absolute)[-top_n:].sum() / absolute.sum()) if absolute.sum() > 0 else float("nan")
    return {
        "n_days": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "skewness": float(values.skew()) if len(values) >= 3 else float("nan"),
        "kurtosis": float(values.kurt()) if len(values) >= 4 else float("nan"),
        "top_decile_abs_contribution": contribution,
    }


@metric("long_short_portfolio")
def long_short_portfolio(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    groups = int(params.get("groups", 10))
    cost_bps = float(params.get("cost_bps", 0))
    min_stocks = int(params.get("min_stocks", 30))
    gross = daily_long_short_series(
        context.factor, context.label, n_deciles=groups, min_stocks=min_stocks,
        decile_means=_shared_decile_means(context, groups, min_stocks),
    )
    # 持有期对齐 label 名义持有期（label_20d → 20）：逐日多空收益重叠时
    # 按持有期节奏采样复利，年化因子同步缩放（与 quantile_portfolio 一致）。
    hold = max(1, int(getattr(context, "label_holding_days", 1)))
    if hold > 1:
        gross = gross.iloc[::hold]
    gross_valid = gross[np.isfinite(gross.to_numpy(dtype=float, copy=False))]
    # This is explicitly a full daily-rebalance cost assumption, not a turnover model.
    net = gross_valid - (2.0 * cost_bps / 10_000.0)
    nw = newey_west_mean_tstat(net, lags=max(1, int(hold)))
    std = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    ir = float(net.mean() / std) if std > 0 and np.isfinite(std) else float("nan")
    eff_annual = 252.0 / hold
    return {
        "groups": groups,
        "cost_bps_per_leg": cost_bps,
        "cost_model": "daily_full_rebalance_assumption",
        "n_days": int(len(net)),
        "gross_daily_mean": float(gross_valid.mean()) if len(gross_valid) else float("nan"),
        "net_daily_mean": float(net.mean()) if len(net) else float("nan"),
        "net_ir_annual": float(ir * math.sqrt(eff_annual)) if np.isfinite(ir) else float("nan"),
        "nw_t_net": nw["t_nw"],
        "nw_se_net": nw["se_nw"],
        "nw_lags": nw["lags"],
    }


@metric("topn_portfolio")
def topn_portfolio(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    """完整回测引擎（core.engine）口径的组合评估。

    选股方式默认为动态百分比（selection_pct），自动适配停牌/涨跌停导致的
    候选池缩放，避免固定 N 选到不可交易的尾部股票。
    T+1、整手、涨跌停、停牌、费率滑点与流动性参与率全部生效。
    """
    import numpy as np

    from alphaagent.factor.mining.engine_gate import panel_to_engine_frame, run_engine_gate

    panel = context.panel
    dts = panel.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())
    ic_series = context.daily_ic()
    ic_mean = float(np.nanmean(ic_series.to_numpy(dtype=float))) if ic_series.notna().any() else 0.0
    direction = 1 if ic_mean >= 0 else -1
    # direction 统一交给 engine_gate 处理（engine_gate.py scores * direction），
    # 此处不预乘，避免双重相乘。
    values = context.factor.to_numpy(dtype=np.float64)
    selection_pct = float(params.get("selection_pct", 0.02))
    selection_mode = str(params.get("selection_mode", "top_pct"))
    top_n_fallback = int(params.get("top_n", 100))

    # 门禁阈值交给 profile rules；此处只产出指标，阈值放行所有结果。
    lenient = {
        "enabled": True,
        "selection_mode": selection_mode,
        "selection_pct": selection_pct,
        "top_n": top_n_fallback,
        "min_annual_return": -9.0,
        "min_excess_annual": -9.0,
        "min_sharpe": -9.0,
        "max_drawdown": 9.0,
        "min_daily_overlap": 0.0,
    }

    def _unavailable(error: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "selection_mode": selection_mode,
            "selection_pct": selection_pct,
            "top_n": top_n_fallback,
            "direction": direction,
            "source": "core.engine",
            "window": {"start": start, "end": end},
            "available": False,
            "error": error,
        }
        for key in ("annualized_return", "annualized_excess_return", "sharpe",
                    "max_drawdown", "daily_overlap"):
            base[key] = float("nan")
        return base

    try:
        engine_frame = panel_to_engine_frame(panel)
    except Exception as exc:  # noqa: BLE001
        return _unavailable(f"engine_frame_failed: {exc}")

    def _run(freq: str) -> dict[str, Any]:
        gate = run_engine_gate(
            panel, values, val_start=start, val_end=end, direction=direction,
            policy={**lenient, "freq": freq}, engine_frame=engine_frame,
        )
        m = gate.get("metrics") or {}
        return {
            "rebalance": freq,
            "annualized_return": m.get("annual_return"),
            "annualized_excess_return": m.get("excess_annual"),
            "sharpe": m.get("sharpe"),
            "max_drawdown": m.get("max_drawdown"),
            "daily_overlap": m.get("daily_overlap"),
            "win_rate": m.get("win_rate"),
            "total_return": m.get("total_return"),
            "passed": gate.get("passed"),
        }

    try:
        by_freq = {freq: _run(freq) for freq in ("daily", "weekly", "monthly")}
    except Exception as exc:  # noqa: BLE001
        return _unavailable(f"engine_backtest_failed: {exc}")

    # 头条数字默认取 weekly：daily 全约束调仓摩擦过大，会把好因子也打成深负；
    # daily/weekly/monthly 完整结果保留在 by_freq 供下游展示与审计。
    headline_freq = str(params.get("headline_freq", "weekly"))
    if headline_freq not in by_freq:
        headline_freq = "weekly"
    out = dict(by_freq[headline_freq])
    out.update({
        "selection_mode": selection_mode,
        "selection_pct": selection_pct,
        "top_n": top_n_fallback,
        "direction": direction,
        "headline_freq": headline_freq,
        "source": "core.engine",
        "window": {"start": start, "end": end},
        "available": True,
        "by_freq": by_freq,
    })
    return out


@metric("quantile_portfolio")
def quantile_portfolio(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    """分位组合评估（纯多头，A 股口径）。

    每日按因子值 N 等分组，取最高组（Q_N）做纯多头等权持有，
    全市场等权作为基准。不涉及做空，不依赖固定选股数量。

    输出键（供 profile rules 引用）:
      - top_group_annualized_return
      - top_group_annualized_excess_return
      - top_group_sharpe
      - top_group_max_drawdown
      - monotonicity
    """
    from alphaagent.factor.metrics import quantile_portfolio_metrics

    n_groups = int(params.get("groups", 10))
    cost_bps = float(params.get("cost_bps", 0.0))
    min_stocks = int(params.get("min_stocks", 30))
    raw_direction = params.get("direction")
    direction = None if raw_direction is None else int(raw_direction)

    return quantile_portfolio_metrics(
        context.factor,
        context.label,
        n_groups=n_groups,
        min_stocks=min_stocks,
        cost_bps=cost_bps,
        direction=direction,
        holding_days=getattr(context, "label_holding_days", 1),
    )
