"""Built-in transforms and metrics for the plugin evaluation engine."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from alphaagent.factor.evaluation.context import EvaluationContext
from alphaagent.factor.metrics import (
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
        raise ValueError(f"size_residualize_missing_field:{field}")
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
    summary = cs_ic_summary(daily_ic, daily_rank_ic)
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


@metric("monthly_robustness")
def monthly_robustness(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    _ = params
    return monthly_ic_robustness(context.daily_ic())


@metric("mls_fmb")
def mls_fmb(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    return mls_fmb_summary(
        context.factor,
        context.label,
        n_deciles=int(params.get("groups", 10)),
        min_stocks=int(params.get("min_stocks", 30)),
        nw_lags=params.get("nw_lags"),
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
    gross = daily_long_short_series(context.factor, context.label, n_deciles=groups, min_stocks=min_stocks)
    gross_valid = gross[np.isfinite(gross.to_numpy(dtype=float, copy=False))]
    # This is explicitly a full daily-rebalance cost assumption, not a turnover model.
    net = gross_valid - (2.0 * cost_bps / 10_000.0)
    nw = newey_west_mean_tstat(net)
    std = float(net.std(ddof=1)) if len(net) > 1 else float("nan")
    ir = float(net.mean() / std) if std > 0 and np.isfinite(std) else float("nan")
    return {
        "groups": groups,
        "cost_bps_per_leg": cost_bps,
        "cost_model": "daily_full_rebalance_assumption",
        "n_days": int(len(net)),
        "gross_daily_mean": float(gross_valid.mean()) if len(gross_valid) else float("nan"),
        "net_daily_mean": float(net.mean()) if len(net) else float("nan"),
        "net_ir_annual": float(ir * math.sqrt(252)) if np.isfinite(ir) else float("nan"),
        "nw_t_net": nw["t_nw"],
        "nw_se_net": nw["se_nw"],
        "nw_lags": nw["lags"],
    }
