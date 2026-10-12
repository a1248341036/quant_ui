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
    label_f64,
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
        "decile_mean_label": decile_mean_label(values, label_f64(context.label), n_deciles=10),
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
      - depth_curve（L1，追加键）：同标签同成本下 top-k 深度曲线 +
        Q_N 参考行；诊断 alpha 集中在哪一层（引擎 top_pct 小组合口径
        缺口的主要来源）。depth_ks=None 时不产出。
      - depth_curve_tradable（L1，追加键）：可成交域透镜——先按引擎同源
        规则（一手可负担 / 次日非涨停停牌 / am20 流动性下限）收窄候选池
        再算同一条曲线；与 depth_curve 的差 = 可成交性损耗。
    """
    from alphaagent.factor.metrics import quantile_portfolio_metrics

    n_groups = int(params.get("groups", 10))
    cost_bps = float(params.get("cost_bps", 0.0))
    min_stocks = int(params.get("min_stocks", 30))
    raw_direction = params.get("direction")
    direction = None if raw_direction is None else int(raw_direction)

    raw_depth = params.get("depth_ks", (5, 10, 20, 50, 100))
    depth_ks = None
    if raw_depth not in (None, (), []):
        try:
            depth_ks = tuple(int(k) for k in raw_depth)
        except (TypeError, ValueError):
            depth_ks = None

    out = quantile_portfolio_metrics(
        context.factor,
        context.label,
        n_groups=n_groups,
        min_stocks=min_stocks,
        cost_bps=cost_bps,
        direction=direction,
        holding_days=getattr(context, "label_holding_days", 1),
        depth_ks=depth_ks,
    )

    # ── L1 可成交域透镜：第二路 depth_only 调用，主口径键零影响 ──
    if depth_ks and params.get("tradable_domain", True):
        try:
            from alphaagent.factor.metrics.tradable import tradable_mask
            from core import trading_config

            mask, diag = tradable_mask(
                context.panel,
                capital=float(params.get("tradable_capital", trading_config.GATE_CAPITAL)),
                target_count=int(params.get("tradable_top_n", trading_config.GATE_TOP_N)),
                lot_size=int(trading_config.LOT_SIZE),
                min_am20_yuan=float(
                    params.get("tradable_min_am20_yuan", trading_config.GATE_MIN_AM20_YUAN)
                ),
            )
            sub = quantile_portfolio_metrics(
                context.factor,
                context.label,
                n_groups=n_groups,
                min_stocks=min_stocks,
                cost_bps=cost_bps,
                direction=direction,
                holding_days=getattr(context, "label_holding_days", 1),
                depth_ks=depth_ks,
                eligibility=mask.to_numpy(dtype=bool),
                depth_only=True,
            )
            sub_curve = sub.get("depth_curve")
            out["depth_curve_tradable"] = sub_curve
            out["tradable_domain"] = {
                "available": bool(sub.get("available")),
                "mask_coverage": diag.get("coverage"),
                "avg_daily_coverage": diag.get("avg_daily_coverage"),
                "min_daily_coverage": diag.get("min_daily_coverage"),
                "budget_per_name": diag.get("budget_per_name"),
                "affordable_max_price": diag.get("affordable_max_price"),
                "note": (
                    "可成交域透镜：先按引擎同源规则收窄候选池（一手可负担=执行日开盘价×"
                    "lot_size<=等权预算、次日非涨停/停牌、am20>=流动性下限），再算同一条"
                    "深度曲线。与 depth_curve 的差 = 可成交性损耗；不复制引擎盈亏口径，"
                    "最终裁决仍是 engine_gate。"
                ),
            }
        except Exception as exc:  # noqa: BLE001 — 透镜失败不影响主指标
            out["depth_curve_tradable"] = None
            out["tradable_domain"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return out

# ── 行业/风格轮动诊断 metrics（P1，2026-10-12；只披露、不参与任何准入规则）──


def _industry_ic_table(
    context, field: str, min_cover: int,
) -> tuple[pd.Series, pd.Series] | None:
    """按 (日, 行业) 组计算行业内 Rank IC。

    返回 (ic, weights)：ic 为 MultiIndex (date, industry) 的 Spearman IC，
    weights 为对应组的有效样本数（后续按组内样本加权）。组内样本数 < min_cover
    的组被剔除；无任何有效组返回 None。算法与 cross_sectional_rank_ic 同构：
    组内 rank 化后按 Pearson 公式（ddof=1）一次聚合，避免逐日 Python 循环。

    结果按 (field, min_cover) 缓存到 context.cache：同一 profile 评估内
    industry_within_rank_ic 与 industry_ic_dispersion 总是成对出现，共享
    一次矩阵计算（挖掘循环对每因子 × 3 profile 各少一遍 groupby.rank）。
    """
    cache_key = ("industry_ic_table", field, int(min_cover))
    if cache_key in context.cache:
        return context.cache[cache_key]

    dates = context.panel.index.get_level_values("datetime")
    industry = context.panel[field].to_numpy(copy=False)
    factor = context.factor.to_numpy(dtype=np.float64, copy=False)
    label = context.label.to_numpy(dtype=np.float64, copy=False)
    df = pd.DataFrame({"d": dates, "g": industry, "f": factor, "y": label})
    df = df.dropna(subset=["f", "y", "g"])
    table = None
    if not df.empty:
        df["fr"] = df.groupby(["d", "g"], sort=False)["f"].rank()
        df["yr"] = df.groupby(["d", "g"], sort=False)["y"].rank()
        df["xy"] = df["fr"] * df["yr"]
        agg = df.groupby(["d", "g"], sort=False).agg(
            n=("xy", "size"), ex=("fr", "mean"), ey=("yr", "mean"),
            exy=("xy", "mean"), sx=("fr", "std"), sy=("yr", "std"),
        )
        agg = agg[agg["n"] >= max(int(min_cover), 2)]
        if not agg.empty:
            n = agg["n"].to_numpy(dtype=np.float64)
            cov = (agg["exy"].to_numpy() - agg["ex"].to_numpy() * agg["ey"].to_numpy()) * n / (n - 1)
            denom = agg["sx"].to_numpy() * agg["sy"].to_numpy()
            safe_denom = np.where(denom > 1e-12, denom, np.nan)
            ic_vals = cov / safe_denom
            ic = pd.Series(ic_vals, index=agg.index, name="industry_ic")
            ic = ic[np.isfinite(ic.to_numpy(dtype=float, copy=False))]
            if not ic.empty:
                table = ic, agg.loc[ic.index, "n"]
    context.cache[cache_key] = table
    return table


def _missing_industry(field: str) -> dict[str, Any]:
    return {
        "within_rank_ic": float("nan"),
        "market_rank_ic": float("nan"),
        "within_minus_market": float("nan"),
        "n_days": 0,
        "n_industries_used": 0,
        "note": f"panel 缺少行业字段 {field}（P1 轮动诊断未计算）",
    }


@metric("industry_within_rank_ic")
def industry_within_rank_ic(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    """行业内 Rank IC：按行业分组、组内日度 Spearman，按组内样本数加权。

    与全市场 Rank IC 对比：within 显著低于 market ⇒ 因子 alpha 主要是行业贝塔
    而非行业内选股能力（行业轮动假阳性的第一道观测）。只披露，不设门槛。
    """
    field = str(params.get("field", "industry_sw_l1"))
    min_cover = int(params.get("min_cover", 15))
    if field not in context.panel.columns:
        return _missing_industry(field)
    table = _industry_ic_table(context, field, min_cover)
    if table is None:
        return {**_missing_industry(field), "note": "无有效行业组（行业内样本不足）"}
    ic, weights = table
    w = weights.to_numpy(dtype=np.float64)
    total = float(w.sum())
    within_ic = float(np.sum(ic.to_numpy(dtype=np.float64) * w) / total) if total > 0 else float("nan")
    market = context.daily_rank_ic()
    market_ic = float(market.mean()) if market.notna().any() else float("nan")
    diff = within_ic - market_ic if np.isfinite(market_ic) else float("nan")
    return {
        "within_rank_ic": within_ic,
        "market_rank_ic": market_ic,
        "within_minus_market": diff,
        "n_days": int(ic.index.get_level_values(0).nunique()),
        "n_industries_used": int(ic.index.get_level_values(1).nunique()),
        "note": "行业内日度 Rank IC 按行业有效样本数加权；within_minus_market<0 越大=越依赖行业贝塔。",
    }


@metric("industry_exposure_r2")
def industry_exposure_r2(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    """行业暴露 R²：逐日因子值对行业组均值的方差解释占比（SS_between/SS_total）。

    R²=0 无行业暴露；≈1 纯行业哑变量。配合 industry_within_rank_ic 判断
    「全市场 IC 高、行业内 IC 低、暴露高」= 行业 beta 而非行业内 alpha。
    """
    field = str(params.get("field", "industry_sw_l1"))
    if field not in context.panel.columns:
        return {"exposure_r2": float("nan"), "n_days": 0, "note": f"panel 缺少行业字段 {field}"}
    dates = context.panel.index.get_level_values("datetime")
    industry = context.panel[field].to_numpy(copy=False)
    factor = context.factor.to_numpy(dtype=np.float64, copy=False)
    df = pd.DataFrame({"d": dates, "g": industry, "f": factor}).dropna(subset=["f", "g"])
    if df.empty:
        return {"exposure_r2": float("nan"), "n_days": 0, "note": "无有效因子/行业对"}
    day_mean = df.groupby("d")["f"].transform("mean")
    group_mean = df.groupby(["d", "g"])["f"].transform("mean")
    ss_total = (df["f"] - day_mean).pow(2).groupby(df["d"]).sum()
    ss_between = (group_mean - day_mean).pow(2).groupby(df["d"]).sum()
    r2 = (ss_between / ss_total.replace(0, np.nan)).dropna()
    if r2.empty:
        return {"exposure_r2": float("nan"), "n_days": 0, "note": "方差分解退化（单行业/单股票日）"}
    return {
        "exposure_r2": float(r2.mean()),
        "n_days": int(len(r2)),
        "note": "逐日 SS_between/SS_total 均值；≈1 表示因子几乎完全由行业哑变量解释。",
    }


@metric("industry_ic_dispersion")
def industry_ic_dispersion(context: EvaluationContext, params: dict[str, Any]) -> dict[str, Any]:
    """行业 IC 离散度：各行业整段 IC 均值的截面分布。

    行业间差异大 ⇒ 「行业间有差异化 alpha」的轮动信号候选；差异小 ⇒ 因子各行业
    普适（稳定 alpha 而非轮动）。每行业至少 min_days 个有效日才纳入统计。
    """
    field = str(params.get("field", "industry_sw_l1"))
    min_cover = int(params.get("min_cover", 15))
    min_days = int(params.get("min_days", 20))
    if field not in context.panel.columns:
        return {"dispersion_std": float("nan"), "dispersion_iqr": float("nan"),
                "mean_ic": float("nan"), "n_industries": 0, "top3": {}, "bottom3": {},
                "note": f"panel 缺少行业字段 {field}"}
    table = _industry_ic_table(context, field, min_cover)
    if table is None:
        return {"dispersion_std": float("nan"), "dispersion_iqr": float("nan"),
                "mean_ic": float("nan"), "n_industries": 0, "top3": {}, "bottom3": {},
                "note": "无有效行业组（行业内样本不足）"}
    ic, _ = table
    df = ic.reset_index()
    df.columns = ["d", "g", "ic"]
    per = df.groupby("g")["ic"].agg(n_days="count", mean="mean")
    per = per[per["n_days"] >= max(int(min_days), 1)]
    if per.empty:
        return {"dispersion_std": float("nan"), "dispersion_iqr": float("nan"),
                "mean_ic": float("nan"), "n_industries": 0, "top3": {}, "bottom3": {},
                "note": f"无行业有效日 ≥ {min_days}（min_days 过大或样本过短）"}
    means = per["mean"]
    q1, q3 = np.nanpercentile(means.to_numpy(dtype=float), [25, 75])
    rounded = lambda kv: {str(k): round(float(v), 5) for k, v in kv.items()}  # noqa: E731
    return {
        "dispersion_std": float(means.std(ddof=1)) if len(means) > 1 else float("nan"),
        "dispersion_iqr": float(q3 - q1) if np.isfinite(q1) and np.isfinite(q3) else float("nan"),
        "mean_ic": float(means.mean()),
        "n_industries": int(len(means)),
        "top3": rounded(means.nlargest(3).to_dict()),
        "bottom3": rounded(means.nsmallest(3).to_dict()),
        "note": "行业 IC 均值截面 std/IQR；top3/bottom3 为最强/最弱行业（展示用，不设门槛）。",
    }
