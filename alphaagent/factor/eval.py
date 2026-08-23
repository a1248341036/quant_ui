"""因子评估：DSL 求值 + IC/MLS 指标。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.dsl import eval_factor
from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.factor.align import align_series_to_panel
from alphaagent.factor.metrics import (
    coverage,
    cross_sectional_ic,
    cross_sectional_lag1_pearson_autocorr,
    cross_sectional_rank_ic,
    cs_ic_summary,
    decile_mean_label,
    factor_skew_kurtosis,
    label_quantile_buckets,
    monthly_ic_robustness,
    mls_fmb_summary,
    pearson_ic,
)

from alphaagent.data.panel import slice_panel
from alphaagent.factor.metrics import evaluate_on_panel
from alphaagent.factor.types import DEFAULT_LABEL_COL


def evaluate_factor(
    expr: str,
    panel: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    start: str | None = None,
    end: str | None = None,
    min_pairs: int = 5,
) -> dict[str, Any]:
    """在全量 panel 上求值 DSL，再按日期窗计算 IC/ICIR/RANKIC/MLS 等指标。"""
    panel_full = panel.sort_index()
    if label_col not in panel_full.columns:
        raise KeyError(f"panel 缺少标签列: {label_col}")
    out = eval_factor(expr, panel_full)
    if not isinstance(out, pd.Series):
        raise TypeError(f"因子输出须为 Series，得到 {type(out)!r}")
    values = align_series_to_panel(out, panel_full)
    eval_panel = slice_panel(panel_full, start=start, end=end)
    if eval_panel.empty:
        raise ValueError(f"评估切片为空: start={start!r} end={end!r}")
    pos = panel_full.index.isin(eval_panel.index)
    metrics = evaluate_on_panel(
        values[pos], eval_panel, label_col=label_col, min_ic_pairs=min_pairs
    )
    metrics["eval_start"] = start
    metrics["eval_end"] = end
    metrics["label_col"] = label_col
    return metrics




def _build_summary(
    factor_series: pd.Series,
    label_series: pd.Series,
    values: np.ndarray,
    daily_ic: pd.Series,
    daily_rank_ic: pd.Series,
    *,
    min_cs_autocorr_pairs: int = 30,
) -> dict[str, Any]:
    cs = cs_ic_summary(daily_ic, daily_rank_ic)
    n_inst = int(factor_series.index.get_level_values("instrument").nunique())
    skew, kurt = factor_skew_kurtosis(values)
    fac = factor_series.to_numpy(dtype=float, copy=False)
    lab = label_series.to_numpy(dtype=float, copy=False)
    cs_autocorr_pairs = min(min_cs_autocorr_pairs, max(n_inst - 1, 2))
    mls_min_stocks = min(30, max(n_inst, 10))
    return {
        "ic": cs["ic"],
        "icir": cs["icir"],
        "rank_ic": cs["rank_ic"],
        "n_days": cs["n_days"],
        "n_instruments": n_inst,
        "factor_coverage": coverage(values),
        "factor_skewness": skew,
        "factor_kurtosis": kurt,
        "cs_pearson_autocorr": cross_sectional_lag1_pearson_autocorr(
            factor_series,
            min_pairs=cs_autocorr_pairs,
        ),
        "decile_mean_label": decile_mean_label(fac, lab, n_deciles=10),
        "mls_fmb": mls_fmb_summary(
            factor_series,
            label_series,
            min_stocks=mls_min_stocks,
        ),
    }


def _detail_tables(
    factor_series: pd.Series,
    label_series: pd.Series,
    daily_ic: pd.Series,
    daily_rank_ic: pd.Series,
) -> dict[str, Any]:
    by_month_rows: list[dict[str, Any]] = []
    if not daily_ic.empty:
        s = daily_ic.copy()
        s.index = pd.to_datetime(s.index, errors="coerce")
        r = daily_rank_ic.copy()
        r.index = pd.to_datetime(r.index, errors="coerce")
        for month, grp in s.groupby(s.index.to_period("M"), sort=True):
            r_grp = r.loc[grp.index]
            by_month_rows.append(
                {
                    "month": str(month),
                    "mean_ic": float(grp.mean(skipna=True)),
                    "mean_rank_ic": float(r_grp.mean(skipna=True)),
                    "n_days": int(grp.notna().sum()),
                }
            )

    by_symbol_rows: list[dict[str, Any]] = []
    for inst, f_sub in factor_series.groupby(level="instrument", sort=False):
        y_sub = label_series.xs(inst, level="instrument")
        ts_ic = pearson_ic(
            f_sub.to_numpy(dtype=np.float64, copy=False),
            y_sub.to_numpy(dtype=np.float64, copy=False),
            min_pairs=5,
        )
        by_symbol_rows.append({"instrument": str(inst), "ts_ic": ts_ic})

    return {"by_month": by_month_rows, "by_symbol": by_symbol_rows}


def evaluate_factor_on_split(
    session,
    *,
    split: str,
    multi_line_expr: str,
    factor_name: str = "expr",
    include_detail_tables: bool = False,
    label_quantile_n: int = 10,
) -> dict[str, Any]:
    """在 train/val 窗上评估多行 DSL，返回未格式化的原始结果 dict。"""
    ctx = session.ctx
    panel, start, end = session.get_split_panel(split)
    date_range = {"start": start, "end": end}

    if panel.empty:
        return {
            "ok": False,
            "error": "日期窗内无 panel 数据",
            "error_type": "EmptyData",
            "split": split,
            "date_range": date_range,
        }

    if ctx.label_col not in panel.columns:
        return {
            "ok": False,
            "error": f"panel 缺少标签列: {ctx.label_col}",
            "error_type": "MissingLabelColumn",
            "split": split,
            "date_range": date_range,
        }

    timing: dict[str, float] = {}
    t0 = time.perf_counter()

    try:
        t_eval = time.perf_counter()
        out = eval_factor(multi_line_expr, panel)
        timing["eval_ms"] = (time.perf_counter() - t_eval) * 1000
    except MultiLineFactorEvalError as e:
        return {
            "ok": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "split": split,
            "date_range": date_range,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "split": split,
            "date_range": date_range,
        }

    if not isinstance(out, pd.Series):
        return {
            "ok": False,
            "error": f"因子输出须为 Series，得到 {type(out)!r}",
            "error_type": "TypeError",
            "split": split,
            "date_range": date_range,
        }

    t_align = time.perf_counter()
    values = align_series_to_panel(out, panel)
    factor_series = pd.Series(values, index=panel.index, name=factor_name, dtype=np.float32)
    label_series = panel[ctx.label_col]
    timing["align_ms"] = (time.perf_counter() - t_align) * 1000

    t_metrics = time.perf_counter()
    daily_ic = cross_sectional_ic(factor_series, label_series, min_pairs=5)
    daily_rank_ic = cross_sectional_rank_ic(factor_series, label_series, min_pairs=5)
    summary = _build_summary(factor_series, label_series, values, daily_ic, daily_rank_ic)
    monthly_rob = monthly_ic_robustness(daily_ic)

    buckets: list[dict[str, Any]] = []
    if label_quantile_n >= 2:
        buckets = label_quantile_buckets(
            factor_series.to_numpy(dtype=float, copy=False),
            label_series.to_numpy(dtype=float, copy=False),
            n_quantiles=label_quantile_n,
        )

    detail: dict[str, Any] = {}
    if include_detail_tables:
        detail = _detail_tables(factor_series, label_series, daily_ic, daily_rank_ic)

    timing["metrics_ms"] = (time.perf_counter() - t_metrics) * 1000
    timing["total_ms"] = (time.perf_counter() - t0) * 1000

    result: dict[str, Any] = {
        "ok": True,
        "split": split,
        "date_range": date_range,
        "eval_wall_seconds": timing["total_ms"] / 1000.0,
        "timing_ms": timing,
        "summary": summary,
        "monthly_corr_robustness": monthly_rob,
        "label_quantile_buckets": buckets,
        "label_quantile_n": int(label_quantile_n),
        "label_col": ctx.label_col,
        "bar_interval": "1d",
    }
    result.update(detail)
    return result
