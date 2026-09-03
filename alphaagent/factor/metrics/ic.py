"""metrics 子模块：截面 IC / ICIR / 评估入口。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphaagent.factor import metrics_fast as _mf
from alphaagent.factor.types import DEFAULT_LABEL_COL

from ._core import coverage, pearson_ic, spearman_ic
from .decile import decile_mean_label, _compute_daily_decile_mean_labels


def cross_sectional_ic(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    min_pairs: int = 10,
    _day_slices=None,
) -> pd.Series:
    """按 datetime 截面 Pearson IC，返回逐日 IC 序列。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("cross_sectional_ic 需要 MultiIndex 面板 (datetime, instrument)")
    if time_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r}")

    f_arr = factor.to_numpy(dtype=np.float64, copy=False)
    l_arr = label.to_numpy(dtype=np.float64, copy=False)
    slices = _day_slices(factor.index, time_level)
    if slices is not None:
        bounds, day_vals = slices
        if _mf.HAS_NUMBA:
            rows = _mf.pearson_ic_days(f_arr, l_arr, bounds, min_pairs)
        else:
            rows = [
                pearson_ic(f_arr[st:en], l_arr[st:en], min_pairs=min_pairs)
                for st, en in zip(bounds[:-1].tolist(), bounds[1:].tolist())
            ]
        return pd.Series(rows, index=pd.Index(day_vals, name=time_level), dtype=float)

    rows: list[float] = []
    idx: list[object] = []
    grouped = factor.groupby(level=time_level, sort=False)
    for ts, f_sub in grouped:
        y_sub = label.xs(ts, level=time_level)
        ic = pearson_ic(
            f_sub.to_numpy(dtype=np.float64, copy=False),
            y_sub.to_numpy(dtype=np.float64, copy=False),
            min_pairs=min_pairs,
        )
        rows.append(ic)
        idx.append(ts)
    return pd.Series(rows, index=pd.Index(idx, name=time_level), dtype=float)


def cross_sectional_rank_ic(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    min_pairs: int = 10,
    _day_slices=None,
) -> pd.Series:
    """按 datetime 截面 Spearman Rank IC，返回逐日 RANKIC 序列。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("cross_sectional_rank_ic 需要 MultiIndex 面板 (datetime, instrument)")
    if time_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r}")

    f_arr = factor.to_numpy(dtype=np.float64, copy=False)
    l_arr = label.to_numpy(dtype=np.float64, copy=False)
    slices = _day_slices(factor.index, time_level)
    if slices is not None:
        bounds, day_vals = slices
        if _mf.HAS_NUMBA:
            rows = _mf.rank_ic_days(f_arr, l_arr, bounds, min_pairs)
        else:
            rows = [
                spearman_ic(f_arr[st:en], l_arr[st:en], min_pairs=min_pairs)
                for st, en in zip(bounds[:-1].tolist(), bounds[1:].tolist())
            ]
        return pd.Series(rows, index=pd.Index(day_vals, name=time_level), dtype=float)

    rows: list[float] = []
    idx: list[object] = []
    grouped = factor.groupby(level=time_level, sort=False)
    for ts, f_sub in grouped:
        y_sub = label.xs(ts, level=time_level)
        ic = spearman_ic(
            f_sub.to_numpy(dtype=np.float64, copy=False),
            y_sub.to_numpy(dtype=np.float64, copy=False),
            min_pairs=min_pairs,
        )
        rows.append(ic)
        idx.append(ts)
    return pd.Series(rows, index=pd.Index(idx, name=time_level), dtype=float)


def cross_sectional_lag1_pearson_autocorr_series(
    factor: pd.Series,
    *,
    time_level: str = "datetime",
    instrument_level: str = "instrument",
    min_pairs: int = 30,
    _day_slices=None,
) -> pd.Series:
    """逐日横截面 lag-1 Pearson 自相关：corr_CS(f_t, f_{t-1})。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("cross_sectional_lag1_pearson_autocorr_series 需要 MultiIndex 面板")
    if time_level not in factor.index.names or instrument_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r} 或 {instrument_level!r}")

    lag1 = factor.groupby(level=instrument_level, sort=False).shift(1)
    f_arr = factor.to_numpy(dtype=np.float64, copy=False)
    g_arr = lag1.to_numpy(dtype=np.float64, copy=False)
    slices = _day_slices(factor.index, time_level)
    if slices is not None:
        bounds, day_vals = slices
        if _mf.HAS_NUMBA:
            rows = _mf.pearson_ic_days(f_arr, g_arr, bounds, min_pairs)
        else:
            rows = [
                pearson_ic(f_arr[st:en], g_arr[st:en], min_pairs=min_pairs)
                for st, en in zip(bounds[:-1].tolist(), bounds[1:].tolist())
            ]
        return pd.Series(rows, index=pd.Index(day_vals, name=time_level), dtype=float)

    rows: list[float] = []
    idx: list[object] = []
    for ts, cur in factor.groupby(level=time_level, sort=False):
        prev = lag1.xs(ts, level=time_level)
        corr = pearson_ic(
            cur.to_numpy(dtype=np.float64, copy=False),
            prev.to_numpy(dtype=np.float64, copy=False),
            min_pairs=min_pairs,
        )
        rows.append(corr)
        idx.append(ts)
    return pd.Series(rows, index=pd.Index(idx, name=time_level), dtype=float)


def cross_sectional_lag1_pearson_autocorr(
    factor: pd.Series,
    *,
    min_pairs: int = 30,
    _day_slices=None,
) -> float:
    """逐日横截面 lag-1 Pearson 自相关的均值。"""
    daily = cross_sectional_lag1_pearson_autocorr_series(factor, min_pairs=min_pairs, _day_slices=_day_slices)
    finite = daily[np.isfinite(daily.to_numpy(dtype=float, copy=False))]
    return float(finite.mean()) if len(finite) else float("nan")


def cs_ic_summary(
    daily_ic: pd.Series,
    daily_rank_ic: pd.Series | None = None,
    *,
    holding_days: int = 1,
) -> dict[str, float | int]:
    """由 IC / RANKIC 汇总 IC、ICIR、RANKIC。

    持有期 >1（如 label_20d → 20）时逐日 IC 来自重叠收益，std 被低估导致
    ICIR 虚高。此处按持有期节奏重采样去重叠（每 hold 个交易日取一点），与
    quantile_portfolio 的采样口径一致；label_1d 时 hold=1 无任何变化。
    """
    hold = max(1, int(holding_days))
    if hold > 1 and len(daily_ic) > hold:
        daily_ic = daily_ic.iloc[::hold]
        if daily_rank_ic is not None:
            daily_rank_ic = daily_rank_ic.iloc[::hold]
    ic_vals = daily_ic[np.isfinite(daily_ic.to_numpy(dtype=float, copy=False))]
    n_days = int(len(ic_vals))
    if n_days == 0:
        return {"ic": float("nan"), "icir": float("nan"), "rank_ic": float("nan"), "n_days": 0}

    ic_mean = float(ic_vals.mean())
    ic_std = float(ic_vals.std(ddof=1)) if n_days > 1 else float("nan")
    icir = ic_mean / ic_std if ic_std and np.isfinite(ic_std) and ic_std > 0 else float("nan")

    rank_ic_mean = float("nan")
    if daily_rank_ic is not None:
        rv = daily_rank_ic[np.isfinite(daily_rank_ic.to_numpy(dtype=float, copy=False))]
        rank_ic_mean = float(rv.mean()) if len(rv) else float("nan")

    return {
        "ic": ic_mean,
        "icir": icir,
        "rank_ic": rank_ic_mean,
        "n_days": n_days,
    }


def evaluate_cs_on_panel(
    values: np.ndarray,
    panel: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    min_pairs: int = 5,
    holding_days: int = 1,
    _day_slices=None,
    _fast_equal_freq_codes=None,
) -> dict[str, Any]:
    """在 panel 上计算截面 IC / ICIR / RANKIC 与 coverage。"""
    panel = panel.sort_index()
    if label_col not in panel.columns:
        raise ValueError(f"panel 缺少标签列: {label_col}")
    if len(values) != len(panel):
        raise ValueError(f"values 长度 {len(values)} != panel 行数 {len(panel)}")

    factor_series = pd.Series(values, index=panel.index, dtype=np.float32)
    label_series = panel[label_col]
    daily_ic = cross_sectional_ic(factor_series, label_series, min_pairs=min_pairs, _day_slices=_day_slices)
    daily_rank_ic = cross_sectional_rank_ic(factor_series, label_series, min_pairs=min_pairs, _day_slices=_day_slices)
    cs = cs_ic_summary(daily_ic, daily_rank_ic, holding_days=holding_days)
    cs_autocorr_min_pairs = min(30, max(int(panel.index.get_level_values("instrument").nunique()) - 1, min_pairs))
    mls_min_stocks = min(30, max(min_pairs * 6, 10))
    fac = factor_series.to_numpy(dtype=float, copy=False)
    lab = label_series.to_numpy(dtype=float, copy=False)

    from .mls import mls_fmb_summary
    return {
        "coverage": coverage(values),
        "ic": cs["ic"],
        "icir": cs["icir"],
        "rank_ic": cs["rank_ic"],
        "n_days": cs["n_days"],
        "cs_pearson_autocorr": cross_sectional_lag1_pearson_autocorr(
            factor_series,
            min_pairs=cs_autocorr_min_pairs,
            _day_slices=_day_slices,
        ),
        "decile_mean_label": decile_mean_label(fac, lab, n_deciles=10),
        "mls_fmb": mls_fmb_summary(
            factor_series,
            label_series,
            min_stocks=mls_min_stocks,
            _day_slices=_day_slices,
            _fast_equal_freq_codes=_fast_equal_freq_codes,
        ),
        "label_col": label_col,
    }


def evaluate_on_panel(
    values: np.ndarray,
    panel: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    min_ic_pairs: int = 5,
    holding_days: int = 1,
    _day_slices=None,
    _fast_equal_freq_codes=None,
) -> dict[str, Any]:
    """在 panel 上计算 coverage 与截面 IC / ICIR / RANKIC。"""
    return evaluate_cs_on_panel(
        values,
        panel,
        label_col=label_col,
        min_pairs=min_ic_pairs,
        holding_days=holding_days,
        _day_slices=_day_slices,
        _fast_equal_freq_codes=_fast_equal_freq_codes,
    )


def cross_sectional_winsorize_values(
    values: np.ndarray,
    panel: pd.DataFrame,
    *,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> np.ndarray:
    """对每个交易日的有效因子值按分位数截尾，保留原 panel 行序。"""
    if len(values) != len(panel):
        raise ValueError(f"values 长度 {len(values)} != panel 行数 {len(panel)}")
    if not isinstance(panel.index, pd.MultiIndex) or "datetime" not in panel.index.names:
        raise ValueError("cross_sectional_winsorize_values 需要含 datetime 的 MultiIndex panel")

    out = np.asarray(values, dtype=np.float32).copy()
    dates = panel.index.get_level_values("datetime")
    for _, positions in pd.Series(np.arange(len(out)), index=dates).groupby(level=0, sort=False):
        idx = positions.to_numpy(dtype=np.int64, copy=False)
        current = out[idx]
        finite = np.isfinite(current)
        if int(finite.sum()) < 2:
            continue
        lower, upper = np.nanpercentile(current[finite], [lower_pct, upper_pct])
        current[finite] = np.clip(current[finite], lower, upper)
        out[idx] = current
    return out


def cross_sectional_size_neutralize_values(
    values: np.ndarray,
    panel: pd.DataFrame,
    *,
    market_cap_field: str = "float_cap",
    log_scale: bool = True,
    min_valid: int = 3,
    _day_slices=None,
) -> np.ndarray:
    """对每个交易日按市值字段回归取残差（市值中性化），保留原 panel 行序。

    与 size_residualize transform 同口径（逐日对 log(float_cap) 做截面线性回归
    取残差），供 size_neutral_decay 诊断 metric 复用，避免 plugin/ingest 双写。
    """
    if len(values) != len(panel):
        raise ValueError(f"values 长度 {len(values)} != panel 行数 {len(panel)}")
    if not isinstance(panel.index, pd.MultiIndex) or "datetime" not in panel.index.names:
        raise ValueError("cross_sectional_size_neutralize_values 需要含 datetime 的 MultiIndex panel")
    if market_cap_field not in panel.columns:
        raise ValueError(f"panel 缺少市值字段: {market_cap_field}")

    out = np.asarray(values, dtype=np.float64).copy()
    size = panel[market_cap_field].to_numpy(dtype=np.float64, copy=False)
    if log_scale:
        size = np.where(size > 0, np.log(size), np.nan)
    slices = _day_slices(panel.index) if _day_slices else None
    if slices is not None and _mf.HAS_NUMBA:
        bounds, _ = slices
        _mf.size_resid_days(out, size, bounds, min_valid)
        return out.astype(np.float32, copy=False)
    if slices is not None:
        bounds, _ = slices
        day_indices = (
            np.arange(st, en) for st, en in zip(bounds[:-1].tolist(), bounds[1:].tolist())
        )
    else:
        dates = panel.index.get_level_values("datetime")
        day_indices = (
            positions.to_numpy(dtype=np.int64, copy=False)
            for _, positions in pd.Series(np.arange(len(out)), index=dates).groupby(level=0, sort=False)
        )
    for idx in day_indices:
        y, x = out[idx], size[idx]
        valid = np.isfinite(y) & np.isfinite(x)
        if int(valid.sum()) < min_valid:
            continue
        x_valid, y_valid = x[valid], y[valid]
        variance = float(np.var(x_valid))
        if not np.isfinite(variance) or variance <= 1e-15:
            continue
        beta = float(np.cov(x_valid, y_valid, ddof=0)[0, 1] / variance)
        alpha = float(np.mean(y_valid) - beta * np.mean(x_valid))
        y[valid] = y_valid - (alpha + beta * x_valid)
        out[idx] = y
    return out.astype(np.float32, copy=False)


def annualized_long_group_excess_return(
    factor: pd.Series,
    label: pd.Series,
    *,
    direction: int,
    n_deciles: int = 10,
    min_stocks: int = 30,
    annualization_factor: float = 252.0,
    holding_days: int = 1,
) -> float:
    """方向自适应的多头组相对当日全市场等权收益的复利年化超额。

    ``holding_days`` 对齐 label 名义持有期（label_20d → 20）：label>1 时
    相邻日超额重叠，按持有期节奏重采样复利，且年化因子同步缩小为
    ``annualization_factor / hold``（与 quantile_portfolio 口径一致），
    避免 20 日收益被逐日重叠计入并虚高年化 ~3.6 倍。label_1d 时 hold=1
    完全退化为原行为。
    """
    hold = max(1, int(holding_days))
    eff_annual = annualization_factor / hold
    excesses: list[float] = []
    for i, (_, f_sub) in enumerate(factor.groupby(level="datetime", sort=False)):
        # 持有期 >1 时按节奏采样，去掉重叠收益
        if hold > 1 and i % hold != 0:
            continue
        y_sub = label.xs(f_sub.index.get_level_values("datetime")[0], level="datetime")
        xf = f_sub.to_numpy(dtype=np.float64, copy=False)
        yl = y_sub.to_numpy(dtype=np.float64, copy=False)
        mask = np.isfinite(xf) & np.isfinite(yl)
        if int(mask.sum()) < max(min_stocks, n_deciles):
            continue
        xf = xf[mask]
        returns = yl[mask]
        try:
            bins = pd.qcut(pd.Series(xf), n_deciles, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.qcut(pd.Series(xf).rank(method="first"), n_deciles, labels=False, duplicates="drop")
        valid_bins = np.asarray(bins, dtype=float)
        if not np.isfinite(valid_bins).any():
            continue
        chosen = np.nanmax(valid_bins) if direction >= 0 else np.nanmin(valid_bins)
        long_returns = returns[valid_bins == chosen]
        if len(long_returns) == 0:
            continue
        excesses.append(float(np.mean(long_returns) - np.mean(returns)))
    if not excesses:
        return float("nan")
    arr = np.asarray(excesses, dtype=np.float64)
    if np.any(arr <= -1.0):
        return float("nan")
    return float(np.prod(1.0 + arr) ** (eff_annual / len(arr)) - 1.0)


def monthly_detail_rows(
    daily_ic: pd.Series,
    daily_rank_ic: pd.Series,
) -> list[dict[str, Any]]:
    """逐日 IC/RankIC → 按月聚合明细（mean_ic / mean_rank_ic / n_days）。"""
    by_month_rows: list[dict[str, Any]] = []
    if daily_ic.empty:
        return by_month_rows
    s = daily_ic.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    r = daily_rank_ic.copy()
    r.index = pd.to_datetime(r.index, errors="coerce")
    for month, grp in s.groupby(s.index.to_period("M"), sort=True):
        r_grp = r.loc[grp.index]
        by_month_rows.append({
            "month": str(month),
            "mean_ic": float(grp.mean(skipna=True)),
            "mean_rank_ic": float(r_grp.mean(skipna=True)),
            "n_days": int(grp.notna().sum()),
        })
    return by_month_rows


def by_symbol_ts_ic(
    factor: pd.Series,
    label: pd.Series,
    *,
    min_pairs: int = 5,
) -> list[dict[str, Any]]:
    """逐标的时序 IC 明细（instrument → ts_ic）。"""
    rows: list[dict[str, Any]] = []
    for inst, f_sub in factor.groupby(level="instrument", sort=False):
        y_sub = label.xs(inst, level="instrument")
        ts_ic = pearson_ic(
            f_sub.to_numpy(dtype=np.float64, copy=False),
            y_sub.to_numpy(dtype=np.float64, copy=False),
            min_pairs=min_pairs,
        )
        rows.append({"instrument": str(inst), "ts_ic": ts_ic})
    return rows


def monthly_ic_robustness(daily_ic: pd.Series) -> dict[str, Any]:
    """由逐日截面 IC 聚合月度稳健性（股票口径）。"""
    if daily_ic.empty:
        return {}
    s = daily_ic.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[np.isfinite(s.to_numpy(dtype=float, copy=False))]
    if s.empty:
        return {}

    monthly_means: list[float] = []
    months: list[str] = []
    for month, grp in s.groupby(s.index.to_period("M"), sort=True):
        vals = grp.to_numpy(dtype=float, copy=False)
        m = float(np.mean(vals)) if len(vals) else float("nan")
        months.append(str(month))
        monthly_means.append(m)

    n_months = len(monthly_means)
    finite_means = [x for x in monthly_means if np.isfinite(x)]
    mean_monthly = float(np.mean(finite_means)) if finite_means else float("nan")
    share_ic_pos = (
        float(sum(m > 0 for m in finite_means) / len(finite_means)) if finite_means else float("nan")
    )

    return {
        "n_months": n_months,
        "mean_monthly_ic": mean_monthly,
        "share_months_ic_positive": share_ic_pos,
        "note": "股票口径：逐日横截面 IC 按自然月取均值；share_months_ic_positive=月均 IC>0 的月份占比。",
    }


def factor_skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    """因子有限值偏度与超额峰度。"""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan")
    s = pd.Series(x)
    return float(s.skew()), float(s.kurtosis())
