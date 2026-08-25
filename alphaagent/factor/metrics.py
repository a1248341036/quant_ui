"""股票因子评估指标。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.factor.types import DEFAULT_LABEL_COL


def coverage(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def pearson_ic(factor: np.ndarray, label: np.ndarray, *, min_pairs: int = 30) -> float:
    x = np.asarray(factor, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < min_pairs:
        return float("nan")
    xs = x[mask]
    ys = y[mask]
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = float(np.sqrt((xs * xs).sum() * (ys * ys).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((xs * ys).sum() / denom)


def spearman_ic(factor: np.ndarray, label: np.ndarray, *, min_pairs: int = 10) -> float:
    """Spearman 秩相关（截面或任意配对样本）。"""
    x = np.asarray(factor, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < min_pairs:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_ic(xr, yr, min_pairs=min_pairs)


def cross_sectional_ic(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    min_pairs: int = 10,
) -> pd.Series:
    """按 datetime 截面 Pearson IC，返回逐日 IC 序列。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("cross_sectional_ic 需要 MultiIndex 面板 (datetime, instrument)")
    if time_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r}")

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
) -> pd.Series:
    """按 datetime 截面 Spearman Rank IC，返回逐日 RANKIC 序列。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("cross_sectional_rank_ic 需要 MultiIndex 面板 (datetime, instrument)")
    if time_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r}")

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
) -> pd.Series:
    """逐日横截面 lag-1 Pearson 自相关：corr_CS(f_t, f_{t-1})。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("cross_sectional_lag1_pearson_autocorr_series 需要 MultiIndex 面板")
    if time_level not in factor.index.names or instrument_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r} 或 {instrument_level!r}")

    lag1 = factor.groupby(level=instrument_level, sort=False).shift(1)
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
) -> float:
    """逐日横截面 lag-1 Pearson 自相关的均值。"""
    daily = cross_sectional_lag1_pearson_autocorr_series(factor, min_pairs=min_pairs)
    finite = daily[np.isfinite(daily.to_numpy(dtype=float, copy=False))]
    return float(finite.mean()) if len(finite) else float("nan")


def cs_ic_summary(
    daily_ic: pd.Series,
    daily_rank_ic: pd.Series | None = None,
) -> dict[str, float | int]:
    """由逐日截面 IC / RANKIC 汇总 IC、ICIR、RANKIC。"""
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
) -> dict[str, Any]:
    """在 panel 上计算截面 IC / ICIR / RANKIC 与 coverage。"""
    panel = panel.sort_index()
    if label_col not in panel.columns:
        raise KeyError(f"panel 缺少标签列: {label_col}")
    if len(values) != len(panel):
        raise ValueError(f"values 长度 {len(values)} != panel 行数 {len(panel)}")

    factor_series = pd.Series(values, index=panel.index, dtype=np.float32)
    label_series = panel[label_col]
    daily_ic = cross_sectional_ic(factor_series, label_series, min_pairs=min_pairs)
    daily_rank_ic = cross_sectional_rank_ic(factor_series, label_series, min_pairs=min_pairs)
    cs = cs_ic_summary(daily_ic, daily_rank_ic)
    cs_autocorr_min_pairs = min(30, max(int(panel.index.get_level_values("instrument").nunique()) - 1, min_pairs))
    mls_min_stocks = min(30, max(min_pairs * 6, 10))
    fac = factor_series.to_numpy(dtype=float, copy=False)
    lab = label_series.to_numpy(dtype=float, copy=False)

    return {
        "coverage": coverage(values),
        "ic": cs["ic"],
        "icir": cs["icir"],
        "rank_ic": cs["rank_ic"],
        "n_days": cs["n_days"],
        "cs_pearson_autocorr": cross_sectional_lag1_pearson_autocorr(
            factor_series,
            min_pairs=cs_autocorr_min_pairs,
        ),
        "decile_mean_label": decile_mean_label(fac, lab, n_deciles=10),
        "mls_fmb": mls_fmb_summary(
            factor_series,
            label_series,
            min_stocks=mls_min_stocks,
        ),
        "label_col": label_col,
    }


def evaluate_on_panel(
    values: np.ndarray,
    panel: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    min_ic_pairs: int = 5,
) -> dict[str, Any]:
    """在 panel 上计算 coverage 与截面 IC / ICIR / RANKIC。"""
    return evaluate_cs_on_panel(
        values,
        panel,
        label_col=label_col,
        min_pairs=min_ic_pairs,
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


def annualized_long_group_excess_return(
    factor: pd.Series,
    label: pd.Series,
    *,
    direction: int,
    n_deciles: int = 10,
    min_stocks: int = 30,
    annualization_factor: float = 252.0,
) -> float:
    """方向自适应的多头组相对当日全市场等权收益的复利年化超额。"""
    excesses: list[float] = []
    for _, f_sub in factor.groupby(level="datetime", sort=False):
        y_sub = label.xs(f_sub.index.get_level_values("datetime")[0], level="datetime")
        xf = f_sub.to_numpy(dtype=np.float64, copy=False)
        yl = y_sub.to_numpy(dtype=np.float64, copy=False)
        mask = np.isfinite(xf) & np.isfinite(yl)
        if int(mask.sum()) < max(min_stocks, n_deciles):
            continue
        fac = pd.Series(xf[mask])
        returns = yl[mask]
        try:
            bins = pd.qcut(fac, n_deciles, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.qcut(fac.rank(method="first"), n_deciles, labels=False, duplicates="drop")
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
    return float(np.prod(1.0 + arr) ** (annualization_factor / len(arr)) - 1.0)


def _round_label_mean(value: float) -> float:
    return float(round(value, 6))


def label_quantile_buckets(
    factor: np.ndarray,
    label: np.ndarray,
    *,
    n_quantiles: int = 10,
) -> list[dict[str, Any]]:
    """按因子值等频分位分桶，每桶内 label 均值；Q1 = 因子最低组。"""
    if n_quantiles < 2:
        return []
    xf = np.asarray(factor, dtype=float)
    yl = np.asarray(label, dtype=float)
    m = np.isfinite(xf) & np.isfinite(yl)
    xf, yl = xf[m], yl[m]
    if xf.size < n_quantiles:
        return []
    fac_s = pd.Series(xf)
    try:
        qbins = pd.qcut(fac_s, n_quantiles, duplicates="drop")
    except ValueError:
        qbins = pd.qcut(fac_s.rank(method="first"), n_quantiles, duplicates="drop")
    n_q = int(qbins.cat.categories.size)
    codes = qbins.cat.codes.to_numpy()
    out: list[dict[str, Any]] = []
    for i in range(n_q):
        mask = codes == i
        cnt = int(mask.sum())
        if cnt == 0:
            mean_v = None
        else:
            mv = float(np.mean(yl[mask]))
            mean_v = _round_label_mean(mv) if np.isfinite(mv) else None
        cat = qbins.cat.categories[i]
        out.append(
            {
                "quantile": i + 1,
                "n": cnt,
                "mean_label": mean_v,
                "factor_bin": str(cat),
            }
        )
    return out


def decile_mean_label(
    factor: np.ndarray,
    label: np.ndarray,
    *,
    n_deciles: int = 10,
) -> list[dict[str, Any]]:
    """全样本等频十分组 label 均值；D1=因子最低组，D10=最高组（同 AQRA label_quantile_buckets）。"""
    buckets = label_quantile_buckets(factor, label, n_quantiles=n_deciles)
    return [{"decile": b["quantile"], "mean_label": b["mean_label"]} for b in buckets]


def _cross_section_decile_mean_labels(
    factor: np.ndarray,
    label: np.ndarray,
    *,
    n_deciles: int = 10,
    min_stocks: int = 30,
) -> list[float] | None:
    """单日截面上按因子等频分组的 label 均值；D1=因子最低组。"""
    xf = np.asarray(factor, dtype=float)
    yl = np.asarray(label, dtype=float)
    mask = np.isfinite(xf) & np.isfinite(yl)
    xf, yl = xf[mask], yl[mask]
    if xf.size < min_stocks or xf.size < n_deciles:
        return None

    fac_s = pd.Series(xf)
    try:
        qbins = pd.qcut(fac_s, n_deciles, duplicates="drop")
    except ValueError:
        qbins = pd.qcut(fac_s.rank(method="first"), n_deciles, duplicates="drop")

    n_q = int(qbins.cat.categories.size)
    if n_q < 2:
        return None

    codes = qbins.cat.codes.to_numpy()
    means: list[float] = []
    for i in range(n_q):
        bucket = codes == i
        if not bucket.any():
            means.append(float("nan"))
        else:
            means.append(float(np.mean(yl[bucket])))
    return means


def _iter_daily_decile_mean_labels(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    n_deciles: int = 10,
    min_stocks: int = 30,
):
    """逐日截面十分组 label 均值；样本不足日跳过。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("_iter_daily_decile_mean_labels 需要 MultiIndex 面板 (datetime, instrument)")
    if time_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r}")

    for ts, f_sub in factor.groupby(level=time_level, sort=False):
        y_sub = label.xs(ts, level=time_level)
        means = _cross_section_decile_mean_labels(
            f_sub.to_numpy(dtype=np.float64, copy=False),
            y_sub.to_numpy(dtype=np.float64, copy=False),
            n_deciles=n_deciles,
            min_stocks=min_stocks,
        )
        if means is not None:
            yield ts, means


def daily_decile_monotonicity_series(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    n_deciles: int = 10,
    min_stocks: int = 30,
    min_deciles_for_rho: int = 3,
) -> pd.Series:
    """逐日截面单调性分量 ρ_t = Spearman({1..K}, {R_{1,t},..,R_{K,t}})。"""
    rows: list[float] = []
    idx: list[object] = []
    ranks_template = np.arange(1, n_deciles + 1, dtype=np.float64)

    for ts, means in _iter_daily_decile_mean_labels(
        factor,
        label,
        time_level=time_level,
        n_deciles=n_deciles,
        min_stocks=min_stocks,
    ):
        means_arr = np.asarray(means, dtype=np.float64)
        valid = np.isfinite(means_arr)
        n_valid = int(valid.sum())
        if n_valid < min_deciles_for_rho:
            rho = float("nan")
        else:
            ranks = ranks_template[: len(means_arr)][valid]
            rho = spearman_ic(ranks, means_arr[valid], min_pairs=min_deciles_for_rho)
        rows.append(rho)
        idx.append(ts)

    return pd.Series(rows, index=pd.Index(idx, name=time_level), dtype=float)


def daily_long_short_series(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    n_deciles: int = 10,
    min_stocks: int = 30,
) -> pd.Series:
    """逐日截面多空分量 LS_t = R_{Q10,t} - R_{Q1,t}（最高组减最低组）。"""
    rows: list[float] = []
    idx: list[object] = []

    for ts, means in _iter_daily_decile_mean_labels(
        factor,
        label,
        time_level=time_level,
        n_deciles=n_deciles,
        min_stocks=min_stocks,
    ):
        top = means[-1]
        bottom = means[0]
        if np.isfinite(top) and np.isfinite(bottom):
            rows.append(float(top - bottom))
        else:
            rows.append(float("nan"))
        idx.append(ts)

    return pd.Series(rows, index=pd.Index(idx, name=time_level), dtype=float)


def newey_west_mean_tstat(
    series: pd.Series | np.ndarray,
    *,
    lags: int | None = None,
) -> dict[str, float | int]:
    """Newey–West t 统计量，检验序列均值是否显著异于 0（Bartlett 核）。"""
    if isinstance(series, pd.Series):
        x = series.to_numpy(dtype=np.float64, copy=False)
    else:
        x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    t_n = int(len(x))
    nan_out: dict[str, float | int] = {
        "mean": float("nan"),
        "se_nw": float("nan"),
        "t_nw": float("nan"),
        "n": t_n,
        "lags": 0,
    }
    if t_n < 2:
        return nan_out

    if lags is None:
        lags = max(1, int(4.0 * (t_n / 100.0) ** (2.0 / 9.0)))
    lags = min(int(lags), t_n - 1)

    x_mean = float(x.mean())
    u = x - x_mean
    gamma0 = float(np.dot(u, u) / t_n)
    nw_var_mean = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma_k = float(np.dot(u[k:], u[:-k]) / t_n)
        nw_var_mean += 2.0 * w * gamma_k
    nw_var_mean /= t_n

    if nw_var_mean <= 0.0 or not np.isfinite(nw_var_mean):
        se_nw = float("nan")
        t_nw = float("nan")
    else:
        se_nw = float(math.sqrt(nw_var_mean))
        t_nw = float(x_mean / se_nw)

    return {
        "mean": x_mean,
        "se_nw": se_nw,
        "t_nw": t_nw,
        "n": t_n,
        "lags": lags,
    }


def _resolve_mls_params(
    factor: pd.Series,
    *,
    n_deciles: int,
    min_stocks: int,
    instrument_level: str = "instrument",
) -> tuple[int, int]:
    """按截面股票数自适应分位数个数与最低样本门槛。"""
    n_inst = int(factor.index.get_level_values(instrument_level).nunique())
    eff_deciles = min(int(n_deciles), n_inst)
    eff_min = min(int(min_stocks), n_inst)
    if eff_min < eff_deciles:
        eff_min = eff_deciles
    return eff_deciles, eff_min


def mls_fmb_summary(
    factor: pd.Series,
    label: pd.Series,
    *,
    n_deciles: int = 10,
    min_stocks: int = 30,
    annualization_factor: float = 252.0,
    nw_lags: int | None = None,
) -> dict[str, Any]:
    """MLS-FMB：逐日截面计算十分组单调性 ρ_t 与 Q10−Q1 多空 LS_t，再按 Fama–MacBeth 时序聚合并用 Newey–West t 检验显著性，综合得分 MLS = mean(ρ) × 年化 IR_LS。"""
    eff_deciles, eff_min_stocks = _resolve_mls_params(
        factor,
        n_deciles=n_deciles,
        min_stocks=min_stocks,
    )
    rho_series = daily_decile_monotonicity_series(
        factor,
        label,
        n_deciles=eff_deciles,
        min_stocks=eff_min_stocks,
    )
    ls_series = daily_long_short_series(
        factor,
        label,
        n_deciles=eff_deciles,
        min_stocks=eff_min_stocks,
    )

    rho_nw = newey_west_mean_tstat(rho_series, lags=nw_lags)
    ls_nw = newey_west_mean_tstat(ls_series, lags=nw_lags)

    rho_vals = rho_series[np.isfinite(rho_series.to_numpy(dtype=float, copy=False))]
    ls_vals = ls_series[np.isfinite(ls_series.to_numpy(dtype=float, copy=False))]

    mean_rho = float(rho_vals.mean()) if len(rho_vals) else float("nan")
    mean_ls = float(ls_vals.mean()) if len(ls_vals) else float("nan")

    if len(ls_vals) > 1:
        ls_std = float(ls_vals.std(ddof=1))
    elif len(ls_vals) == 1:
        ls_std = float("nan")
    else:
        ls_std = float("nan")

    ir_ls = mean_ls / ls_std if ls_std and np.isfinite(ls_std) and ls_std > 0 else float("nan")
    ir_ls_annual = (
        float(ir_ls * math.sqrt(annualization_factor))
        if np.isfinite(ir_ls)
        else float("nan")
    )
    mls = (
        float(mean_rho * ir_ls_annual)
        if np.isfinite(mean_rho) and np.isfinite(ir_ls_annual)
        else float("nan")
    )

    return {
        "mean_rho": mean_rho,
        "mean_ls": mean_ls,
        "ir_ls": ir_ls,
        "ir_ls_annual": ir_ls_annual,
        "mls": mls,
        "nw_t_rho": rho_nw["t_nw"],
        "nw_t_ls": ls_nw["t_nw"],
        "nw_se_rho": rho_nw["se_nw"],
        "nw_se_ls": ls_nw["se_nw"],
        "n_days_rho": int(rho_nw["n"]),
        "n_days_ls": int(ls_nw["n"]),
        "nw_lags": int(rho_nw["lags"]),
        "n_deciles": int(eff_deciles),
        "n_deciles_requested": int(n_deciles),
        "min_stocks": int(eff_min_stocks),
        "min_stocks_requested": int(min_stocks),
        "annualization_factor": float(annualization_factor),
        "note": (
            "MLS+FMB 非参数版：ρ_t=逐日十分组 Spearman 单调性，LS_t=Q10-Q1 多空；"
            "IR_LS 为日频 LS 均值/标准差，ir_ls_annual=IR_LS×√252；MLS=mean(ρ)×ir_ls_annual。"
        ),
    }


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

    n_months = len(months)
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


def _rebalance_dates(ts_list: list, rebalance: str) -> set:
    """按调仓周期标记调仓日：daily=每日；weekly=每周最后交易日；monthly=每月最后交易日。"""
    if rebalance == "daily":
        return set(ts_list)
    idx = pd.DatetimeIndex(ts_list)
    s = pd.Series(idx, index=idx)
    if rebalance == "weekly":
        groups = [s.groupby([idx.year, idx.isocalendar().week.astype(int)]).max()]
    elif rebalance == "monthly":
        groups = [s.groupby([idx.year, idx.month]).max()]
    else:
        raise ValueError(f"unknown rebalance freq: {rebalance!r}")
    out: set = set()
    for g in groups:
        out.update(g.tolist())
    return out


def topn_selection_overlap(
    factor: pd.Series,
    *,
    time_level: str = "datetime",
    top_n: int | None = None,
    selection_pct: float | None = None,
    rebalance: str = "daily",
) -> float:
    """TopN 选股名单的相邻调仓重合率（纯排名统计，不涉及价格/收益）。

    支持两种选股模式：
    - 固定 N：传 top_n=30 每日选 30 只
    - 动态百分比：传 selection_pct=0.02 每日选前 2%（适配停牌/涨跌停导致的候选池缩放）
    两者都传时以 selection_pct 为准。
    """
    rb_dates = _rebalance_dates(list(factor.groupby(level=time_level, sort=False).groups.keys()), rebalance)
    prev: set[str] | None = None
    overlap_sum = 0.0
    count = 0
    for ts, f_sub in factor.groupby(level=time_level, sort=False):
        if ts not in rb_dates:
            continue
        xf = f_sub.to_numpy(dtype=np.float64, copy=False)
        inst = np.asarray(f_sub.index.get_level_values("instrument"))
        valid = np.isfinite(xf)
        n_valid = int(valid.sum())
        if selection_pct is not None:
            n_pick = max(1, int(np.ceil(n_valid * float(selection_pct))))
        else:
            n_pick = int(top_n or 30)
        if n_valid < max(n_pick, 10):
            continue
        order = np.argsort(-xf[valid], kind="stable")[:n_pick]
        current = set(inst[valid][order].tolist())
        if prev is not None:
            overlap_sum += len(current & prev) / max(len(current), 1)
            count += 1
        prev = current
    return round(overlap_sum / count, 6) if count else float("nan")


def topn_portfolio_summary(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    top_n: int = 30,
    cost_bps: float = 15.0,
    annualization_factor: float = 252.0,
    rebalance: str = "daily",
) -> dict[str, Any]:
    """TopN 等权多头组合模拟（含换手成本），支持 daily/weekly/monthly 调仓。

    .. deprecated:: 已被 core.engine 完整约束回测取代（见
       alphaagent.factor.mining.engine_gate）。仅保留给离线对照实验使用，
       评估管线与入库门禁不再调用。

    调仓日按当日因子取前 ``top_n`` 只并持有到下一调仓日，收益用次日
    open-to-open label；非调仓日沿用既有持仓（停牌股自然缺席当日均值）。
    成本仅在调仓日按实际换手比例计双边 ``cost_bps``。未建模 T+1 涨跌停
    锁定与冲击成本，最终仍需旧交易引擎复核。
    """
    per_day: list[tuple[object, np.ndarray, np.ndarray]] = []
    for ts, f_sub in factor.groupby(level=time_level, sort=False):
        inst = f_sub.index.get_level_values("instrument")
        y_sub = label.xs(ts, level=time_level).reindex(inst)
        xf = f_sub.to_numpy(dtype=np.float64, copy=False)
        yl = y_sub.to_numpy(dtype=np.float64, copy=False)
        mask = np.isfinite(xf) & np.isfinite(yl)
        n_valid = int(mask.sum())
        if n_valid < max(top_n, 10):
            continue
        per_day.append((ts, np.asarray(inst)[mask], xf[mask], yl[mask]))
    if not per_day:
        return {
            "top_n": int(top_n),
            "rebalance": str(rebalance),
            "n_days": 0,
            "annualized_return": float("nan"),
            "annualized_excess_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "annual_turnover": float("nan"),
            "daily_overlap": float("nan"),
            "avg_names": float(top_n),
            "cost_bps_per_side": float(cost_bps),
            "note": "TopN 等权多头模拟；成本=调仓日换手×双边cost_bps；未含T+1涨跌停锁定/冲击成本。",
        }

    rb_dates = _rebalance_dates([row[0] for row in per_day], rebalance)
    held_names: set[str] = set()
    prev_held: set[str] | None = None
    turnover_sum = 0.0
    rebalance_count = 0
    overlap_sum = 0.0
    overlap_count = 0
    net: list[float] = []
    excess: list[float] = []

    for ts, names_arr, fac, ret in per_day:
        universe_ret = float(ret.mean())
        is_rebalance_day = ts in rb_dates
        day_cost = 0.0
        if is_rebalance_day:
            order = np.argsort(-fac, kind="stable")[:top_n]
            current = set(names_arr[order].tolist())
            if prev_held is not None:
                changed = len(current - prev_held) / max(len(current), 1)
                overlap_sum += len(current & prev_held) / max(len(current), 1)
                overlap_count += 1
                day_cost = 2.0 * (cost_bps / 10_000.0) * changed
            else:
                changed = 1.0  # 建仓：单边买入
                day_cost = (cost_bps / 10_000.0) * changed
            turnover_sum += changed
            rebalance_count += 1
            prev_held = current
            held_names = current
        name_to_ret = dict(zip(names_arr.tolist(), ret.tolist(), strict=False))
        held_returns = [name_to_ret[n] for n in held_names if n in name_to_ret]
        if not held_returns:
            continue
        port_ret = float(np.mean(held_returns))
        net.append(port_ret - day_cost)
        excess.append((port_ret - universe_ret) - day_cost)

    def _compound_ann(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if np.any(arr <= -1.0):
            return float("nan")
        return float(np.prod(1.0 + arr) ** (annualization_factor / len(arr)) - 1.0)

    arr = np.asarray(net, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    running_max = np.maximum.accumulate(np.cumprod(1.0 + arr))
    drawdowns = 1.0 - np.cumprod(1.0 + arr) / running_max
    years = max(len(per_day) / annualization_factor, 1e-9)

    return {
        "top_n": int(top_n),
        "rebalance": str(rebalance),
        "n_days": int(len(per_day)),
        "annualized_return": _compound_ann(net),
        "annualized_excess_return": _compound_ann(excess),
        "sharpe": float(arr.mean() / std * math.sqrt(annualization_factor)) if std > 0 and np.isfinite(std) else float("nan"),
        "max_drawdown": float(drawdowns.max()) if len(drawdowns) else float("nan"),
        "annual_turnover": round(turnover_sum / years, 4) if rebalance_count else float("nan"),
        "daily_overlap": round(overlap_sum / overlap_count, 6) if overlap_count else float("nan"),
        "avg_names": float(top_n),
        "cost_bps_per_side": float(cost_bps),
        "note": "TopN 等权多头模拟；成本=调仓日换手×双边cost_bps；未含T+1涨跌停锁定/冲击成本。",
    }
