"""metrics 子模块：MLS-FMB 与 Newey-West 统计量。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ._core import spearman_ic
from .decile import _iter_daily_decile_mean_labels


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


def daily_decile_monotonicity_series(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    n_deciles: int = 10,
    min_stocks: int = 30,
    min_deciles_for_rho: int = 3,
    decile_means: dict[object, list[float]] | None = None,
    _day_slices=None,
    _fast_equal_freq_codes=None,
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
        decile_means=decile_means,
        _day_slices=_day_slices,
        _fast_equal_freq_codes=_fast_equal_freq_codes,
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
    decile_means: dict[object, list[float]] | None = None,
    _day_slices=None,
    _fast_equal_freq_codes=None,
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
        decile_means=decile_means,
        _day_slices=_day_slices,
        _fast_equal_freq_codes=_fast_equal_freq_codes,
    ):
        top = means[-1]
        bottom = means[0]
        if np.isfinite(top) and np.isfinite(bottom):
            rows.append(float(top - bottom))
        else:
            rows.append(float("nan"))
        idx.append(ts)

    return pd.Series(rows, index=pd.Index(idx, name=time_level), dtype=float)


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
    holding_days: int = 1,
    _day_slices=None,
    _fast_equal_freq_codes=None,
) -> dict[str, Any]:
    """MLS-FMB：逐日截面计算十分组单调性 ρ_t 与 Q10−Q1 多空 LS_t，再按 Fama–MacBeth 时序聚合并用 Newey–West t 检验显著性，综合得分 MLS = mean(ρ) × 年化 IR_LS。

    ``holding_days`` 对齐 label 名义持有期（label_20d → 20）：label>1 时
    逐日 LS_t 来自重叠收益，均值无偏但 std 被低估导致 IR 虚高。按持有期
    节奏重采样去重叠，并把年化因子缩小为 ``annualization_factor / hold``，
    NW 滞后至少覆盖持有期以吸收残余自相关。label_1d 时 hold=1 完全退化
    为原行为。
    """
    hold = max(1, int(holding_days))
    eff_annual = annualization_factor / hold
    # NW 滞后至少覆盖持有期，消解重叠收益自相关（否则 std 被低估、t 虚高）
    if nw_lags is None:
        nw_lags = max(1, int(hold))
    eff_deciles, eff_min_stocks = _resolve_mls_params(
        factor,
        n_deciles=n_deciles,
        min_stocks=min_stocks,
    )
    # 十分组 label 均值只算一次，rho 与 ls 两个 series 共享（避免重复 2 次逐日 qcut）
    from .decile import _compute_daily_decile_mean_labels
    decile_means = _compute_daily_decile_mean_labels(
        factor,
        label,
        n_deciles=eff_deciles,
        min_stocks=eff_min_stocks,
        _day_slices=_day_slices,
        _fast_equal_freq_codes=_fast_equal_freq_codes,
    )
    rho_series = daily_decile_monotonicity_series(
        factor,
        label,
        n_deciles=eff_deciles,
        min_stocks=eff_min_stocks,
        decile_means=decile_means,
        _day_slices=_day_slices,
        _fast_equal_freq_codes=_fast_equal_freq_codes,
    )
    ls_series = daily_long_short_series(
        factor,
        label,
        n_deciles=eff_deciles,
        min_stocks=eff_min_stocks,
        decile_means=decile_means,
        _day_slices=_day_slices,
        _fast_equal_freq_codes=_fast_equal_freq_codes,
    )

    # 持有期 >1 时按节奏重采样，去掉重叠收益的自相关
    if hold > 1:
        rho_series = rho_series.iloc[::hold]
        ls_series = ls_series.iloc[::hold]

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
        float(ir_ls * math.sqrt(eff_annual))
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
        "annualization_factor": float(eff_annual),
        "holding_days": int(hold),
        "note": (
            "MLS+FMB 非参数版：ρ_t=逐日十分组 Spearman 单调性，LS_t=Q10-Q1 多空；"
            f"IR_LS 为日频 LS 均值/标准差，ir_ls_annual=IR_LS×√{eff_annual:g}；"
            "MLS=mean(ρ)×ir_ls_annual。"
        ),
    }
