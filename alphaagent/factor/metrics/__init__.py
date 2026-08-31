"""股票因子评估指标 — 向后兼容 re-export 层。

原 alphaagent/factor/metrics.py 已拆分为:
- _core: coverage, pearson_ic, spearman_ic
- decile: label_quantile_buckets, decile_mean_label, _cross_section_decile_mean_labels, ...
- ic: cross_sectional_ic, evaluate_cs_on_panel, cross_sectional_winsorize_values, ...
- mls: mls_fmb_summary, newey_west_mean_tstat, daily_decile_monotonicity_series, ...
- portfolio: quantile_portfolio_metrics, topn_selection_overlap, ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── override 变量（测试注入点）────────────────────────────────────────
# 非 None 时替换/禁用快路径（设为返回 None 的函数即强制回落旧路径）
_day_slices_override = None
_fast_equal_freq_codes_override = None


def _day_slices(
    index: pd.Index, time_level: str = "datetime"
) -> tuple[np.ndarray, np.ndarray] | None:
    if _day_slices_override is not None:
        return _day_slices_override(index, time_level)
    """datetime 层逐日连续区间 ``(bounds, day_values)``；面板未按 datetime 排序时返回 None。

    ``bounds`` 形如 ``[起0, 起1, ..., 末]``（长度 = 天数 + 1）。panel 按
    (datetime, instrument) 排序时逐日切片是零拷贝连续视图，供各 metric 把
    ``groupby + xs``（每天 O(n log n) 全表查找 × 数千天）替换为 O(1) 切片。
    """
    if not isinstance(index, pd.MultiIndex) or time_level not in index.names:
        return None
    dts = index.get_level_values(time_level)
    n = len(dts)
    if n == 0:
        return None
    dt_np = dts._values
    if len(dt_np) > 1 and not (dt_np[1:] >= dt_np[:-1]).all():
        return None
    change = np.flatnonzero(dt_np[1:] != dt_np[:-1]) + 1
    bounds = np.concatenate(([0], change, [n])).astype(np.int64)
    return bounds, dt_np[bounds[:-1]]


def _fast_equal_freq_codes(xf: np.ndarray, n_groups: int) -> np.ndarray | None:
    if _fast_equal_freq_codes_override is not None:
        return _fast_equal_freq_codes_override(xf, n_groups)
    """等频分箱 codes（与 ``pd.qcut(x, K, labels=False, duplicates='drop')`` 的
    右闭分箱一致）；分位边界无重复时走 O(n log n) 快路径，有重复（离散值扎堆，
    需要丢弃边界）时返回 None 由调用方回落 ``pd.qcut``。"""
    if xf.size < n_groups:
        return None
    q = np.linspace(0.0, 1.0, n_groups + 1)
    edges = np.quantile(xf, q)
    uniq = np.unique(edges)
    if uniq.size != n_groups + 1:
        return None
    interior = uniq[1:-1]
    # 右闭分箱 (b_i, b_{i+1}]（首 bin 含最小值）：bin = 严格小于 v 的内部边界数。
    # 若有样本恰好落在边界上（浮点重合），pandas 的分位边界可能与 np.quantile
    # 差最后一位 ulp 导致归属不可靠，此时返回 None 交回 pd.qcut 处理。
    lower = np.searchsorted(interior, xf, side="left")
    upper = np.searchsorted(interior, xf, side="right")
    if not np.array_equal(lower, upper):
        return None
    return lower


# ── 子模块的函数都通过包装器 re-export，并注入 _day_slices / _fast_equal_freq_codes ──
# 这样测试 patch M._day_slices_override / M._fast_equal_freq_codes_override 后，
# 所有函数走包装器读取的是本模块（__init__）的 override 变量。

from ._core import coverage, pearson_ic, spearman_ic  # noqa: E402
from .decile import (  # noqa: E402
    label_quantile_buckets,
    decile_mean_label,
    _cross_section_decile_mean_labels,
    _cross_section_decile_mean_labels as _cs_decile_raw,
    _compute_daily_decile_mean_labels as _compute_dml_raw,
    _iter_daily_decile_mean_labels as _iter_dml_raw,
    daily_quantile_group_returns as _dqgr_raw,
    _round_label_mean,
)
from .ic import (  # noqa: E402
    cross_sectional_ic as _csi_raw,
    cross_sectional_rank_ic as _csri_raw,
    cross_sectional_lag1_pearson_autocorr_series as _cslpas_raw,
    cross_sectional_lag1_pearson_autocorr as _cslpa_raw,
    cs_ic_summary,
    evaluate_cs_on_panel as _ecsp_raw,
    evaluate_on_panel as _eop_raw,
    cross_sectional_winsorize_values,
    cross_sectional_size_neutralize_values as _cssnv_raw,
    annualized_long_group_excess_return,
    monthly_detail_rows,
    by_symbol_ts_ic,
    monthly_ic_robustness,
    factor_skew_kurtosis,
)
from .mls import (  # noqa: E402
    newey_west_mean_tstat,
    daily_decile_monotonicity_series as _ddms_raw,
    daily_long_short_series as _dlss_raw,
    _resolve_mls_params,
    mls_fmb_summary as _mfs_raw,
)
from .portfolio import (  # noqa: E402
    _rebalance_dates,
    topn_selection_overlap,
    quantile_portfolio_metrics as _qpm_raw,
)


# ── 包装器：注入 _day_slices / _fast_equal_freq_codes ──────────────────
import functools  # noqa: E402

_DSLICE = lambda: _day_slices  # 本模块的函数，动态读取 override
_FCODE = lambda: _fast_equal_freq_codes


def cross_sectional_ic(factor, label, *, time_level="datetime", min_pairs=10):
    return _csi_raw(factor, label, time_level=time_level, min_pairs=min_pairs,
                    _day_slices=_DSLICE())


def cross_sectional_rank_ic(factor, label, *, time_level="datetime", min_pairs=10):
    return _csri_raw(factor, label, time_level=time_level, min_pairs=min_pairs,
                     _day_slices=_DSLICE())


def cross_sectional_lag1_pearson_autocorr_series(factor, *, time_level="datetime",
                                                  instrument_level="instrument",
                                                  min_pairs=30):
    return _cslpas_raw(factor, time_level=time_level, instrument_level=instrument_level,
                       min_pairs=min_pairs, _day_slices=_DSLICE())


def cross_sectional_lag1_pearson_autocorr(factor, *, min_pairs=30):
    return _cslpa_raw(factor, min_pairs=min_pairs, _day_slices=_DSLICE())


def evaluate_cs_on_panel(values, panel, *, label_col=None, min_pairs=5):
    kw = {"_day_slices": _DSLICE(), "_fast_equal_freq_codes": _FCODE()}
    if label_col is not None:
        kw["label_col"] = label_col
    return _ecsp_raw(values, panel, min_pairs=min_pairs, **kw)


def evaluate_on_panel(values, panel, *, label_col=None, min_ic_pairs=5):
    kw = {"_day_slices": _DSLICE(), "_fast_equal_freq_codes": _FCODE()}
    if label_col is not None:
        kw["label_col"] = label_col
    return _eop_raw(values, panel, min_ic_pairs=min_ic_pairs, **kw)


def cross_sectional_size_neutralize_values(values, panel, *, market_cap_field="float_cap",
                                           log_scale=True, min_valid=3):
    return _cssnv_raw(values, panel, market_cap_field=market_cap_field,
                      log_scale=log_scale, min_valid=min_valid, _day_slices=_DSLICE())


def _compute_daily_decile_mean_labels(factor, label, *, time_level="datetime",
                                       n_deciles=10, min_stocks=30):
    return _compute_dml_raw(factor, label, time_level=time_level, n_deciles=n_deciles,
                            min_stocks=min_stocks, _day_slices=_DSLICE(),
                            _fast_equal_freq_codes=_FCODE())


def _iter_daily_decile_mean_labels(factor, label, *, time_level="datetime",
                                    n_deciles=10, min_stocks=30, decile_means=None):
    return _iter_dml_raw(factor, label, time_level=time_level, n_deciles=n_deciles,
                        min_stocks=min_stocks, decile_means=decile_means,
                        _day_slices=_DSLICE(), _fast_equal_freq_codes=_FCODE())


def daily_quantile_group_returns(factor, label, *, time_level="datetime",
                                  n_groups=10, min_stocks=30):
    return _dqgr_raw(factor, label, time_level=time_level, n_groups=n_groups,
                    min_stocks=min_stocks, _day_slices=_DSLICE(),
                    _fast_equal_freq_codes=_FCODE())


def daily_decile_monotonicity_series(factor, label, *, time_level="datetime",
                                     n_deciles=10, min_stocks=30,
                                     min_deciles_for_rho=3, decile_means=None):
    return _ddms_raw(factor, label, time_level=time_level, n_deciles=n_deciles,
                    min_stocks=min_stocks, min_deciles_for_rho=min_deciles_for_rho,
                    decile_means=decile_means, _day_slices=_DSLICE(),
                    _fast_equal_freq_codes=_FCODE())


def daily_long_short_series(factor, label, *, time_level="datetime",
                            n_deciles=10, min_stocks=30, decile_means=None):
    return _dlss_raw(factor, label, time_level=time_level, n_deciles=n_deciles,
                    min_stocks=min_stocks, decile_means=decile_means,
                    _day_slices=_DSLICE(), _fast_equal_freq_codes=_FCODE())


def mls_fmb_summary(factor, label, *, n_deciles=10, min_stocks=30,
                    annualization_factor=252.0, nw_lags=None):
    return _mfs_raw(factor, label, n_deciles=n_deciles, min_stocks=min_stocks,
                   annualization_factor=annualization_factor, nw_lags=nw_lags,
                   _day_slices=_DSLICE(), _fast_equal_freq_codes=_FCODE())


def quantile_portfolio_metrics(factor, label, *, time_level="datetime",
                               n_groups=10, min_stocks=30, cost_bps=15.0,
                               annualization_factor=252.0, direction=None):
    return _qpm_raw(factor, label, time_level=time_level, n_groups=n_groups,
                   min_stocks=min_stocks, cost_bps=cost_bps,
                   annualization_factor=annualization_factor, direction=direction,
                   _day_slices=_DSLICE(), _fast_equal_freq_codes=_FCODE())


__all__ = [
    # _core
    "coverage",
    "pearson_ic",
    "spearman_ic",
    # override 注入点
    "_day_slices",
    "_fast_equal_freq_codes",
    "_day_slices_override",
    "_fast_equal_freq_codes_override",
    # ic
    "cross_sectional_ic",
    "cross_sectional_rank_ic",
    "cross_sectional_lag1_pearson_autocorr_series",
    "cross_sectional_lag1_pearson_autocorr",
    "cs_ic_summary",
    "evaluate_cs_on_panel",
    "evaluate_on_panel",
    "cross_sectional_winsorize_values",
    "cross_sectional_size_neutralize_values",
    "annualized_long_group_excess_return",
    "monthly_detail_rows",
    "by_symbol_ts_ic",
    "monthly_ic_robustness",
    "factor_skew_kurtosis",
    # decile
    "label_quantile_buckets",
    "decile_mean_label",
    "_cross_section_decile_mean_labels",
    "_compute_daily_decile_mean_labels",
    "_iter_daily_decile_mean_labels",
    "daily_quantile_group_returns",
    "_round_label_mean",
    # mls
    "newey_west_mean_tstat",
    "daily_decile_monotonicity_series",
    "daily_long_short_series",
    "_resolve_mls_params",
    "mls_fmb_summary",
    # portfolio
    "_rebalance_dates",
    "topn_selection_overlap",
    "quantile_portfolio_metrics",
]
