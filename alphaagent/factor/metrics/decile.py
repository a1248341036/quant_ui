"""metrics 子模块：分位/分组的辅助函数。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphaagent.factor import metrics_fast as _mf
from ._core import pearson_ic, spearman_ic


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
    _fast_equal_freq_codes=None,
) -> list[float] | None:
    """单日截面上按因子等频分组的 label 均值；D1=因子最低组。"""
    xf = np.asarray(factor, dtype=float)
    yl = np.asarray(label, dtype=float)
    mask = np.isfinite(xf) & np.isfinite(yl)
    xf, yl = xf[mask], yl[mask]
    if xf.size < min_stocks or xf.size < n_deciles:
        return None

    codes = _fast_equal_freq_codes(xf, n_deciles) if _fast_equal_freq_codes else None
    if codes is None:
        # 分位边界有重复（离散值扎堆），走 pd.qcut 的 duplicates='drop' 语义
        fac_s = pd.Series(xf)
        try:
            qbins = pd.qcut(fac_s, n_deciles, duplicates="drop")
        except ValueError:
            qbins = pd.qcut(fac_s.rank(method="first"), n_deciles, duplicates="drop")
        n_q = int(qbins.cat.categories.size)
        if n_q < 2:
            return None
        codes = qbins.cat.codes.to_numpy()
        n_q = codes.max() + 1 if codes.size and codes.max() >= 0 else 0
        if n_q < 2:
            return None
    else:
        n_q = n_deciles
    means: list[float] = []
    for i in range(n_q):
        bucket = codes == i
        if not bucket.any():
            means.append(float("nan"))
        else:
            means.append(float(np.mean(yl[bucket])))
    return means


# 逐日十分组 label 均值的计算入口：返回 {ts: means} 按日字典（空 dict 表示无达标日）。
# 不设全局记忆化——因子对象地址会随 GC 复用，id 记忆化有错误命中风险。
# 同一次 evaluate 内由 mls_fmb 显式计算一次并传给 rho/ls 两个 series 函数共享。
def _compute_daily_decile_mean_labels(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    n_deciles: int = 10,
    min_stocks: int = 30,
    _day_slices=None,
    _fast_equal_freq_codes=None,
) -> dict[object, list[float]]:
    """逐日截面十分组 label 均值（{ts: means}）；样本不足日跳过。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("_iter_daily_decile_mean_labels 需要 MultiIndex 面板 (datetime, instrument)")
    if time_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r}")

    dts = factor.index.get_level_values(time_level)
    f_arr = factor.to_numpy(dtype=np.float64, copy=False)
    l_arr = label.to_numpy(dtype=np.float64, copy=False)
    n = len(f_arr)
    all_means: dict[object, list[float]] = {}
    if n == 0:
        return all_means

    # 连续运行区间（datetime 层非递减）；否则回落 groupby
    dt_np = dts._values
    sorted_ok = True
    if len(dt_np) > 1 and not (dt_np[1:] >= dt_np[:-1]).all():
        sorted_ok = False
    if not sorted_ok:
        for ts, f_sub in factor.groupby(level=time_level, sort=False):
            y_sub = label.xs(ts, level=time_level)
            means = _cross_section_decile_mean_labels(
                f_sub.to_numpy(dtype=np.float64, copy=False),
                y_sub.to_numpy(dtype=np.float64, copy=False),
                n_deciles=n_deciles,
                min_stocks=min_stocks,
                _fast_equal_freq_codes=_fast_equal_freq_codes,
            )
            if means is not None:
                all_means[ts] = means
        return all_means

    change = np.flatnonzero(dt_np[1:] != dt_np[:-1]) + 1
    bounds = np.concatenate(([0], change, [n])).astype(np.int64)

    if _mf.HAS_NUMBA:
        # numba 逐日内核；ok=0 的日子（边界重合/样本不足）回落 Python 路径
        means2d, ok = _mf.decile_label_means_days(f_arr, l_arr, bounds, n_deciles, min_stocks)
        for i in range(len(bounds) - 1):
            st, en = int(bounds[i]), int(bounds[i + 1])
            if ok[i]:
                all_means[dts[st]] = list(means2d[i])
                continue
            if en - st < min_stocks or en - st < n_deciles:
                continue
            means = _cross_section_decile_mean_labels(
                f_arr[st:en], l_arr[st:en], n_deciles=n_deciles, min_stocks=min_stocks,
                _fast_equal_freq_codes=_fast_equal_freq_codes,
            )
            if means is not None:
                all_means[dts[st]] = means
        return all_means

    for i in range(len(bounds) - 1):
        st, en = int(bounds[i]), int(bounds[i + 1])
        if en - st < min_stocks or en - st < n_deciles:
            continue
        f_day = f_arr[st:en]
        l_day = l_arr[st:en]
        mask = np.isfinite(f_day) & np.isfinite(l_day)
        if int(mask.sum()) < min_stocks or int(mask.sum()) < n_deciles:
            continue
        means = _cross_section_decile_mean_labels(
            f_day, l_day, n_deciles=n_deciles, min_stocks=min_stocks,
            _fast_equal_freq_codes=_fast_equal_freq_codes,
        )
        if means is not None:
            all_means[dts[st]] = means
    return all_means


def _iter_daily_decile_mean_labels(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    n_deciles: int = 10,
    min_stocks: int = 30,
    decile_means: dict[object, list[float]] | None = None,
    _day_slices=None,
    _fast_equal_freq_codes=None,
):
    """逐日截面十分组 label 均值；样本不足日跳过。

    ``decile_means`` 为调用方预计算的 ``{ts: means}``（mls_fmb 传给 rho/ls 共享），
    缺省时内部计算一次。
    """
    if decile_means is None:
        decile_means = _compute_daily_decile_mean_labels(
            factor, label, time_level=time_level, n_deciles=n_deciles, min_stocks=min_stocks,
            _day_slices=_day_slices, _fast_equal_freq_codes=_fast_equal_freq_codes,
        )
    for ts, means in decile_means.items():
        yield ts, means


def daily_quantile_group_returns(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    n_groups: int = 10,
    min_stocks: int = 30,
    _day_slices=None,
    _fast_equal_freq_codes=None,
) -> pd.DataFrame:
    """逐日截面 N 分组等权收益矩阵。

    返回 DataFrame: index=datetime, columns=group(1..N), values=该组当日等权 label 均值。
    group 1 = 因子值最低组, group N = 因子值最高组。
    """
    f_arr_all = factor.to_numpy(dtype=np.float64, copy=False)
    l_arr_all = label.to_numpy(dtype=np.float64, copy=False)
    rows: list[dict[str, Any]] = []
    slices = _day_slices(factor.index, time_level) if _day_slices else None
    if slices is not None:
        bounds, day_vals = slices
        day_iter = list(zip(day_vals.tolist(), bounds[:-1].tolist(), bounds[1:].tolist()))
    else:
        day_iter = None
    if day_iter is not None:
        for ts, st, en in day_iter:
            means = _cross_section_decile_mean_labels(
                f_arr_all[st:en], l_arr_all[st:en],
                n_deciles=n_groups, min_stocks=min_stocks,
                _fast_equal_freq_codes=_fast_equal_freq_codes,
            )
            if means is not None:
                row = {time_level: ts}
                for i, m in enumerate(means):
                    row[i + 1] = m
                rows.append(row)
    else:
        for ts, f_sub in factor.groupby(level=time_level, sort=False):
            y_sub = label.xs(ts, level=time_level)
            means = _cross_section_decile_mean_labels(
                f_sub.to_numpy(dtype=np.float64, copy=False),
                y_sub.to_numpy(dtype=np.float64, copy=False),
                n_deciles=n_groups,
                min_stocks=min_stocks,
                _fast_equal_freq_codes=_fast_equal_freq_codes,
            )
            if means is not None:
                row = {time_level: ts}
                for i, m in enumerate(means):
                    row[i + 1] = m
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index(time_level)
    df.columns = [int(c) for c in df.columns]
    return df
