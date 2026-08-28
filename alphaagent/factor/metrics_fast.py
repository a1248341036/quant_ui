"""逐日截面统计的 Numba 内核（``nogil``：多评估并发时真正多核）。

覆盖评估 metrics / transforms 里的逐日 Python 循环热点：
- 逐日 Pearson IC / Spearman Rank IC（含逐日平均秩）
- 逐日十分组 label 均值（与 ``_fast_equal_freq_codes`` + ``pd.qcut`` 语义一致，
  边界重合 / 重复边界的日子标记 ``ok=0`` 由调用方回落 Python 路径）
- 逐日市值对数回归残差（size_neutralize）
- 逐日截面 zscore / winsorize（评估 transforms）

所有内核假定数组按 datetime 层非递减排序、``bounds`` 为 ``[起0, ..., 末]`` 的
逐日连续区间（见 ``metrics._day_slices``）。numba 缺失时调用方走原有 Python 路径。
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco if not args else args[0]


@njit(cache=True, nogil=True)
def _finite_pair_sums(x, y, st, en):
    """[st,en) 内有限配对的两遍统计：返回 (cnt, sxx, syy, sxy)。"""
    cnt = 0
    sx = 0.0
    sy = 0.0
    for i in range(st, en):
        a = x[i]
        b = y[i]
        if a == a and b == b:
            cnt += 1
            sx += a
            sy += b
    if cnt == 0:
        return 0, 0.0, 0.0, 0.0
    mx = sx / cnt
    my = sy / cnt
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for i in range(st, en):
        a = x[i]
        b = y[i]
        if a == a and b == b:
            da = a - mx
            db = b - my
            sxx += da * da
            syy += db * db
            sxy += da * db
    return cnt, sxx, syy, sxy


@njit(cache=True, nogil=True)
def _pearson_ic_days_impl(
    f: np.ndarray, y: np.ndarray, bounds: np.ndarray, min_pairs: int
) -> np.ndarray:
    nd = bounds.shape[0] - 1
    out = np.empty(nd, dtype=np.float64)
    for d in range(nd):
        st = bounds[d]
        en = bounds[d + 1]
        cnt, sxx, syy, sxy = _finite_pair_sums(f, y, st, en)
        if cnt < min_pairs:
            out[d] = np.nan
            continue
        denom = np.sqrt(sxx * syy)
        if denom <= 0.0 or not (denom == denom):
            out[d] = np.nan
        else:
            out[d] = sxy / denom
    return out


@njit(cache=True, nogil=True)
def _average_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """升序值的平均秩（1 起，并列取平均），与 pandas ``rank(method='average')`` 一致。"""
    m = sorted_vals.shape[0]
    ranks = np.empty(m, dtype=np.float64)
    i = 0
    while i < m:
        j = i
        while j + 1 < m and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    return ranks


@njit(cache=True, nogil=True)
def _spearman_ic_days_impl(
    f: np.ndarray, y: np.ndarray, bounds: np.ndarray, min_pairs: int
) -> np.ndarray:
    nd = bounds.shape[0] - 1
    out = np.empty(nd, dtype=np.float64)
    for d in range(nd):
        st = bounds[d]
        en = bounds[d + 1]
        cnt = 0
        for i in range(st, en):
            if f[i] == f[i] and y[i] == y[i]:
                cnt += 1
        if cnt < min_pairs:
            out[d] = np.nan
            continue
        xs = np.empty(cnt, dtype=np.float64)
        ys = np.empty(cnt, dtype=np.float64)
        j = 0
        for i in range(st, en):
            a = f[i]
            b = y[i]
            if a == a and b == b:
                xs[j] = a
                ys[j] = b
                j += 1
        # 平均秩要映射回原位置：对排序副本求秩，再按 argsort 散射回去
        xr = np.empty(cnt, dtype=np.float64)
        yr = np.empty(cnt, dtype=np.float64)
        order_x = np.argsort(xs)
        rank_sorted = _average_ranks(xs[order_x])
        for k in range(cnt):
            xr[order_x[k]] = rank_sorted[k]
        order_y = np.argsort(ys)
        rank_sorted = _average_ranks(ys[order_y])
        for k in range(cnt):
            yr[order_y[k]] = rank_sorted[k]
        # 秩均值 (cnt+1)/2，必须中心化后才能用内积公式（与 pearson_ic 一致）
        mean_r = 0.5 * (cnt + 1)
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for k in range(cnt):
            dx = xr[k] - mean_r
            dy = yr[k] - mean_r
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy
        denom = np.sqrt(sxx * syy)
        if denom <= 0.0:
            out[d] = np.nan
        else:
            out[d] = sxy / denom
    return out


@njit(cache=True, nogil=True)
def _decile_label_means_days_impl(
    f: np.ndarray,
    y: np.ndarray,
    bounds: np.ndarray,
    n_deciles: int,
    min_stocks: int,
    means_out: np.ndarray,
    ok_out: np.ndarray,
) -> None:
    """逐日十分组 label 均值。

    ``ok_out[d]=1``：means_out[d, :] 有效（快速等频分箱，无边界重合/重复边界）；
    ``ok_out[d]=0``：该日需调用方回落 Python 路径（``pd.qcut`` / 样本不足）。
    """
    nd = bounds.shape[0] - 1
    q = np.linspace(0.0, 1.0, n_deciles + 1)
    for d in range(nd):
        st = bounds[d]
        en = bounds[d + 1]
        cnt = 0
        for i in range(st, en):
            if f[i] == f[i] and y[i] == y[i]:
                cnt += 1
        if cnt < min_stocks or cnt < n_deciles:
            ok_out[d] = 0
            continue
        xs = np.empty(cnt, dtype=np.float64)
        ys = np.empty(cnt, dtype=np.float64)
        j = 0
        for i in range(st, en):
            a = f[i]
            b = y[i]
            if a == a and b == b:
                xs[j] = a
                ys[j] = b
                j += 1
        xs_sorted = np.sort(xs)

        # 线性插值分位边界（与 np.quantile 默认 linear 一致）
        edges = np.empty(n_deciles + 1, dtype=np.float64)
        m1 = cnt - 1
        for k in range(n_deciles + 1):
            pos = q[k] * m1
            lo = int(np.floor(pos))
            hi = int(np.ceil(pos))
            frac = pos - lo
            if frac == 0.0:
                edges[k] = xs_sorted[lo]
            else:
                edges[k] = xs_sorted[lo] * (1.0 - frac) + xs_sorted[hi] * frac

        # 重复边界 → 回落（qcut duplicates='drop' 语义）
        interior = np.empty(n_deciles - 1, dtype=np.float64)
        for k in range(1, n_deciles):
            interior[k - 1] = edges[k]
            if not (edges[k] > edges[k - 1]):
                ok_out[d] = 0
                break
        if ok_out[d] == 0:
            continue
        if not (edges[n_deciles] > edges[n_deciles - 1]):
            ok_out[d] = 0
            continue

        # 有值恰好压在内部边界上（浮点重合，pandas 边界可能有 ulp 差）→ 回落
        p = 0
        coincide = False
        for j2 in range(cnt):
            while p < n_deciles - 1 and interior[p] < xs_sorted[j2]:
                p += 1
            if p < n_deciles - 1 and interior[p] == xs_sorted[j2]:
                coincide = True
                break
        if coincide:
            ok_out[d] = 0
            continue

        # 右闭分箱 (b_i, b_{i+1}]：bin = 严格小于 v 的内部边界数；顺带累加组内 label 和
        bin_sum = np.zeros(n_deciles, dtype=np.float64)
        bin_cnt = np.zeros(n_deciles, dtype=np.int64)
        p = 0
        for j2 in range(cnt):
            while p < n_deciles - 1 and interior[p] < xs_sorted[j2]:
                p += 1
            bin_sum[p] += ys[j2]
            bin_cnt[p] += 1
        for k in range(n_deciles):
            if bin_cnt[k] > 0:
                means_out[d, k] = bin_sum[k] / bin_cnt[k]
            else:
                means_out[d, k] = np.nan
        ok_out[d] = 1


@njit(cache=True, nogil=True)
def _size_resid_days_impl(
    values: np.ndarray,
    size: np.ndarray,
    bounds: np.ndarray,
    min_valid: int,
) -> None:
    """逐日对 size 截面线性回归取残差（values 原地修改）。

    与 ``np.cov(x, y, ddof=0)/np.var(x)`` 斜率口径一致。
    """
    nd = bounds.shape[0] - 1
    for d in range(nd):
        st = bounds[d]
        en = bounds[d + 1]
        cnt = 0
        sx = 0.0
        sy = 0.0
        for i in range(st, en):
            a = size[i]
            b = values[i]
            if a == a and b == b:
                cnt += 1
                sx += a
                sy += b
        if cnt < min_valid:
            continue
        mx = sx / cnt
        my = sy / cnt
        sxx = 0.0
        sxy = 0.0
        for i in range(st, en):
            a = size[i]
            b = values[i]
            if a == a and b == b:
                da = a - mx
                sxx += da * da
                sxy += da * (b - my)
        variance = sxx / cnt
        if not (variance == variance) or variance <= 1e-15:
            continue
        beta = (sxy / cnt) / variance
        alpha = my - beta * mx
        for i in range(st, en):
            a = size[i]
            b = values[i]
            if a == a and b == b:
                values[i] = b - (alpha + beta * a)


@njit(cache=True, nogil=True)
def _zscore_days_impl(out: np.ndarray, bounds: np.ndarray) -> None:
    nd = bounds.shape[0] - 1
    for d in range(nd):
        st = bounds[d]
        en = bounds[d + 1]
        cnt = 0
        s = 0.0
        for i in range(st, en):
            v = out[i]
            if v == v:
                cnt += 1
                s += v
        if cnt < 2:
            continue
        mean = s / cnt
        var = 0.0
        for i in range(st, en):
            v = out[i]
            if v == v:
                dv = v - mean
                var += dv * dv
        std = np.sqrt(var / cnt)
        if std > 0.0 and std == std:
            for i in range(st, en):
                v = out[i]
                if v == v:
                    out[i] = (v - mean) / std


@njit(cache=True, nogil=True)
def _winsorize_days_impl(
    out: np.ndarray, bounds: np.ndarray, lower_pct: float, upper_pct: float
) -> None:
    nd = bounds.shape[0] - 1
    for d in range(nd):
        st = bounds[d]
        en = bounds[d + 1]
        cnt = 0
        for i in range(st, en):
            if out[i] == out[i]:
                cnt += 1
        if cnt < 2:
            continue
        vals = np.empty(cnt, dtype=np.float64)
        j = 0
        for i in range(st, en):
            v = out[i]
            if v == v:
                vals[j] = v
                j += 1
        vals_sorted = np.sort(vals)
        lo_v = _interp_percentile(vals_sorted, lower_pct)
        hi_v = _interp_percentile(vals_sorted, upper_pct)
        for i in range(st, en):
            v = out[i]
            if v == v:
                if v < lo_v:
                    out[i] = lo_v
                elif v > hi_v:
                    out[i] = hi_v


@njit(cache=True, nogil=True)
def _interp_percentile(sorted_vals: np.ndarray, pct: float) -> float:
    """线性插值百分位（与 ``np.percentile`` 默认 linear 一致），pct ∈ [0,100]。"""
    m = sorted_vals.shape[0]
    if m == 0:
        return np.nan
    if m == 1:
        return sorted_vals[0]
    pos = (pct / 100.0) * (m - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo < 0:
        lo = 0
    if hi > m - 1:
        hi = m - 1
    frac = pos - lo
    if frac == 0.0:
        return sorted_vals[lo]
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# -----------------------------------------------------------------------------
# 公开包装（输入均为已抽取的 float64 一维数组）
# -----------------------------------------------------------------------------


def pearson_ic_days(f: np.ndarray, y: np.ndarray, bounds: np.ndarray, min_pairs: int) -> np.ndarray:
    """逐日截面 Pearson IC（长度 = 天数）。"""
    return _pearson_ic_days_impl(
        np.asarray(f, dtype=np.float64), np.asarray(y, dtype=np.float64),
        np.asarray(bounds, dtype=np.int64), int(min_pairs),
    )


def rank_ic_days(f: np.ndarray, y: np.ndarray, bounds: np.ndarray, min_pairs: int) -> np.ndarray:
    """逐日截面 Spearman Rank IC（逐日平均秩 + Pearson）。"""
    return _spearman_ic_days_impl(
        np.asarray(f, dtype=np.float64), np.asarray(y, dtype=np.float64),
        np.asarray(bounds, dtype=np.int64), int(min_pairs),
    )


def decile_label_means_days(
    f: np.ndarray, y: np.ndarray, bounds: np.ndarray, n_deciles: int, min_stocks: int
) -> tuple[np.ndarray, np.ndarray]:
    """逐日十分组 label 均值 → ``(means[n_days, K], ok[n_days])``。

    ``ok=0`` 的日子由调用方回落 Python 路径（``pd.qcut`` / 样本不足）。
    """
    nd = int(np.asarray(bounds).size - 1)
    means = np.empty((nd, int(n_deciles)), dtype=np.float64)
    ok = np.zeros(nd, dtype=np.uint8)
    _decile_label_means_days_impl(
        np.asarray(f, dtype=np.float64), np.asarray(y, dtype=np.float64),
        np.asarray(bounds, dtype=np.int64), int(n_deciles), int(min_stocks),
        means, ok,
    )
    return means, ok


def size_resid_days(values: np.ndarray, size: np.ndarray, bounds: np.ndarray, min_valid: int) -> None:
    """逐日市值对数回归残差（values 原地修改）。"""
    _size_resid_days_impl(
        values, np.asarray(size, dtype=np.float64),
        np.asarray(bounds, dtype=np.int64), int(min_valid),
    )


def zscore_days(out: np.ndarray, bounds: np.ndarray) -> None:
    """逐日截面 zscore（out 原地修改）。"""
    _zscore_days_impl(out, np.asarray(bounds, dtype=np.int64))


def winsorize_days(out: np.ndarray, bounds: np.ndarray, lower_pct: float, upper_pct: float) -> None:
    """逐日截面分位裁剪（out 原地修改）。"""
    _winsorize_days_impl(out, np.asarray(bounds, dtype=np.int64), float(lower_pct), float(upper_pct))
