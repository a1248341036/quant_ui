"""底层 njit 辅助函数：bin 索引、直方图构建、metric 计算、peak 分析、sigma、DIP。"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@njit(cache=True)
def _chip_pmin_pmax(lo_arr: np.ndarray, hi_arr: np.ndarray, lo: int, hi: int, eps: float):
    pmin = np.inf
    pmax = -np.inf
    for j in range(lo, hi + 1):
        lj = lo_arr[j]
        hj = hi_arr[j]
        if lj == lj and hj == hj and hj >= lj:
            if lj < pmin:
                pmin = lj
            if hj > pmax:
                pmax = hj
    ok = (pmax - pmin) > eps
    return pmin, pmax, ok


@njit(cache=True)
def _chip_bin_index(price: float, pmin: float, bin_w: float, nbins: int) -> int:
    b = int((price - pmin) / bin_w)
    if b < 0:
        b = 0
    if b >= nbins:
        b = nbins - 1
    return b


@njit(cache=True)
def _chip_add_uniform_range(
    q: np.ndarray,
    weight: float,
    low_p: float,
    high_p: float,
    pmin: float,
    bin_w: float,
    nbins: int,
) -> None:
    if weight <= 0.0 or not (low_p == low_p) or not (high_p == high_p):
        return
    if high_p < low_p:
        return
    b_lo = _chip_bin_index(low_p, pmin, bin_w, nbins)
    b_hi = _chip_bin_index(high_p, pmin, bin_w, nbins)
    n_cov = b_hi - b_lo + 1
    if n_cov < 1:
        return
    per = weight / float(n_cov)
    for b in range(b_lo, b_hi + 1):
        q[b] += per


@njit(cache=True)
def _chip_add_triangular_range(
    q: np.ndarray,
    weight: float,
    low_p: float,
    high_p: float,
    peak_p: float,
    pmin: float,
    bin_w: float,
    nbins: int,
    eps: float,
) -> None:
    if weight <= 0.0 or not (low_p == low_p) or not (high_p == high_p) or not (peak_p == peak_p):
        return
    if high_p <= low_p + eps:
        b = _chip_bin_index(peak_p, pmin, bin_w, nbins)
        q[b] += weight
        return
    span = high_p - low_p
    wsum = 0.0
    ws = np.empty(nbins, dtype=np.float64)
    for k in range(nbins):
        ws[k] = 0.0
    for k in range(nbins):
        center = pmin + (k + 0.5) * bin_w
        if center < low_p or center > high_p:
            continue
        if center <= peak_p:
            denom = peak_p - low_p
            if denom <= eps:
                wk = 1.0
            else:
                wk = (center - low_p) / denom
        else:
            denom = high_p - peak_p
            if denom <= eps:
                wk = 1.0
            else:
                wk = (high_p - center) / denom
        ws[k] = wk
        wsum += wk
    if wsum <= eps:
        return
    inv = weight / wsum
    for k in range(nbins):
        q[k] += ws[k] * inv


@njit(cache=True)
def _chip_turnover_rate(close_p: float, volume: float, cap: float, eps: float) -> float:
    if not (close_p == close_p) or not (volume == volume) or not (cap == cap):
        return 0.0
    if cap <= eps or volume <= 0.0:
        return 0.0
    tr = (close_p * volume) / cap
    if tr > 0.999:
        tr = 0.999
    if tr < 0.0:
        tr = 0.0
    return tr


@njit(cache=True)
def _chip_build_hist_window(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    lo: int,
    hi: int,
    nbins: int,
    method_id: int,
    q: np.ndarray,
    eps: float,
) -> tuple:
    """构建窗口 [lo,hi] 归一化筹码直方图 q。返回 (pmin, bin_w, ok)。"""
    pmin, pmax, ok = _chip_pmin_pmax(low, high, lo, hi, eps)
    if not ok:
        return pmin, 0.0, False
    bin_w = (pmax - pmin) / float(nbins)
    for k in range(nbins):
        q[k] = 0.0

    if method_id == 1:
        for j in range(lo, hi + 1):
            lj = low[j]
            hj = high[j]
            if not (lj == lj and hj == hj and hj >= lj):
                continue
            tr = _chip_turnover_rate(close[j], volume[j], aux[j], eps)
            if tr <= 0.0:
                continue
            for k in range(nbins):
                q[k] *= (1.0 - tr)
            _chip_add_uniform_range(q, tr, lj, hj, pmin, bin_w, nbins)
    elif method_id == 2:
        for j in range(lo, hi + 1):
            lj = low[j]
            hj = high[j]
            vj = volume[j]
            pj = aux[j]
            if not (lj == lj and hj == hj and vj == vj and pj == pj and vj > 0.0):
                continue
            if hj < lj:
                continue
            peak = pj
            if peak < lj:
                peak = lj
            if peak > hj:
                peak = hj
            _chip_add_triangular_range(q, vj, lj, hj, peak, pmin, bin_w, nbins, eps)
    else:
        for j in range(lo, hi + 1):
            lj = low[j]
            hj = high[j]
            vj = volume[j]
            if not (lj == lj and hj == hj and vj == vj and vj > 0.0):
                continue
            if hj < lj:
                continue
            _chip_add_uniform_range(q, vj, lj, hj, pmin, bin_w, nbins)

    total = 0.0
    for k in range(nbins):
        total += q[k]
    if total <= eps:
        return pmin, bin_w, False
    inv = 1.0 / total
    for k in range(nbins):
        q[k] *= inv
    return pmin, bin_w, True


@njit(cache=True)
def _chip_metric_from_q(
    q: np.ndarray,
    pmin: float,
    bin_w: float,
    nbins: int,
    P: float,
    op: int,
    eps: float,
) -> float:
    if op == 0:
        istar = 0
        qmax = q[0]
        for k in range(1, nbins):
            if q[k] > qmax:
                qmax = q[k]
                istar = k
        peak_price = pmin + (istar + 0.5) * bin_w
        if abs(P) > eps:
            return (peak_price - P) / P
        return np.nan
    if op == 1:
        H = 0.0
        for k in range(nbins):
            p = q[k]
            if p > 0.0:
                H -= p * np.log(p)
        norm = np.log(float(nbins))
        if norm > 0.0:
            return H / norm
        return np.nan
    if op == 2:
        bar_p = 0.0
        for k in range(nbins):
            center = pmin + (k + 0.5) * bin_w
            bar_p += center * q[k]
        if abs(P) > eps:
            return (bar_p - P) / P
        return np.nan
    if op == 3:
        pos = (P - pmin) / bin_w
        if pos < 0.0:
            pos = 0.0
        if pos > float(nbins):
            pos = float(nbins)
        bP = int(pos)
        if bP >= nbins:
            bP = nbins - 1
        frac_below = pos - float(bP)
        if frac_below < 0.0:
            frac_below = 0.0
        if frac_below > 1.0:
            frac_below = 1.0
        below = 0.0
        for k in range(bP):
            below += q[k]
        below += q[bP] * frac_below
        return below - (1.0 - below)
    if op == 4:
        istar = 0
        qmax = q[0]
        for k in range(1, nbins):
            if q[k] > qmax:
                qmax = q[k]
                istar = k
        if istar == 0:
            qL = q[istar + 1]
            qR = q[istar + 1]
        elif istar == nbins - 1:
            qL = q[istar - 1]
            qR = q[istar - 1]
        else:
            qL = q[istar - 1]
            qR = q[istar + 1]
        return (2.0 * qmax - qL - qR) / (qmax + eps)
    return np.nan


@njit(cache=True)
def _chip_peak_curvature(q: np.ndarray, nbins: int, eps: float) -> float:
    istar = 0
    qmax = q[0]
    for k in range(1, nbins):
        if q[k] > qmax:
            qmax = q[k]
            istar = k
    if istar == 0:
        qL = q[istar + 1]
        qR = q[istar + 1]
    elif istar == nbins - 1:
        qL = q[istar - 1]
        qR = q[istar - 1]
    else:
        qL = q[istar - 1]
        qR = q[istar + 1]
    return (2.0 * qmax - qL - qR) / (qmax + eps)


@njit(cache=True)
def _chip_peak_fwhm_width(q: np.ndarray, nbins: int, bin_w: float, eps: float) -> float:
    istar = 0
    qmax = q[0]
    for k in range(1, nbins):
        if q[k] > qmax:
            qmax = q[k]
            istar = k
    half = 0.5 * qmax
    left = istar
    while left > 0 and q[left - 1] >= half:
        left -= 1
    right = istar
    while right < nbins - 1 and q[right + 1] >= half:
        right += 1
    return (float(right - left) + 1.0) * bin_w


@njit(cache=True)
def _chip_peak_fwhm(q: np.ndarray, nbins: int, bin_w: float, span: float, eps: float) -> float:
    fwhm = _chip_peak_fwhm_width(q, nbins, bin_w, eps)
    if span <= eps:
        return 1.0
    val = fwhm / span
    if val > 1.0:
        val = 1.0
    if val < 0.0:
        val = 0.0
    return val


@njit(cache=True)
def _chip_sigma_close_vol(
    close: np.ndarray, volume: np.ndarray, lo: int, hi: int, eps: float
) -> float:
    total = 0.0
    mean = 0.0
    for j in range(lo, hi + 1):
        v = close[j]
        m = volume[j]
        if (v == v) and (m == m) and m > 0.0:
            mean += v * m
            total += m
    if total <= 0.0:
        return eps
    mean /= total
    var = 0.0
    for j in range(lo, hi + 1):
        v = close[j]
        m = volume[j]
        if (v == v) and (m == m) and m > 0.0:
            d = v - mean
            var += m * d * d
    var /= total
    if var <= 0.0:
        return eps
    return np.sqrt(var) + eps


@njit(cache=True)
def _chip_hist_dip(q: np.ndarray, nbins: int) -> float:
    csum = 0.0
    for k in range(nbins):
        csum += q[k]
    if csum <= 0.0:
        return 0.0
    inv = 1.0 / csum
    max_dev = 0.0
    c = 0.0
    for k in range(nbins):
        c += q[k] * inv
        uni = float(k + 1) / float(nbins)
        dev = c - uni
        if dev < 0.0:
            dev = -dev
        if dev > max_dev:
            max_dev = dev
    return max_dev
