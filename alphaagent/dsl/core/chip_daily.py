"""日频筹码分布内核：uniform / cyq / triangular 三种构建方式。"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

# method: 0=uniform, 1=cyq, 2=triangular
_CHIP_METHOD = {
    "uniform": 0,
    "cyq": 1,
    "tri": 2,
    "triangular": 2,
}

# metric op: 0=peak_loc, 1=entropy, 2=com_w_gap, 3=mass_asym, 4=peak_sharpness
CHIP_OP = {
    "peak_loc": 0,
    "entropy": 1,
    "com_w_gap": 2,
    "mass_asym": 3,
    "peak_sharpness": 4,
}


def chip_method_id(method: str) -> int:
    k = str(method).strip().lower()
    if k not in _CHIP_METHOD:
        raise ValueError(
            'method must be "uniform", "cyq", or "tri" (alias: "triangular")'
        )
    return _CHIP_METHOD[k]


def chip_wass_implementation_id(name: str) -> int:
    k = str(name).strip().lower()
    if k == "moment":
        return 0
    if k in ("transport", "w1", "earth"):
        return 1
    raise ValueError(
        'implementation must be "moment" or "transport" (aliases: "w1", "earth")'
    )


def chip_peak_sharpness_impl_id(name: str) -> int:
    k = str(name).strip().lower()
    if k in ("curvature", "curv", "s_curv"):
        return 0
    if k in ("fwhm", "s_fwhm"):
        return 1
    if k in ("combined", "sharp", "s_sharp"):
        return 2
    raise ValueError('implementation must be "curvature", "fwhm", or "combined"')


def chip_bimodal_impl_id(name: str) -> int:
    k = str(name).strip().lower()
    if k in ("simple", "ratio"):
        return 0
    if k in ("dip", "hartigan"):
        return 1
    raise ValueError('implementation must be "simple" or "dip"')


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


@njit(cache=False)
def roll_chip_metric_daily_numba(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    op: int,
    method_id: int,
) -> np.ndarray:
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    eps = 1e-12
    q = np.zeros(nbins, dtype=np.float32)

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        P = close[i]
        if not (P == P):
            out[i] = np.nan
            continue

        pmin, bin_w, ok = _chip_build_hist_window(
            close, volume, low, high, aux, lo, i, nbins, method_id, q, eps
        )
        if not ok:
            if op == 4:
                out[i] = 1.0
            elif op in (0, 1, 2, 3):
                out[i] = 0.0
            else:
                out[i] = np.nan
            continue
        out[i] = _chip_metric_from_q(q, pmin, bin_w, nbins, P, op, eps)

    return out


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


@njit(cache=False)
def roll_chip_peak_sharpness_daily_numba(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    impl: int,
    method_id: int,
) -> np.ndarray:
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    eps = 1e-12
    q = np.zeros(nbins, dtype=np.float32)

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        if not (close[i] == close[i]):
            out[i] = np.nan
            continue

        pmin, bin_w, ok = _chip_build_hist_window(
            close, volume, low, high, aux, lo, i, nbins, method_id, q, eps
        )
        if not ok:
            out[i] = 1.0
            continue

        pmax = pmin + bin_w * float(nbins)
        span = pmax - pmin
        s_curv = _chip_peak_curvature(q, nbins, eps)
        if impl == 0:
            out[i] = s_curv
        elif impl == 1:
            out[i] = _chip_peak_fwhm(q, nbins, bin_w, span, eps)
        else:
            fwhm_price = _chip_peak_fwhm_width(q, nbins, bin_w, eps)
            sigma = _chip_sigma_close_vol(close, volume, lo, i, eps)
            decay = np.exp(-fwhm_price / (4.0 * sigma))
            val = s_curv * decay
            if val < 0.0:
                val = 0.0
            out[i] = val

    return out


@njit(cache=False)
def roll_chip_bimodal_daily_numba(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    impl: int,
    method_id: int,
    lambda_scale: float,
) -> np.ndarray:
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    eps = 1e-12
    q = np.zeros(nbins, dtype=np.float32)
    lam = lambda_scale
    if lam <= eps:
        lam = 1.0

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        if not (close[i] == close[i]):
            out[i] = np.nan
            continue

        pmin, bin_w, ok = _chip_build_hist_window(
            close, volume, low, high, aux, lo, i, nbins, method_id, q, eps
        )
        if not ok:
            out[i] = 0.0
            continue

        if impl == 1:
            out[i] = _chip_hist_dip(q, nbins)
            continue

        istar = 0
        qmax = q[0]
        for k in range(1, nbins):
            if q[k] > qmax:
                qmax = q[k]
                istar = k
        p_star = pmin + (istar + 0.5) * bin_w

        istar2 = -1
        q2max = 0.0
        for k in range(nbins):
            if k >= istar - 1 and k <= istar + 1:
                continue
            if q[k] > q2max:
                q2max = q[k]
                istar2 = k

        if istar2 < 0 or q2max <= eps:
            out[i] = 0.0
            continue

        R_peak = q2max / (qmax + eps)
        p_2 = pmin + (istar2 + 0.5) * bin_w
        sigma = _chip_sigma_close_vol(close, volume, lo, i, eps)
        D_peak = abs(p_star - p_2) / sigma
        out[i] = R_peak * np.exp(-(D_peak * D_peak) / (2.0 * lam * lam))

    return out


@njit(cache=False)
def roll_chip_wass_dist_daily_numba(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    rho: np.ndarray,
    nbins: int,
    impl: int,
    method_id: int,
) -> np.ndarray:
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    eps = 1e-12
    q_curr = np.zeros(nbins, dtype=np.float32)
    q_lag = np.zeros(nbins, dtype=np.float32)

    for i in range(n):
        w_a = wa[i]
        if w_a < 1:
            w_a = 1
        if w_a > i + 1:
            w_a = i + 1
        w_b = wb[i]
        if w_b < 1:
            w_b = 1
        rho_i = rho[i]
        if rho_i < 0:
            rho_i = 0

        hi_a = i
        lo_a = hi_a + 1 - w_a
        hi_b = i - rho_i
        if hi_b < 0:
            out[i] = np.nan
            continue
        if w_b > hi_b + 1:
            w_b = hi_b + 1
        lo_b = hi_b + 1 - w_b
        if lo_a < 0 or lo_b < 0:
            out[i] = np.nan
            continue

        P = close[i]
        if not (P == P):
            out[i] = np.nan
            continue

        pmin_a, pmax_a, ok_a = _chip_pmin_pmax(low, high, lo_a, hi_a, eps)
        pmin_b, pmax_b, ok_b = _chip_pmin_pmax(low, high, lo_b, hi_b, eps)
        if not ok_a or not ok_b:
            out[i] = np.nan
            continue

        pmin = pmin_a
        if pmin_b < pmin:
            pmin = pmin_b
        pmax = pmax_a
        if pmax_b > pmax:
            pmax = pmax_b
        bin_w = (pmax - pmin) / float(nbins)
        if bin_w <= eps:
            out[i] = np.nan
            continue
        span = pmax - pmin

        _, _, ok_ca = _chip_build_hist_window(
            close, volume, low, high, aux, lo_a, hi_a, nbins, method_id, q_curr, eps
        )
        _, _, ok_cb = _chip_build_hist_window(
            close, volume, low, high, aux, lo_b, hi_b, nbins, method_id, q_lag, eps
        )
        if not ok_ca or not ok_cb:
            out[i] = np.nan
            continue

        if impl == 0:
            mean_a = 0.0
            mean_b = 0.0
            for k in range(nbins):
                ck = pmin + (k + 0.5) * bin_w
                mean_a += ck * q_curr[k]
                mean_b += ck * q_lag[k]
            sig = (mean_a - mean_b) / (abs(P) + eps)
            if sig > 1.0:
                sig = 1.0
            elif sig < -1.0:
                sig = -1.0
            out[i] = sig
        else:
            partial = 0.0
            raw = 0.0
            for k in range(nbins - 1):
                partial += q_curr[k] - q_lag[k]
                raw += abs(partial)
            raw *= bin_w
            val = raw / (span + eps)
            if val > 1.0:
                val = 1.0
            elif val < 0.0:
                val = 0.0
            out[i] = val

    return out


def roll_chip_metric_daily(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    op: str,
    method: str,
) -> np.ndarray:
    op_id = CHIP_OP.get(op)
    if op_id is None:
        raise ValueError(f"Unknown chip op: {op}")
    mid = chip_method_id(method)
    arrays = [
        close.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        low.astype(np.float32, copy=False),
        high.astype(np.float32, copy=False),
        aux.astype(np.float32, copy=False),
    ]
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise ValueError("chip daily arrays must have the same length")
    return roll_chip_metric_daily_numba(
        arrays[0], arrays[1], arrays[2], arrays[3], arrays[4],
        int(window), int(nbins), int(op_id), int(mid),
    )


def roll_chip_peak_sharpness_daily(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    implementation: str,
    method: str,
) -> np.ndarray:
    impl = chip_peak_sharpness_impl_id(implementation)
    mid = chip_method_id(method)
    return roll_chip_peak_sharpness_daily_numba(
        close.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        low.astype(np.float32, copy=False),
        high.astype(np.float32, copy=False),
        aux.astype(np.float32, copy=False),
        int(window), int(nbins), int(impl), int(mid),
    )


def roll_chip_bimodal_daily(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    implementation: str,
    method: str,
    *,
    lambda_scale: float = 1.0,
) -> np.ndarray:
    impl = chip_bimodal_impl_id(implementation)
    mid = chip_method_id(method)
    return roll_chip_bimodal_daily_numba(
        close.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        low.astype(np.float32, copy=False),
        high.astype(np.float32, copy=False),
        aux.astype(np.float32, copy=False),
        int(window), int(nbins), int(impl), int(mid), float(lambda_scale),
    )


def roll_chip_wass_dist_daily(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    rho: np.ndarray,
    nbins: int,
    implementation: str,
    method: str,
) -> np.ndarray:
    impl = chip_wass_implementation_id(implementation)
    mid = chip_method_id(method)
    return roll_chip_wass_dist_daily_numba(
        close.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        low.astype(np.float32, copy=False),
        high.astype(np.float32, copy=False),
        aux.astype(np.float32, copy=False),
        wa.astype(np.int64, copy=False),
        wb.astype(np.int64, copy=False),
        rho.astype(np.int64, copy=False),
        int(nbins), int(impl), int(mid),
    )
