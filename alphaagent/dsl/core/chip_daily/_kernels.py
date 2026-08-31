"""主 njit kernels：roll_chip_metric / peak_sharpness / bimodal / wass_dist。"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

from ._helpers import (
    _chip_build_hist_window,
    _chip_hist_dip,
    _chip_metric_from_q,
    _chip_peak_curvature,
    _chip_peak_fwhm,
    _chip_peak_fwhm_width,
    _chip_pmin_pmax,
    _chip_sigma_close_vol,
)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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
