"""C++ 加速后端管理：统一检测、环境变量控制、双后端调用接口。

环境变量:
- FUTURE_ALPHA_MINER_ACCEL_BACKEND=cxx: 强制使用 C++ 扩展
- FUTURE_ALPHA_MINER_ACCEL_BACKEND=numba: 强制使用 Numba
- 未设置或 C++ 未编译: 自动选择可用后端
- C++ 路径下 ``n_jobs``/``parallel`` 参数为 ``None`` 时 **默认启用 OpenMP 并行**；
  传入 ``parallel=False`` 可强制串行。
"""
from __future__ import annotations

import importlib
import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from . import chip_daily as _chip_daily

_fam_accel = None  # type: ignore[misc]


def _load_cxx_backend():
    """Load the optional C++ backend without binding the core to one project."""
    module_name = os.environ.get("AQRA_DSL_CXX_BACKEND", "").strip()
    if not module_name:
        return None
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def set_cxx_backend(module) -> None:
    """Inject a compiled backend module from the project adapter layer."""
    global _fam_accel, _HAS_CXX, _CXX_ROLL_FIXED_MAX_OP
    _fam_accel = module
    _HAS_CXX = module is not None
    _CXX_ROLL_FIXED_MAX_OP = _cxx_roll_fixed_max_op()


_fam_accel = _load_cxx_backend()
_HAS_CXX = _fam_accel is not None

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    prange = range  # type: ignore[misc, assignment]

    def njit(*args, **kwargs):
        def _wrap(f):
            return f
        return _wrap if not args else args[0]


def _use_cxx_backend() -> bool:
    """根据环境变量和可用性决定是否使用 C++ 后端。"""
    backend = os.environ.get("FUTURE_ALPHA_MINER_ACCEL_BACKEND", "").strip().lower()
    if backend == "cxx":
        if not _HAS_CXX:
            raise RuntimeError("C++ extension requested but not available. "
                             "Install python3-devel and rebuild with 'pip install -e .'")
        return True
    if backend == "numba":
        return False
    # Auto: use C++ if available
    return _HAS_CXX


def accel_available() -> dict:
    """返回加速后端可用性信息。"""
    return {
        "cxx": _HAS_CXX,
        "numba": _HAS_NUMBA,
        "active": "cxx" if _use_cxx_backend() else ("numba" if _HAS_NUMBA else "python"),
    }


# =============================================================================
# Fixed-Window Rolling: MEAN, STD, SUM, MIN, MAX, RANK, VAR, MEDIAN, SKEW, KURT, PROD
# =============================================================================

_OP_MAP_FIXED = {
    "mean": 0,
    "std": 1,
    "sum": 2,
    "min": 3,
    "max": 4,
    "rank_pct": 5,
    "var": 6,
    "median": 7,
    "skew": 8,
    "kurt": 9,
    "prod": 10,
}


def _cxx_has(name: str) -> bool:
    """C++ 扩展是否已编译并包含指定符号（用于新算子的平滑回退）。"""
    return _HAS_CXX and hasattr(_fam_accel, name)


def _cxx_parallel(parallel: Optional[bool]) -> bool:
    """传给 C++ 内核的 OpenMP 开关：``parallel is None`` 时默认启用并行。"""
    return True if parallel is None else bool(parallel)


def _cxx_roll_fixed_max_op() -> int:
    """一次性探测当前编译的 C++ 内核 ``roll_fixed`` 支持到哪个 op，
    避免旧扩展与新 Python 层（op=9 kurt）版本错位。"""
    if not _HAS_CXX:
        return -1
    probe = np.zeros(4, dtype=np.float32)
    max_op = 8
    for op in range(9, 15):
        try:
            _fam_accel.roll_fixed(probe, 1, op, 1, False)
            max_op = op
        except Exception:
            break
    return max_op


_CXX_ROLL_FIXED_MAX_OP = _cxx_roll_fixed_max_op()


def roll_fixed(
    vals: np.ndarray,
    window: int,
    kind: str,
    *,
    ddof: int = 1,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗滚动聚合，自动选择 C++ 或 Numba 后端。
    
    Args:
        vals: 1-D float64 数组
        window: 窗口长度
        kind: "mean", "std", "sum", "min", "max", "rank_pct", "var", "median", "skew", "kurt", "prod"
        ddof: 标准差自由度（默认 1）
        parallel: 是否并行；``None`` 时在 C++ 后端下默认 ``True``（OpenMP）
    """
    use_cxx = _use_cxx_backend()
    op = _OP_MAP_FIXED.get(kind)
    if op is None:
        raise ValueError(f"Unknown kind: {kind}")
    
    if use_cxx and op <= _CXX_ROLL_FIXED_MAX_OP:
        return np.asarray(_fam_accel.roll_fixed(vals, window, op, ddof, _cxx_parallel(parallel)), dtype=np.float32)
    
    # Numba fallback (also used when C++ extension is older than current Python layer)
    return _roll_fixed_numba(vals, window, op, ddof)


@njit(cache=True)
def _roll_fixed_numba(vals: np.ndarray, window: int, op: int, ddof: int) -> np.ndarray:
    """Numba 实现的固定窗滚动（与 C++ 语义一致）。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        
        if op == 0:  # mean
            s = 0.0
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:  # not nan
                    s += v
                    c += 1
            out[i] = s / c if c > 0 else np.nan
        elif op == 1:  # std
            s = 0.0
            sq = 0.0
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    s += v
                    sq += v * v
                    c += 1
            if c > ddof:
                mean = s / c
                var = (sq - 2 * mean * s + c * mean * mean) / (c - ddof)
                out[i] = np.sqrt(max(0.0, var))
            else:
                out[i] = np.nan
        elif op == 2:  # sum
            s = 0.0
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    s += v
                    c += 1
            out[i] = s if c > 0 else np.nan
        elif op == 3:  # min
            m = np.inf
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v and v < m:
                    m = v
            out[i] = m if m != np.inf else np.nan
        elif op == 4:  # max
            m = -np.inf
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v and v > m:
                    m = v
            out[i] = m if m != -np.inf else np.nan
        elif op == 5:  # rank_pct — 与 pandas ``rolling().rank(pct=True)``：average rank / nvalid
            nvalid = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    nvalid += 1
            if nvalid == 0:
                out[i] = np.nan
            else:
                curr = vals[i]
                if curr != curr:
                    out[i] = np.nan
                else:
                    less = 0
                    equal = 0
                    for j in range(lo, i + 1):
                        v = vals[j]
                        if v == v:
                            if v < curr:
                                less += 1
                            elif v == curr:
                                equal += 1
                    rank_low = float(less + 1)
                    rank_high = float(less + equal)
                    rank_avg = (rank_low + rank_high) / 2.0
                    out[i] = rank_avg / float(nvalid)
        elif op == 6:  # var
            s = 0.0
            sq = 0.0
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    s += v
                    sq += v * v
                    c += 1
            if c > ddof:
                mean = s / c
                var = (sq - 2 * mean * s + c * mean * mean) / (c - ddof)
                out[i] = max(0.0, var)
            else:
                out[i] = np.nan
        elif op == 7:  # median
            wlen = i - lo + 1
            tmp = np.empty(wlen, dtype=np.float32)
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    tmp[c] = v
                    c += 1
            if c == 0:
                out[i] = np.nan
            else:
                buf = np.sort(tmp[:c])
                if c % 2 == 1:
                    out[i] = buf[c // 2]
                else:
                    out[i] = (buf[c // 2 - 1] + buf[c // 2]) / 2.0
        elif op == 8:  # skew — pandas nanops.nanskew
            wlen = i - lo + 1
            tmp = np.empty(wlen, dtype=np.float32)
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    tmp[c] = v
                    c += 1
            if c < 3:
                out[i] = np.nan
            else:
                buf = tmp[:c]
                sm = 0.0
                for t in range(c):
                    sm += buf[t]
                mean = sm / c
                m2 = 0.0
                m3 = 0.0
                for t in range(c):
                    d = buf[t] - mean
                    d2 = d * d
                    m2 += d2
                    m3 += d2 * d
                if m2 == 0.0:
                    out[i] = 0.0
                else:
                    out[i] = (
                        (c * np.sqrt(c - 1) / (c - 2)) * (m3 / (m2 ** 1.5))
                    )
        elif op == 9:  # kurt — pandas rolling().kurt() adjusted Fisher–Pearson (excess)
            wlen = i - lo + 1
            tmp = np.empty(wlen, dtype=np.float32)
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    tmp[c] = v
                    c += 1
            if c < 4:
                out[i] = np.nan
            else:
                buf = tmp[:c]
                sm = 0.0
                for t in range(c):
                    sm += buf[t]
                mean = sm / c
                m2 = 0.0
                m4 = 0.0
                for t in range(c):
                    d = buf[t] - mean
                    d2 = d * d
                    m2 += d2
                    m4 += d2 * d2
                if m2 == 0.0:
                    out[i] = 0.0
                else:
                    nf = float(c)
                    numer = nf * (nf - 1.0) * (nf + 1.0) * m4
                    denom = (nf - 2.0) * (nf - 3.0) * m2 * m2
                    adj = 3.0 * (nf - 1.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0))
                    out[i] = numer / denom - adj
        elif op == 10:  # prod — NaN as 1
            p = 1.0
            for j in range(lo, i + 1):
                v = vals[j]
                p *= v if v == v else 1.0
            out[i] = p
        else:
            out[i] = np.nan
    
    return out


# =============================================================================
# Fixed shift (DELAY 整数窗)
# =============================================================================


def shift_fixed(
    vals: np.ndarray,
    periods: int,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """与 ``pandas.Series.shift(periods)`` 对齐的 1-D 滞后；C++ 可用时用 ``shift_fixed`` 内核。"""
    p = int(periods)
    if p < 0:
        p = 0
    use_cxx = _use_cxx_backend()
    if use_cxx:
        return np.asarray(
            _fam_accel.shift_fixed(vals, p, _cxx_parallel(parallel)), dtype=np.float32
        )
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        j = i - p
        out[i] = vals[j] if j >= 0 else np.nan
    return out


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================


def ema(vals: np.ndarray, span: int, parallel: Optional[bool] = None) -> np.ndarray:
    """与 ``pandas.Series.ewm(span, min_periods=1, adjust=False).mean()`` 一致。

    在 C++ 扩展可用且未强制 Numba 时使用 ``_fam_accel.ema``；否则回退到 pandas。
    """
    use_cxx = _use_cxx_backend()
    if use_cxx:
        return np.asarray(
            _fam_accel.ema(vals, span, _cxx_parallel(parallel)), dtype=np.float32
        )
    return (
        pd.Series(vals, dtype=np.float32)
        .ewm(span=span, min_periods=1, adjust=False)
        .mean()
        .to_numpy(dtype=np.float32, copy=False)
    )


# =============================================================================
# WMA (Weighted Moving Average)
# =============================================================================


def wma(vals: np.ndarray, window: int, parallel: Optional[bool] = None) -> np.ndarray:
    """WMA，自动选择 C++ 或 Numba 后端。"""
    use_cxx = _use_cxx_backend()
    
    if use_cxx:
        return np.asarray(_fam_accel.wma(vals, window, _cxx_parallel(parallel)), dtype=np.float32)
    
    return _wma_numba(vals, window)


@njit(cache=True)
def _wma_numba(vals: np.ndarray, window: int) -> np.ndarray:
    """与 ``function_registry`` 原 pandas 实现一致：窗口长 L 时使用 ``1..window`` 的后 L 项作权。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        L = i - lo + 1
        first_w = float(window - L + 1)
        
        weighted_sum = 0.0
        weight_sum = 0.0
        for t in range(L):
            j = lo + t
            v = vals[j]
            if v == v:
                weight = first_w + float(t)
                weighted_sum += v * weight
                weight_sum += weight
        
        out[i] = weighted_sum / weight_sum if weight_sum > 0 else np.nan
    
    return out


# =============================================================================
# DELTA (Difference)
# =============================================================================


def delta(vals: np.ndarray, periods: int = 1, parallel: Optional[bool] = None) -> np.ndarray:
    """DELTA，自动选择 C++ 或 Numba 后端。"""
    use_cxx = _use_cxx_backend()
    
    if use_cxx:
        return np.asarray(_fam_accel.delta(vals, periods, _cxx_parallel(parallel)), dtype=np.float32)
    
    return _delta_numba(vals, periods)


@njit(cache=True)
def _delta_numba(vals: np.ndarray, periods: int) -> np.ndarray:
    """Numba DELTA 实现。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        j = i - periods
        if j >= 0:
            curr = vals[i]
            prev = vals[j]
            if curr == curr and prev == prev:  # both not nan
                out[i] = curr - prev
            else:
                out[i] = np.nan
        else:
            out[i] = np.nan
    
    return out


# =============================================================================
# TS_PCTCHANGE (Percent Change)
# =============================================================================


def pctchange(vals: np.ndarray, periods: int = 1, parallel: Optional[bool] = None) -> np.ndarray:
    """TS_PCTCHANGE，自动选择 C++ 或 Numba 后端。"""
    use_cxx = _use_cxx_backend()
    
    if use_cxx:
        return np.asarray(_fam_accel.pctchange(vals, periods, _cxx_parallel(parallel)), dtype=np.float32)
    
    return _pctchange_numba(vals, periods)


@njit(cache=True)
def _pctchange_numba(vals: np.ndarray, periods: int) -> np.ndarray:
    """Numba TS_PCTCHANGE 实现。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        j = i - periods
        if j >= 0:
            curr = vals[i]
            prev = vals[j]
            if curr == curr and prev == prev and prev != 0.0:
                r = (curr - prev) / prev
                # Handle inf
                if np.isinf(r):
                    out[i] = np.nan
                else:
                    out[i] = r
            else:
                out[i] = np.nan
        else:
            out[i] = np.nan
    
    return out


# =============================================================================
# TS_LAST_ARGPEAK / … (confirmed centered local extremes) + 最大左右振幅选峰/谷
# =============================================================================


@njit(cache=True)
def _marks_center_extreme_numba(
    vals: np.ndarray, half_window: int, want_max: int
) -> np.ndarray:
    """某位置 ``j`` 为中心、窗 ``[j-w,j+w]`` 内的局部峰(1) / 谷(0)；并列取窗内最右侧。"""
    n = vals.shape[0]
    marks = np.zeros(n, dtype=np.uint8)
    if n == 0 or half_window < 1:
        return marks
    for j in range(half_window, n - half_window):
        center = vals[j]
        if center != center:
            continue
        lo = j - half_window
        hi = j + half_window
        ok = True
        if want_max == 1:
            for t in range(lo, hi + 1):
                v = vals[t]
                if v == v and (v > center or (t > j and v == center)):
                    ok = False
                    break
        else:
            for t in range(lo, hi + 1):
                v = vals[t]
                if v == v and (v < center or (t > j and v == center)):
                    ok = False
                    break
        if ok:
            marks[j] = 1
    return marks


def arg_local_extreme(
    vals: np.ndarray,
    half_window: int = 10,
    *,
    want_max: bool,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """最近一次已确认中心局部极值距今 bar 数。

    定义某位置 ``j`` 为 peak/trough 的条件是：``vals[j]`` 在
    ``[j-half_window, j+half_window]`` 内分别为最高/最低值。为了避免数据窥探，
    只有在 ``j+half_window`` 时刻之后，该拐点才会被确认并出现在输出里。

    对并列极值采用“右侧优先”消歧：若窗口内右边还有相同高/低点，则当前 ``j`` 不算，
    仅保留该中心窗内最右侧的那个极值点。
    """
    hw = int(half_window)
    if hw < 1:
        raise ValueError("half_window must be >= 1")
    use_cxx = _use_cxx_backend()

    if use_cxx:
        return np.asarray(
            _fam_accel.arg_local_extreme(vals, hw, 1 if want_max else 0, _cxx_parallel(parallel)),
            dtype=np.float32,
        )

    return _arg_local_extreme_numba(vals, hw, 1 if want_max else 0)


def local_extreme_value(
    vals: np.ndarray,
    half_window: int = 10,
    *,
    want_max: bool,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """最近一次已确认中心局部峰/谷的价格值。"""
    hw = int(half_window)
    if hw < 1:
        raise ValueError("half_window must be >= 1")
    use_cxx = _use_cxx_backend()

    if use_cxx:
        return np.asarray(
            _fam_accel.local_extreme_value(
                vals, hw, 1 if want_max else 0, _cxx_parallel(parallel)
            ),
            dtype=np.float32,
        )

    return _local_extreme_value_numba(vals, hw, 1 if want_max else 0)


@njit(cache=True)
def _arg_local_extreme_numba(vals: np.ndarray, half_window: int, want_max: int) -> np.ndarray:
    """Numba 版本的已确认局部峰/谷定位。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        out[i] = np.nan

    if n == 0 or half_window < 1:
        return out

    marks = _marks_center_extreme_numba(vals, half_window, want_max)

    last = -1
    for i in range(n):
        cand = i - half_window
        if cand >= 0 and marks[cand] == 1:
            last = cand
        if last >= 0:
            out[i] = float(i - last)

    return out


@njit(cache=True)
def _local_extreme_value_numba(
    vals: np.ndarray, half_window: int, want_max: int
) -> np.ndarray:
    """Numba 版本的最近一次已确认局部峰/谷价格。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        out[i] = np.nan

    if n == 0 or half_window < 1:
        return out

    marks = _marks_center_extreme_numba(vals, half_window, want_max)

    last = -1
    for i in range(n):
        cand = i - half_window
        if cand >= 0 and marks[cand] == 1:
            last = cand
        if last >= 0:
            out[i] = vals[last]

    return out


# =============================================================================
# 三 K 线顶/底分型（高+低联合判定），带 1 根确认延迟
# =============================================================================


@njit(cache=True)
def _fractal_chan_3bar_marks(high: np.ndarray, low: np.ndarray, want_top: int) -> np.ndarray:
    """中心下标 ``j`` 上是否形成分型（严格不等）；与 bar 1,2,3 = j-1,j,j+1 对应。"""
    n = high.shape[0]
    marks = np.zeros(n, dtype=np.uint8)
    for j in range(1, n - 1):
        h0 = high[j - 1]
        h1 = high[j]
        h2 = high[j + 1]
        l0 = low[j - 1]
        l1 = low[j]
        l2 = low[j + 1]
        if not (h0 == h0 and h1 == h1 and h2 == h2 and l0 == l0 and l1 == l1 and l2 == l2):
            continue
        if want_top == 1:
            if h0 < h1 and h1 > h2 and l0 < l1 and l1 > l2:
                marks[j] = 1
        else:
            if h0 > h1 and h1 < h2 and l0 > l1 and l1 < l2:
                marks[j] = 1
    return marks


@njit(cache=True)
def _fractal_chan_last_from_marks(
    high: np.ndarray,
    low: np.ndarray,
    marks: np.ndarray,
    want_top: int,
    want_arg: int,
) -> np.ndarray:
    """分型中心 ``j`` 仅在 ``i=j+1``（第三根收盘）后确认；与 TS_LAST_ARGPEAK 的滞后对齐方式一致。"""
    n = high.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan
    last = -1
    for i in range(n):
        cand = i - 1
        if cand >= 1 and cand < n - 1 and marks[cand] == 1:
            last = cand
        if last >= 0:
            if want_arg == 1:
                out[i] = float(i - last)
            else:
                out[i] = high[last] if want_top == 1 else low[last]
    return out


def fractal_chan_3bar_last(
    high: np.ndarray,
    low: np.ndarray,
    *,
    want_top_fractal: bool,
    want_arg: bool,
) -> np.ndarray:
    """三 K 线顶/底分型：双输入 high/low，输出距今 bar 数或分型中枢价（顶=中 K 高、底=中 K 低）。"""
    h = np.asarray(high, dtype=np.float32)
    l = np.asarray(low, dtype=np.float32)
    if h.shape != l.shape:
        raise ValueError("fractal_chan_3bar_last: high and low must have the same shape")
    wt = 1 if want_top_fractal else 0
    wa = 1 if want_arg else 0
    marks = _fractal_chan_3bar_marks(h, l, wt)
    return _fractal_chan_last_from_marks(h, l, marks, wt, wa)


def _rolling_ending_min_max(
    vals: np.ndarray, half_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对每个 ``j``：``[j-w,j]`` 与 ``[j,j+w]`` 上的 ``min`` / ``max``（O(n) ``rolling``，含反序）。"""
    s = pd.Series(np.asarray(vals, dtype=np.float32), copy=False)
    w = int(half_window)
    if w < 1:
        raise ValueError("half_window must be >= 1")
    k = w + 1
    left_min = s.rolling(window=k, min_periods=1).min().to_numpy(dtype=np.float32, copy=False)
    left_max = s.rolling(window=k, min_periods=1).max().to_numpy(dtype=np.float32, copy=False)
    sr = s.iloc[::-1]
    rmin = sr.rolling(window=k, min_periods=1).min().iloc[::-1].to_numpy(
        dtype=np.float32, copy=False
    )
    rmax = sr.rolling(window=k, min_periods=1).max().iloc[::-1].to_numpy(
        dtype=np.float32, copy=False
    )
    return left_min, rmin, left_max, rmax


def _maxamp_from_marks(
    vals: np.ndarray,
    half_window: int,
    want_max: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """在已确认峰/谷中，选左右振幅和最大的那一个；与 ``_marks_center_extreme_numba`` 一致。"""
    w = int(half_window)
    n = len(vals)
    out_arg = np.full(n, np.nan, dtype=np.float32)
    out_val = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or w < 1:
        return out_arg, out_val

    marks = _marks_center_extreme_numba(
        np.asarray(vals, dtype=np.float32), w, 1 if want_max else 0
    )
    lmin, rmin, lmax, rmax = _rolling_ending_min_max(vals, w)

    best_a = -np.inf
    best_j = -1
    for i in range(n):
        cj = i - w
        if 2 * w <= i < n and marks[cj] == 1:
            vj = float(vals[cj])
            if want_max:
                a = 2.0 * vj - float(lmin[cj]) - float(rmin[cj])
            else:
                a = float(lmax[cj]) + float(rmax[cj]) - 2.0 * vj
            if np.isfinite(a) and (
                a > best_a or (a == best_a and cj > best_j)
            ):
                best_a, best_j = a, cj
        if best_j >= 0:
            out_arg[i] = float(i - best_j)
            out_val[i] = float(vals[best_j])

    return out_arg, out_val


def maxamp_arg_local_extreme(
    vals: np.ndarray,
    half_window: int = 10,
    *,
    want_max: bool,
) -> np.ndarray:
    """在已确认局部峰/谷中，选（左峰谷距 + 右峰谷距）最大者，输出距今 bar 数。

    不在 C++ 中实现，始终与 ``rolling``/marks 的 Numba 路径一致；避免 C++ 与 Python 双份语义。
    """
    a, _ = _maxamp_from_marks(vals, int(half_window), want_max)
    return a


def maxamp_local_extreme_value(
    vals: np.ndarray,
    half_window: int = 10,
    *,
    want_max: bool,
) -> np.ndarray:
    """在已确认局部峰/谷中，选左右振幅和最大者，输出该点价格。"""
    _, v = _maxamp_from_marks(vals, int(half_window), want_max)
    return v


# =============================================================================
# Rolling Covariance & Correlation (bivariate, fixed window)
# =============================================================================


@njit(cache=True)
def _roll_cov_numba(xvals: np.ndarray, yvals: np.ndarray, window: int, ddof: int) -> np.ndarray:
    n = xvals.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w

        sx = 0.0
        sy = 0.0
        sxy = 0.0
        c = 0
        for j in range(lo, i + 1):
            x = xvals[j]
            y = yvals[j]
            if x == x and y == y:  # both not NaN
                sx += x
                sy += y
                sxy += x * y
                c += 1
        if c > ddof:
            mx = sx / c
            my = sy / c
            out[i] = (sxy - c * mx * my) / (c - ddof)
        else:
            out[i] = np.nan

    return out


@njit(cache=True)
def _roll_corr_numba(xvals: np.ndarray, yvals: np.ndarray, window: int) -> np.ndarray:
    n = xvals.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w

        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        c = 0
        for j in range(lo, i + 1):
            x = xvals[j]
            y = yvals[j]
            if x == x and y == y:
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
                c += 1
        if c < 2:
            out[i] = np.nan
        else:
            cn = float(c)
            mx = sx / cn
            my = sy / cn
            vx = sxx - cn * mx * mx
            vy = syy - cn * my * my
            if vx <= 0.0 or vy <= 0.0:
                out[i] = np.nan
            else:
                cov = sxy - cn * mx * my
                out[i] = cov / np.sqrt(vx * vy)

    return out


def roll_cov_fixed(
    xvals: np.ndarray,
    yvals: np.ndarray,
    window: int,
    *,
    ddof: int = 1,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗滚动协方差，自动选择 C++ 或 Numba 后端。"""
    use_cxx = _use_cxx_backend()
    if use_cxx:
        return np.asarray(_fam_accel.roll_cov_fixed(xvals, yvals, window, ddof, _cxx_parallel(parallel)), dtype=np.float32)
    return _roll_cov_numba(xvals, yvals, window, ddof)


def roll_corr_fixed(
    xvals: np.ndarray,
    yvals: np.ndarray,
    window: int,
    *,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗滚动 Pearson 相关系数，自动选择 C++ 或 Numba 后端。"""
    use_cxx = _use_cxx_backend()
    if use_cxx:
        return np.asarray(_fam_accel.roll_corr_fixed(xvals, yvals, window, _cxx_parallel(parallel)), dtype=np.float32)
    return _roll_corr_numba(xvals, yvals, window)


# =============================================================================
# Rolling Quantile (fixed window, linear interpolation)
# =============================================================================


@njit(cache=True)
def _roll_quantile_numba(vals: np.ndarray, window: int, q: float) -> np.ndarray:
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w

        wlen = i - lo + 1
        tmp = np.empty(wlen, dtype=np.float32)
        c = 0
        for j in range(lo, i + 1):
            v = vals[j]
            if v == v:
                tmp[c] = v
                c += 1
        if c == 0:
            out[i] = np.nan
        else:
            buf = np.sort(tmp[:c])
            k = float(c)
            pos = q * (k - 1.0)
            if pos < 0.0:
                pos = 0.0
            if pos > k - 1.0:
                pos = k - 1.0
            lo_i = int(np.floor(pos))
            hi_i = int(np.ceil(pos))
            frac = pos - float(lo_i)
            out[i] = buf[lo_i] * (1.0 - frac) + buf[hi_i] * frac

    return out


def roll_quantile_fixed(
    vals: np.ndarray,
    window: int,
    q: float,
    *,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗滚动 q 分位数（线性插值，等价 pandas ``rolling(w, min_periods=1).quantile(q)``）。"""
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    if _use_cxx_backend() and _cxx_has("roll_quantile_fixed"):
        return np.asarray(_fam_accel.roll_quantile_fixed(vals, window, q, _cxx_parallel(parallel)), dtype=np.float32)
    return _roll_quantile_numba(vals, window, float(q))


# =============================================================================
# Event-driven kernels: TS_SINCE, TS_RUNLENGTH, TS_CROSS
# =============================================================================


@njit(cache=True)
def _ts_since_numba(cond: np.ndarray) -> np.ndarray:
    """距上一次 truthy（有限且非零）的 bar 数；首个事件前为 NaN。"""
    n = cond.shape[0]
    out = np.empty(n, dtype=np.float32)
    last = -1
    for i in range(n):
        v = cond[i]
        if v == v and v != 0.0:
            last = i
        if last >= 0:
            out[i] = float(i - last)
        else:
            out[i] = np.nan
    return out


def ts_since(cond: np.ndarray) -> np.ndarray:
    """距上一次 cond 为真（有限且非零）的 bar 数。"""
    if _use_cxx_backend() and _cxx_has("ts_since"):
        return np.asarray(_fam_accel.ts_since(cond), dtype=np.float32)
    return _ts_since_numba(cond)


@njit(cache=True)
def _ts_since_nth_numba(cond: np.ndarray, nth: int) -> np.ndarray:
    """距倒数第 nth 次 truthy 事件的 bar 数；nth=1 同 ``_ts_since_numba``。"""
    n = cond.shape[0]
    n_ev = max(1, int(nth))
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seen = 0
        target = -1
        for j in range(i, -1, -1):
            v = cond[j]
            if v == v and v != 0.0:
                seen += 1
                if seen == n_ev:
                    target = j
                    break
        if target >= 0:
            out[i] = float(i - target)
        else:
            out[i] = np.nan
    return out


def ts_since_nth(cond: np.ndarray, nth: int) -> np.ndarray:
    """距倒数第 ``nth`` 次 truthy 事件的 bar 数（``nth=1`` 同 ``ts_since``）。"""
    n_ev = max(1, int(nth))
    if _use_cxx_backend() and _cxx_has("ts_since_nth"):
        return np.asarray(_fam_accel.ts_since_nth(cond, n_ev), dtype=np.float32)
    return _ts_since_nth_numba(cond, n_ev)


@njit(cache=True)
def _ts_runlength_numba(vals: np.ndarray, direction: int) -> np.ndarray:
    """连续严格上行 / 下行 bar 数；NaN 位置输出 NaN 并重置计数。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    run = 0
    for i in range(n):
        curr = vals[i]
        if not (curr == curr):
            run = 0
            out[i] = np.nan
            continue
        if i == 0:
            run = 0
            out[i] = 0.0
            continue
        prev = vals[i - 1]
        if not (prev == prev):
            run = 0
            out[i] = 0.0
            continue
        if direction > 0:
            hit = curr > prev
        else:
            hit = curr < prev
        if hit:
            run += 1
        else:
            run = 0
        out[i] = float(run)
    return out


def ts_runlength(vals: np.ndarray, direction: int) -> np.ndarray:
    """连续严格上涨 (direction=1) 或下跌 (direction=-1) 的根数。"""
    if direction not in (1, -1):
        raise ValueError("direction must be 1 (up) or -1 (down)")
    if _use_cxx_backend() and _cxx_has("ts_runlength"):
        return np.asarray(_fam_accel.ts_runlength(vals, direction), dtype=np.float32)
    return _ts_runlength_numba(vals, direction)


@njit(cache=True)
def _ts_cross_numba(x: np.ndarray, y: np.ndarray, direction: int) -> np.ndarray:
    """上穿 (direction=1) / 下穿 (direction=-1) 事件；输出 0/1，缺失为 NaN。"""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n > 0:
        out[0] = 0.0
    for i in range(1, n):
        xc = x[i]
        yc = y[i]
        xp = x[i - 1]
        yp = y[i - 1]
        if not (xc == xc and yc == yc and xp == xp and yp == yp):
            out[i] = np.nan
            continue
        if direction > 0:
            out[i] = 1.0 if (xp <= yp and xc > yc) else 0.0
        else:
            out[i] = 1.0 if (xp >= yp and xc < yc) else 0.0
    return out


def ts_cross(x: np.ndarray, y: np.ndarray, direction: int) -> np.ndarray:
    """上穿 (direction=1) / 下穿 (direction=-1)：返回 0/1 / NaN 的 1-D 面板。"""
    if direction not in (1, -1):
        raise ValueError("direction must be 1 (above) or -1 (below)")
    if _use_cxx_backend() and _cxx_has("ts_cross"):
        return np.asarray(_fam_accel.ts_cross(x, y, direction), dtype=np.float32)
    return _ts_cross_numba(x, y, direction)


# Event rolling: op 0=count, 1=rate, 2=any, 3=all (truthy = finite && != 0)


@njit(cache=True)
def _ts_event_roll_numba(cond: np.ndarray, window: int, op: int) -> np.ndarray:
    n = cond.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        finite_cnt = 0
        truthy_cnt = 0
        for j in range(lo, i + 1):
            v = cond[j]
            if v == v:
                finite_cnt += 1
                if v != 0.0:
                    truthy_cnt += 1
        if finite_cnt == 0:
            out[i] = np.nan
        elif op == 0:
            out[i] = float(truthy_cnt)
        elif op == 1:
            out[i] = float(truthy_cnt) / float(finite_cnt)
        elif op == 2:
            out[i] = 1.0 if truthy_cnt > 0 else 0.0
        elif op == 3:
            out[i] = 1.0 if truthy_cnt == finite_cnt else 0.0
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def _ts_event_roll_dyn_numba(cond: np.ndarray, wvals: np.ndarray, op: int) -> np.ndarray:
    n = cond.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        w = int(wvals[i])
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        finite_cnt = 0
        truthy_cnt = 0
        for j in range(lo, i + 1):
            v = cond[j]
            if v == v:
                finite_cnt += 1
                if v != 0.0:
                    truthy_cnt += 1
        if finite_cnt == 0:
            out[i] = np.nan
        elif op == 0:
            out[i] = float(truthy_cnt)
        elif op == 1:
            out[i] = float(truthy_cnt) / float(finite_cnt)
        elif op == 2:
            out[i] = 1.0 if truthy_cnt > 0 else 0.0
        elif op == 3:
            out[i] = 1.0 if truthy_cnt == finite_cnt else 0.0
        else:
            out[i] = np.nan
    return out


def ts_event_roll(cond: np.ndarray, window: int, op: int) -> np.ndarray:
    """固定窗事件滚动：op 0=count, 1=rate, 2=any, 3=all。"""
    if op not in (0, 1, 2, 3):
        raise ValueError("op must be 0=count, 1=rate, 2=any, 3=all")
    w = max(1, int(window))
    if _use_cxx_backend() and _cxx_has("ts_event_roll"):
        return np.asarray(_fam_accel.ts_event_roll(cond, w, op), dtype=np.float32)
    return _ts_event_roll_numba(cond, w, op)


def ts_event_roll_dyn(cond: np.ndarray, wvals: np.ndarray, op: int) -> np.ndarray:
    """动态窗事件滚动；``wvals`` 为每 bar 窗长（≥1 整数）。"""
    if op not in (0, 1, 2, 3):
        raise ValueError("op must be 0=count, 1=rate, 2=any, 3=all")
    return _ts_event_roll_dyn_numba(cond, wvals.astype(np.int64, copy=False), op)


@njit(cache=True)
def _ts_streak_numba(cond: np.ndarray) -> np.ndarray:
    n = cond.shape[0]
    out = np.empty(n, dtype=np.float32)
    run = 0
    for i in range(n):
        v = cond[i]
        if not (v == v):
            run = 0
            out[i] = np.nan
            continue
        if v != 0.0:
            run += 1
            out[i] = float(run)
        else:
            run = 0
            out[i] = 0.0
    return out


def ts_streak(cond: np.ndarray) -> np.ndarray:
    """当前连续 truthy 根数；NaN 重置。"""
    if _use_cxx_backend() and _cxx_has("ts_streak"):
        return np.asarray(_fam_accel.ts_streak(cond), dtype=np.float32)
    return _ts_streak_numba(cond)


# =============================================================================
# TS_ARGMEDIAN — 窗口内最接近中位数的位置（距今 bar 数）
# =============================================================================


@njit(cache=True)
def _arg_median_numba(vals: np.ndarray, window: int) -> np.ndarray:
    """返回窗口内值最接近中位数的 bar 距今数（0=当前）。
    若有多个相同距离，取最近（索引最大）的。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w

        # 收集有效值及其原始索引偏移
        wlen = i - lo + 1
        tmp = np.empty(wlen, dtype=np.float32)
        idx_map = np.empty(wlen, dtype=np.int64)
        c = 0
        for j in range(lo, i + 1):
            v = vals[j]
            if v == v:
                tmp[c] = v
                idx_map[c] = j - lo  # 0-based offset within window
                c += 1

        if c == 0:
            out[i] = np.nan
            continue

        # 求中位数
        buf = tmp[:c]
        med = np.median(buf)

        # 找最接近中位数的元素，距离相同取最近的（索引大的优先）
        best_dist = 1e308
        best_offset = 0
        for k in range(c):
            d = abs(tmp[k] - med)
            offset = idx_map[k]
            if d < best_dist or (d == best_dist and offset > best_offset):
                best_dist = d
                best_offset = offset

        out[i] = float(wlen - 1 - best_offset)
    return out


def arg_median_fixed(
    vals: np.ndarray,
    window: int,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗 TS_ARGMEDIAN：返回窗口内最接近中位数的 bar 距今数。"""
    return _arg_median_numba(vals, window)


# =============================================================================
# TS_ARGNTH — 窗口内第 n 大/小的位置（距今 bar 数）
# =============================================================================


@njit(cache=True)
def _arg_nth_numba(vals: np.ndarray, window: int, n: int, ascending: bool, unique: bool = False) -> np.ndarray:
    """返回窗口内第 n 大 (ascending=False) 或第 n 小 (ascending=True) 的 bar 距今数。
    n >= 1；若有效值不足 n 个（unique=True 时为不足 n 个不同值），输出 NaN。
    若有重复值：
      - unique=False: 取位置最近（索引最大）的
      - unique=True: 跳过重复值，找严格第 n 个不同的值"""
    n = max(1, n)
    out = np.empty(vals.shape[0], dtype=np.float32)
    for i in range(vals.shape[0]):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w

        wlen = i - lo + 1
        # 收集有效值及其原始索引偏移
        max_valid = wlen
        val_buf = np.empty(max_valid, dtype=np.float32)
        idx_buf = np.empty(max_valid, dtype=np.int64)
        c = 0
        for j in range(lo, i + 1):
            v = vals[j]
            if v == v:
                val_buf[c] = v
                idx_buf[c] = j - lo  # offset within window
                c += 1

        if c == 0:
            out[i] = np.nan
            continue

        # 排序
        for ii in range(c):
            for jj in range(ii + 1, c):
                vi, vj = val_buf[ii], val_buf[jj]
                ii_idx, jj_idx = idx_buf[ii], idx_buf[jj]
                if not ascending:  # descending: 大的在前
                    # 降序：值大的在前，值相同则偏移大的在前（更近）
                    if vj > vi or (vj == vi and jj_idx > ii_idx):
                        val_buf[ii], val_buf[jj] = vj, vi
                        idx_buf[ii], idx_buf[jj] = jj_idx, ii_idx
                else:  # ascending: 小的在前
                    # 升序：值小的在前，值相同则偏移大的在前
                    if vj < vi or (vj == vi and jj_idx > ii_idx):
                        val_buf[ii], val_buf[jj] = vj, vi
                        idx_buf[ii], idx_buf[jj] = jj_idx, ii_idx

        if unique:
            # 去重：只保留每个值的第一个（排序后最近的）
            uniq_val = np.empty(c, dtype=np.float32)
            uniq_idx = np.empty(c, dtype=np.int64)
            uniq_c = 0
            last_val = np.nan
            for k in range(c):
                if val_buf[k] != last_val:
                    uniq_val[uniq_c] = val_buf[k]
                    uniq_idx[uniq_c] = idx_buf[k]
                    uniq_c += 1
                    last_val = val_buf[k]
            if uniq_c < n:
                out[i] = np.nan
                continue
            target_offset = uniq_idx[n - 1]
        else:
            if c < n:
                out[i] = np.nan
                continue
            target_offset = idx_buf[n - 1]

        out[i] = float(wlen - 1 - target_offset)
    return out

def arg_nth_fixed(
    vals: np.ndarray,
    window: int,
    n: int,
    ascending: bool,
    unique: bool = False,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗 TS_ARGNTH：第 n 大 (ascending=False) 或第 n 小 (ascending=True) 的 bar 距今数。
    Args:
        ascending: 排序方向，False=降序（大的在前），True=升序（小的在前）
        unique: 若为 True，跳过重复值，找严格第 n 个不同的值
    """
    return _arg_nth_numba(vals, window, n, ascending, unique)


# =============================================================================
# Rolling Spearman (rank) Correlation — bivariate, fixed window
# =============================================================================


@njit(cache=True)
def _roll_rankcorr_numba(xvals: np.ndarray, yvals: np.ndarray, window: int) -> np.ndarray:
    """与 C++ ``roll_rankcorr_fixed_impl`` 语义一致：
    每根 bar 收集窗口内 (x,y) 有效对，对 x、y 分别算平均秩（等分布 ``rank(method='average')``），
    再对秩序列求 Pearson 相关系数。有效对数 <2 或任一维方差为零输出 NaN。"""
    n = xvals.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w

        wlen = i - lo + 1
        xb = np.empty(wlen, dtype=np.float32)
        yb = np.empty(wlen, dtype=np.float32)
        c = 0
        for j in range(lo, i + 1):
            xv = xvals[j]
            yv = yvals[j]
            if xv == xv and yv == yv:
                xb[c] = xv
                yb[c] = yv
                c += 1

        if c < 2:
            out[i] = np.nan
            continue

        xr = np.empty(c, dtype=np.float32)
        yr = np.empty(c, dtype=np.float32)
        for k in range(c):
            less_x = 0
            eq_x = 0
            less_y = 0
            eq_y = 0
            xk = xb[k]
            yk = yb[k]
            for j in range(c):
                xj = xb[j]
                yj = yb[j]
                if xj < xk:
                    less_x += 1
                elif xj == xk:
                    eq_x += 1
                if yj < yk:
                    less_y += 1
                elif yj == yk:
                    eq_y += 1
            xr[k] = (2.0 * less_x + eq_x + 1) / 2.0
            yr[k] = (2.0 * less_y + eq_y + 1) / 2.0

        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for k in range(c):
            xr_k = xr[k]
            yr_k = yr[k]
            sx += xr_k
            sy += yr_k
            sxx += xr_k * xr_k
            syy += yr_k * yr_k
            sxy += xr_k * yr_k
        cn = float(c)
        mx = sx / cn
        my = sy / cn
        vx = sxx - cn * mx * mx
        vy = syy - cn * my * my
        if vx <= 0.0 or vy <= 0.0:
            out[i] = np.nan
        else:
            cov = sxy - cn * mx * my
            out[i] = cov / np.sqrt(vx * vy)

    return out


def roll_rankcorr_fixed(
    xvals: np.ndarray,
    yvals: np.ndarray,
    window: int,
    *,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗滚动 Spearman（秩）相关系数，自动选择 C++ 或 Numba 后端。"""
    if _use_cxx_backend() and _cxx_has("roll_rankcorr_fixed"):
        return np.asarray(_fam_accel.roll_rankcorr_fixed(xvals, yvals, window, _cxx_parallel(parallel)), dtype=np.float32)
    return _roll_rankcorr_numba(xvals, yvals, window)


# =============================================================================
# Rolling mutual information — bivariate, price vs lagged volume (histogram / rank bins)
# =============================================================================


@njit(cache=True)
def _roll_mutual_info_lag_numba(
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
    lag: int,
    n_bins: int,
    min_pairs: int,
) -> np.ndarray:
    """窗内估计 I(close(t); volume(t-lag))：对窗内有效样本分别做秩分箱，再算 Shannon MI（nat）。

    每个时点 t：取 j ∈ [t-window+1, t] 且 j≥lag，配对 (close[j], volume[j-lag])；
    跳过非有限值；有效对数 < min_pairs 输出 NaN。
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    B = n_bins
    if B < 2:
        for i in range(n):
            out[i] = np.nan
        return out

    lag_i = lag
    if lag_i < 0:
        lag_i = 0

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w

        max_c = i - lo + 1
        xb = np.empty(max_c, dtype=np.float32)
        yb = np.empty(max_c, dtype=np.float32)
        c = 0
        for j in range(lo, i + 1):
            if j < lag_i:
                continue
            xv = close[j]
            yv = volume[j - lag_i]
            if xv == xv and yv == yv:
                xb[c] = xv
                yb[c] = yv
                c += 1

        if c < min_pairs:
            out[i] = np.nan
            continue

        bx = np.empty(c, dtype=np.int64)
        by = np.empty(c, dtype=np.int64)

        order_x = np.argsort(xb[:c], kind="mergesort")
        for p in range(c):
            orig = int(order_x[p])
            bb = (p * B) // c
            if bb >= B:
                bb = B - 1
            bx[orig] = bb

        order_y = np.argsort(yb[:c], kind="mergesort")
        for p in range(c):
            orig = int(order_y[p])
            bb = (p * B) // c
            if bb >= B:
                bb = B - 1
            by[orig] = bb

        nb2 = B * B
        cnt = np.zeros(nb2, dtype=np.float32)
        for k in range(c):
            idx = bx[k] * B + by[k]
            cnt[idx] += 1.0

        px = np.zeros(B, dtype=np.float32)
        py = np.zeros(B, dtype=np.float32)
        total = float(c)
        for ix in range(B):
            for iy in range(B):
                pxy = cnt[ix * B + iy] / total
                px[ix] += pxy
                py[iy] += pxy

        eps = 1e-15
        mi = 0.0
        for ix in range(B):
            for iy in range(B):
                pxy = cnt[ix * B + iy] / total
                if pxy > 0.0:
                    mi += pxy * (
                        np.log(pxy + eps)
                        - np.log(px[ix] + eps)
                        - np.log(py[iy] + eps)
                    )
        if mi < 0.0:
            mi = 0.0
        out[i] = mi

    return out


def roll_mutual_info_lag_fixed(
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
    lag: int,
    *,
    n_bins: int = 8,
    min_pairs: Optional[int] = None,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗滚动互信息 close(t) vs volume(t-lag)；秩分箱 + 列联表 MI。

    后端：优先 C++（与 Numba 同一数值语义）；否则 Numba；均无则回落到解释执行的
    ``_roll_mutual_info_lag_numba``（无 Numba 时）。
    """
    if int(n_bins) < 2:
        raise ValueError("n_bins must be >= 2")
    w = max(1, int(window))
    lag_i = max(0, int(lag))
    B = int(n_bins)
    mp = int(min_pairs) if min_pairs is not None else max(B + 2, 8)
    if mp < 2:
        raise ValueError("min_pairs must be >= 2")
    c = np.asarray(close, dtype=np.float32)
    v = np.asarray(volume, dtype=np.float32)
    if c.shape[0] != v.shape[0]:
        raise ValueError("close and volume must have the same length")
    if _use_cxx_backend() and _cxx_has("roll_mutual_info_lag_fixed"):
        return np.asarray(
            _fam_accel.roll_mutual_info_lag_fixed(
                c, v, w, lag_i, B, mp, _cxx_parallel(parallel)
            ),
            dtype=np.float32,
        )
    # Numba / 纯 Python：未对 parallel 做 OpenMP；与 historical 行为一致
    return _roll_mutual_info_lag_numba(c, v, w, lag_i, B, mp)


# =============================================================================
# Kaufman Efficiency Ratio — univariate, fixed window
# =============================================================================


@njit(cache=True)
def _efficiency_ratio_segment(vals: np.ndarray, lo: int, i: int) -> float:
    """窗口 [lo, i] 内 Kaufman ER；无效时返回 NaN。"""
    first_valid = -1
    last_valid = -1
    for j in range(lo, i + 1):
        v = vals[j]
        if v == v:
            if first_valid < 0:
                first_valid = j
            last_valid = j

    if first_valid < 0 or last_valid == first_valid:
        return np.nan

    total_path = 0.0
    prev = vals[first_valid]
    for j in range(first_valid + 1, last_valid + 1):
        v = vals[j]
        if v == v:
            d = v - prev
            if d < 0.0:
                d = -d
            total_path += d
            prev = v

    if total_path == 0.0:
        return np.nan

    net = vals[last_valid] - vals[first_valid]
    if net < 0.0:
        net = -net
    return net / total_path


@njit(cache=True)
def _roll_efficiency_ratio_numba(vals: np.ndarray, window: int) -> np.ndarray:
    """ER = |窗口首末价差| / 窗口内逐 bar 绝对变化之和；固定窗。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        w = window
        if w < 2:
            w = 2
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        er = _efficiency_ratio_segment(vals, lo, i)
        out[i] = er if er == er else np.nan

    return out


@njit(cache=True)
def _roll_efficiency_ratio_dynamic_numba(vals: np.ndarray, wvals: np.ndarray) -> np.ndarray:
    """逐 bar 窗长的 Kaufman ER；``wvals[i]`` 为 bar i 处回看长度（≥2）。"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    if wvals.shape[0] != n:
        for i in range(n):
            out[i] = np.nan
        return out

    for i in range(n):
        w = int(wvals[i])
        if w < 2:
            w = 2
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        er = _efficiency_ratio_segment(vals, lo, i)
        out[i] = er if er == er else np.nan

    return out


def roll_efficiency_ratio_fixed(
    vals: np.ndarray,
    window: int,
    *,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗 Kaufman Efficiency Ratio，自动选择 C++ 或 Numba 后端。"""
    if int(window) < 2:
        raise ValueError("window must be >= 2 for efficiency ratio")
    if _use_cxx_backend() and _cxx_has("roll_efficiency_ratio_fixed"):
        return np.asarray(_fam_accel.roll_efficiency_ratio_fixed(vals, window, _cxx_parallel(parallel)), dtype=np.float32)
    return _roll_efficiency_ratio_numba(vals, int(window))


def roll_efficiency_ratio_dynamic(vals: np.ndarray, wvals: np.ndarray) -> np.ndarray:
    """动态窗 Kaufman ER；``wvals`` 与 ``vals`` 等长，逐 bar 窗长（由 ``dynamic_window_int_series`` 预处理）。"""
    w_arr = np.asarray(wvals, dtype=np.int64).ravel()
    if w_arr.shape[0] != np.asarray(vals).shape[0]:
        raise ValueError("dynamic window length must match values length")
    return _roll_efficiency_ratio_dynamic_numba(
        np.asarray(vals, dtype=np.float32).ravel(),
        w_arr,
    )


# Wick efficiency — lagged upper/lower wick coupling (cross-time, non DELTA-able)
# =============================================================================


@njit(cache=True)
def _wick_efficiency_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lag: int,
    eps: float,
) -> np.ndarray:
    """影线能量效率：当前上影线 × (t-k) 下影线 / (当期实体 × 前期实体 + eps)。

    UP_WICK(t) = high - max(open, close)
    DN_WICK(t) = min(open, close) - low
    BODY(t) = |close - open|
    OUT[t] = UP_WICK(t) * DN_WICK(t-k) / (BODY(t) * BODY(t-k) + eps)

    i < lag 或与 t、t-k 任一侧 OHLC 非有限时为 NaN。lag >= 1。
    """
    n = open_.shape[0]
    kk = int(lag)
    out = np.empty(n, dtype=np.float32)
    if kk < 1:
        for i in range(n):
            out[i] = np.nan
        return out

    for i in range(n):
        if i < kk:
            out[i] = np.nan
            continue
        j = i - kk
        ot = open_[i]
        ht = high[i]
        lt = low[i]
        ct = close[i]
        oj = open_[j]
        hj = high[j]
        lj = low[j]
        cj = close[j]
        if not (ot == ot and ht == ht and lt == lt and ct == ct):
            out[i] = np.nan
            continue
        if not (oj == oj and hj == hj and lj == lj and cj == cj):
            out[i] = np.nan
            continue

        oc_max_t = ot if ot >= ct else ct
        up_wick = ht - oc_max_t
        oc_min_j = oj if oj <= cj else cj
        dn_wick_j = oc_min_j - lj

        body_t = ct - ot
        if body_t < 0.0:
            body_t = -body_t
        body_j = cj - oj
        if body_j < 0.0:
            body_j = -body_j

        denom = body_t * body_j + eps
        out[i] = (up_wick * dn_wick_j) / denom

    return out


def wick_efficiency_fixed(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lag: int,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """WICK_EFFICIENCY：四维 OHLC + 滞后 k。"""
    n = open_.shape[0]
    if not (high.shape[0] == n and low.shape[0] == n and close.shape[0] == n):
        raise ValueError("open, high, low, close must have same length")
    if int(lag) < 1:
        raise ValueError("lag must be >= 1 for WICK_EFFICIENCY")
    eps_f = float(eps)
    if _use_cxx_backend() and _cxx_has("wick_efficiency_fixed"):
        return np.asarray(_fam_accel.wick_efficiency_fixed(open_, high, low, close, int(lag), eps_f), dtype=np.float32)
    return _wick_efficiency_numba(open_, high, low, close, int(lag), eps_f)


# KLINE_GEOMETRY — OHLC 矩阵行 SVD 奇异值比 σ₂/σ₁（形态混乱度）
# =============================================================================


def roll_kline_geometry(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    window: Union[int, float, np.ndarray],
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    """固定窗或逐 bar 窗长：窗口内 ``X∈R^{k×4}``（每行 o,h,l,c），``σ₂/σ₁``，∈ [0,1]。

    需要有效窗长 ``≥ 2`` 且窗内 OHLC 全有限；``σ₁≤eps`` 时为 NaN。使用 ``numpy.linalg.svd``
    （``full_matrices=False``），无 C++/Numba 路径。

    ``window`` 可为标量（``int`` / ``float`` / numpy 标量）或与 ``o`` 等长的整数向量。
    """
    if o.shape[0] != h.shape[0] or o.shape[0] != l.shape[0] or o.shape[0] != c.shape[0]:
        raise ValueError("open, high, low, close must have the same length")

    n = int(o.shape[0])
    out = np.full(n, np.nan, dtype=np.float32)
    eps_f = float(eps)

    wa = np.asarray(window)
    if wa.ndim == 0:
        w0 = int(wa)
        if w0 < 2:
            raise ValueError("window must be >= 2")
        for i in range(n):
            wi = w0 if w0 <= i + 1 else i + 1
            if wi < 2:
                continue
            lo = i + 1 - wi
            X = np.empty((wi, 4), dtype=np.float32)
            X[:, 0] = o[lo : i + 1]
            X[:, 1] = h[lo : i + 1]
            X[:, 2] = l[lo : i + 1]
            X[:, 3] = c[lo : i + 1]
            if not np.isfinite(X).all():
                continue
            try:
                _, s, _ = np.linalg.svd(X, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            if s.size < 2 or float(s[0]) <= eps_f:
                continue
            out[i] = float(s[1]) / float(s[0])
        return out

    w_arr = np.maximum(wa.astype(np.int64, copy=False).ravel(), 1)
    if w_arr.shape[0] != n:
        raise ValueError("dynamic window length must match OHLC length")
    for i in range(n):
        wi = int(w_arr[i])
        if wi > i + 1:
            wi = i + 1
        if wi < 2:
            continue
        lo = i + 1 - wi
        X = np.empty((wi, 4), dtype=np.float32)
        X[:, 0] = o[lo : i + 1]
        X[:, 1] = h[lo : i + 1]
        X[:, 2] = l[lo : i + 1]
        X[:, 3] = c[lo : i + 1]
        if not np.isfinite(X).all():
            continue
        try:
            _, s, _ = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if s.size < 2 or float(s[0]) <= eps_f:
            continue
        out[i] = float(s[1]) / float(s[0])

    return out


def roll_kline_geometry_fixed(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    window: int,
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    """``roll_kline_geometry(..., window=int, ...)`` 别名。"""
    return roll_kline_geometry(o, h, l, c, int(window), eps=eps)


@njit(cache=True)
def _roll_permutation_entropy_numba(
    vals: np.ndarray, window: int, order: int
) -> np.ndarray:
    """Bandt-Pompe 排列熵：窗口内所有长度 order 的子序列按序数排列模式计数，
    再按 Shannon 熵归一化到 [0,1]（除以 log(order!)）。
    子序列内含 NaN 则该模式不计；总数为 0 输出 NaN。"""
    n = vals.shape[0]
    m = order
    out = np.empty(n, dtype=np.float32)

    # log(order!) for m in [2,7]
    if m == 2:
        norm = 0.6931471805599453
    elif m == 3:
        norm = 1.791759469228055
    elif m == 4:
        norm = 3.1780538303479458
    elif m == 5:
        norm = 4.787491742782046
    elif m == 6:
        norm = 6.579251212010101
    else:  # m == 7
        norm = 8.525161361065414

    pow_size = 1
    for _ in range(m):
        pow_size *= m

    ranks = np.empty(m, dtype=np.int64)

    for i in range(n):
        w = window
        if w < m:
            w = m
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w
        wlen = i - lo + 1

        if wlen < m:
            out[i] = np.nan
            continue

        counts = np.zeros(pow_size, dtype=np.int64)
        total = 0

        for start in range(lo, i - m + 2):
            valid = True
            for k in range(m):
                v = vals[start + k]
                if not (v == v):
                    valid = False
                    break
            if not valid:
                continue

            for k in range(m):
                r = 0
                vk = vals[start + k]
                for j in range(m):
                    if j == k:
                        continue
                    vj = vals[start + j]
                    if vj < vk or (vj == vk and j < k):
                        r += 1
                ranks[k] = r

            pid = 0
            for k in range(m):
                pid = pid * m + ranks[k]
            counts[pid] += 1
            total += 1

        if total == 0:
            out[i] = np.nan
            continue

        H = 0.0
        inv_tot = 1.0 / float(total)
        for k in range(pow_size):
            c_k = counts[k]
            if c_k > 0:
                p = c_k * inv_tot
                H -= p * np.log(p)
        out[i] = H / norm

    return out


def roll_permutation_entropy_fixed(
    vals: np.ndarray,
    window: int,
    order: int = 3,
    *,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定窗 Bandt-Pompe 排列熵，自动选择 C++ 或 Numba 后端。``order`` 典型 3–5。"""
    if int(window) < 2:
        raise ValueError("window must be >= 2")
    if int(order) < 2 or int(order) > 7:
        raise ValueError("order must be in [2, 7]")
    if _use_cxx_backend() and _cxx_has("roll_permutation_entropy_fixed"):
        return np.asarray(_fam_accel.roll_permutation_entropy_fixed(
            vals, int(window), int(order), _cxx_parallel(parallel)
        ), dtype=np.float32)
    return _roll_permutation_entropy_numba(vals, int(window), int(order))


# =============================================================================
# Chip distribution metrics (daily: uniform / cyq / triangular)
# =============================================================================

_CHIP_OP = _chip_daily.CHIP_OP
chip_wass_implementation_id = _chip_daily.chip_wass_implementation_id
chip_peak_sharpness_impl_id = _chip_daily.chip_peak_sharpness_impl_id
chip_bimodal_impl_id = _chip_daily.chip_bimodal_impl_id


def roll_chip_metric_fixed(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    op: str,
    method: str = "cyq",
    *,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """日频筹码密度指标（uniform / cyq / triangular）。"""
    del parallel
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    if int(nbins) < 2:
        raise ValueError("nbins must be >= 2")
    return _chip_daily.roll_chip_metric_daily(
        close, volume, low, high, aux, int(window), int(nbins), op, method
    )


def _broadcast_chip_win_vec(x: Union[int, np.ndarray], n: int, *, floor: int) -> np.ndarray:
    """标量或可迭代 → 长度 ``n`` 的 int64 向量。"""
    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    if arr.size == 1:
        v = max(floor, int(arr[0]))
        return np.full(n, v, dtype=np.int64)
    if arr.size != n:
        raise ValueError(f"window vector length mismatch: expected {n}, got {arr.size}")
    return np.maximum(arr, floor).astype(np.int64, copy=False)


def roll_chip_wass_dist(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    wa: Union[int, np.ndarray],
    wb: Union[int, np.ndarray],
    rho: Union[int, np.ndarray],
    nbins: int,
    implementation: str = "moment",
    method: str = "cyq",
    *,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """日频筹码双窗漂移。"""
    del parallel
    nb = int(nbins)
    if nb < 2:
        raise ValueError("nbins must be >= 2")
    n = int(close.shape[0])
    for arr in (volume, low, high, aux):
        if arr.shape[0] != n:
            raise ValueError("chip wass arrays must have the same length")
    wa_v = _broadcast_chip_win_vec(wa, n, floor=1)
    wb_v = _broadcast_chip_win_vec(wb, n, floor=1)
    rho_v = _broadcast_chip_win_vec(rho, n, floor=0)
    return _chip_daily.roll_chip_wass_dist_daily(
        close, volume, low, high, aux, wa_v, wb_v, rho_v, nb, implementation, method
    )


def roll_chip_wass_dist_fixed(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    lag: int = 0,
    *,
    implementation: str = "moment",
    method: str = "cyq",
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """固定整窗双窗漂移。"""
    del parallel
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    if int(nbins) < 2:
        raise ValueError("nbins must be >= 2")
    if int(lag) < 0:
        raise ValueError("lag must be >= 0")
    w = int(window)
    return roll_chip_wass_dist(
        close, volume, low, high, aux, w, w, int(lag), int(nbins), implementation, method
    )


def roll_chip_peak_sharpness_fixed(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    implementation: str = "curvature",
    method: str = "cyq",
    *,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """日频主峰尖锐度。"""
    del parallel
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    if int(nbins) < 2:
        raise ValueError("nbins must be >= 2")
    return _chip_daily.roll_chip_peak_sharpness_daily(
        close, volume, low, high, aux, int(window), int(nbins), implementation, method
    )


def roll_chip_bimodal_fixed(
    close: np.ndarray,
    volume: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aux: np.ndarray,
    window: int,
    nbins: int,
    implementation: str = "simple",
    method: str = "cyq",
    *,
    lambda_scale: float = 1.0,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """日频双峰结构得分。"""
    del parallel
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    if int(nbins) < 2:
        raise ValueError("nbins must be >= 2")
    return _chip_daily.roll_chip_bimodal_daily(
        close,
        volume,
        low,
        high,
        aux,
        int(window),
        int(nbins),
        implementation,
        method,
        lambda_scale=lambda_scale,
    )


# VOLUME_CLOCK_VPIN — Volume-Synchronized PIN
# =============================================================================


def vpin_classification_id(name: str) -> int:
    """``tick``（默认，平盘沿用上一方向）/ ``lee_ready``（平盘 50/50 分拆）。"""
    k = str(name).strip().lower().replace("-", "_")
    if k == "tick":
        return 0
    if k in ("lee_ready", "leeready", "lr"):
        return 1
    raise ValueError('classification must be "tick" or "lee_ready"')


@njit(cache=True)
def _vpin_classify_volume(
    price: float,
    price_prev: float,
    vol: float,
    cls_id: int,
    last_sign: int,
) -> tuple:
    """返回 (buy_vol, sell_vol, new_last_sign)。"""
    if not (price == price) or not (vol == vol) or vol <= 0.0:
        return 0.0, 0.0, last_sign
    if not (price_prev == price_prev):
        return 0.0, 0.0, last_sign

    if price > price_prev:
        return vol, 0.0, 1
    if price < price_prev:
        return 0.0, vol, -1

    if cls_id == 0:
        if last_sign > 0:
            return vol, 0.0, last_sign
        if last_sign < 0:
            return 0.0, vol, last_sign
        half = 0.5 * vol
        return half, half, last_sign

    half = 0.5 * vol
    return half, half, last_sign


@njit(cache=True)
def _vpin_push_imbalance(
    imb_buf: np.ndarray,
    n_in_buf: int,
    window: int,
    bucket_buy: float,
    bucket_sell: float,
    eps: float,
) -> int:
    total = bucket_buy + bucket_sell
    if total <= eps:
        return n_in_buf
    imb = abs(bucket_buy - bucket_sell) / (total + eps)
    if imb > 1.0:
        imb = 1.0
    elif imb < 0.0:
        imb = 0.0
    w = window
    if n_in_buf < w:
        imb_buf[n_in_buf] = imb
        return n_in_buf + 1
    for k in range(w - 1):
        imb_buf[k] = imb_buf[k + 1]
    imb_buf[w - 1] = imb
    return w


@njit(cache=True)
def _vpin_buf_mean(
    imb_buf: np.ndarray, n_in_buf: int, window: int, min_buckets: int
) -> float:
    """滚动桶 imbalance 均值；``min_buckets`` 为最少已满桶数，不足则 NaN。"""
    mb = min_buckets
    if mb < 1:
        mb = 1
    if mb > window:
        mb = window
    if n_in_buf < mb:
        return np.nan
    s = 0.0
    for k in range(n_in_buf):
        s += imb_buf[k]
    return s / float(n_in_buf)


@njit(cache=True)
def _vpin_add_to_bucket(
    rem_buy: float,
    rem_sell: float,
    bucket_buy: float,
    bucket_sell: float,
    bucket_fill: float,
    bsize: float,
    imb_buf: np.ndarray,
    n_in_buf: int,
    window: int,
    eps: float,
) -> tuple:
    """将剩余买卖量灌入当前桶；满桶则结算。"""
    while rem_buy + rem_sell > eps:
        space = bsize - bucket_fill
        if space <= eps:
            n_in_buf = _vpin_push_imbalance(imb_buf, n_in_buf, window, bucket_buy, bucket_sell, eps)
            bucket_buy = 0.0
            bucket_sell = 0.0
            bucket_fill = 0.0
            continue

        chunk = rem_buy + rem_sell
        if chunk <= space + eps:
            bucket_buy += rem_buy
            bucket_sell += rem_sell
            bucket_fill += chunk
            rem_buy = 0.0
            rem_sell = 0.0
            if bucket_fill >= bsize - eps:
                n_in_buf = _vpin_push_imbalance(imb_buf, n_in_buf, window, bucket_buy, bucket_sell, eps)
                bucket_buy = 0.0
                bucket_sell = 0.0
                bucket_fill = 0.0
        else:
            ratio = space / chunk
            take_buy = rem_buy * ratio
            take_sell = rem_sell * ratio
            bucket_buy += take_buy
            bucket_sell += take_sell
            bucket_fill += space
            rem_buy -= take_buy
            rem_sell -= take_sell
            n_in_buf = _vpin_push_imbalance(imb_buf, n_in_buf, window, bucket_buy, bucket_sell, eps)
            bucket_buy = 0.0
            bucket_sell = 0.0
            bucket_fill = 0.0
    return rem_buy, rem_sell, bucket_buy, bucket_sell, bucket_fill, n_in_buf


@njit(cache=True)
def _volume_clock_vpin_numba(
    price: np.ndarray,
    volume: np.ndarray,
    window: int,
    bucket_size: float,
    cls_id: int,
    eps: float,
    min_buckets: int,
) -> np.ndarray:
    """成交量同步 VPIN：固定成交量桶 + 买卖分类，滚动桶 imbalance 均值。"""
    n = price.shape[0]
    out = np.empty(n, dtype=np.float32)
    w = window
    if w < 1:
        w = 1
    bsize = bucket_size
    if bsize <= eps:
        for i in range(n):
            out[i] = np.nan
        return out

    imb_buf = np.zeros(w, dtype=np.float32)
    n_in_buf = 0
    bucket_buy = 0.0
    bucket_sell = 0.0
    bucket_fill = 0.0
    last_sign = 0

    for i in range(n):
        p = price[i]
        p_prev = price[i - 1] if i > 0 else np.nan
        buy_v, sell_v, last_sign = _vpin_classify_volume(p, p_prev, volume[i], cls_id, last_sign)

        rem_buy = buy_v
        rem_sell = sell_v
        rem_buy, rem_sell, bucket_buy, bucket_sell, bucket_fill, n_in_buf = _vpin_add_to_bucket(
            rem_buy,
            rem_sell,
            bucket_buy,
            bucket_sell,
            bucket_fill,
            bsize,
            imb_buf,
            n_in_buf,
            w,
            eps,
        )

        out[i] = _vpin_buf_mean(imb_buf, n_in_buf, w, min_buckets)

    return out


def volume_clock_vpin_fixed(
    price: np.ndarray,
    volume: np.ndarray,
    window: int,
    bucket_size: float,
    classification: str = "tick",
    *,
    min_buckets: int = 5,
    eps: float = 1e-12,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """Volume-Synchronized PIN ∈ [0,1]；``window`` 为成交量桶个数（非 bar 数）。"""
    del parallel
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    if float(bucket_size) <= 0.0:
        raise ValueError("bucket_size must be > 0")
    if price.shape[0] != volume.shape[0]:
        raise ValueError("price and volume must have the same length")
    cls = vpin_classification_id(classification)
    mb = int(min_buckets)
    if mb < 1:
        raise ValueError("min_buckets must be >= 1")
    return _volume_clock_vpin_numba(
        price.astype(np.float32, copy=False),
        volume.astype(np.float32, copy=False),
        int(window),
        float(bucket_size),
        cls,
        float(eps),
        mb,
    )


# Generalized crowding (CROWD_*) — rolling dimension-bucket statistics
# =============================================================================

# op: 0=share, 1=mean_ratio, 2=contrast, 3=rank_weighted
# bucket_mode: 0=quantile (side high/low + split), 1=equal_freq (n_buckets + bucket_idx)
# side_high: 1=high tail, 0=low tail (quantile mode only)


@njit(cache=True)
def _crowd_sorted_quantile(buf: np.ndarray, c: int, q: float) -> float:
    if c <= 0:
        return np.nan
    k = float(c)
    pos = q * (k - 1.0)
    if pos < 0.0:
        pos = 0.0
    if pos > k - 1.0:
        pos = k - 1.0
    lo_i = int(np.floor(pos))
    hi_i = int(np.ceil(pos))
    if lo_i == hi_i:
        return buf[lo_i]
    frac = pos - float(lo_i)
    return buf[lo_i] * (1.0 - frac) + buf[hi_i] * frac


@njit(cache=True)
def _crowd_avg_rank(vals: np.ndarray, c: int, ranks: np.ndarray) -> None:
    for k in range(c):
        vk = vals[k]
        less = 0
        eq = 0
        for j in range(c):
            vj = vals[j]
            if vj < vk:
                less += 1
            elif vj == vk:
                eq += 1
        ranks[k] = (2.0 * less + eq + 1) / 2.0


@njit(cache=True)
def _crowd_in_target_quantile(
    dim_val: float,
    dim_buf: np.ndarray,
    c: int,
    split_q: float,
    side_high: int,
) -> bool:
    tmp = np.empty(c, dtype=np.float32)
    for k in range(c):
        tmp[k] = dim_buf[k]
    tmp_sorted = np.sort(tmp)
    thr = _crowd_sorted_quantile(tmp_sorted, c, split_q)
    if side_high == 1:
        return dim_val >= thr
    return dim_val < thr


@njit(cache=True)
def _crowd_equal_freq_bucket(rank: float, n: int, n_buckets: int) -> int:
    b = int((rank - 1.0) / float(n) * float(n_buckets))
    if b < 0:
        b = 0
    if b >= n_buckets:
        b = n_buckets - 1
    return b


@njit(cache=True)
def _crowd_in_target_equal_freq(
    dim_val: float,
    dim_buf: np.ndarray,
    c: int,
    n_buckets: int,
    bucket_idx0: int,
) -> bool:
    ranks = np.empty(c, dtype=np.float32)
    _crowd_avg_rank(dim_buf, c, ranks)
    for k in range(c):
        if dim_buf[k] == dim_val:
            b = _crowd_equal_freq_bucket(ranks[k], c, n_buckets)
            if b == bucket_idx0:
                return True
    return False


@njit(cache=True)
def _crowd_is_target(
    dim_val: float,
    dim_buf: np.ndarray,
    c: int,
    bucket_mode: int,
    split_q: float,
    side_high: int,
    n_buckets: int,
    bucket_idx0: int,
) -> bool:
    if bucket_mode == 0:
        return _crowd_in_target_quantile(dim_val, dim_buf, c, split_q, side_high)
    return _crowd_in_target_equal_freq(dim_val, dim_buf, c, n_buckets, bucket_idx0)


@njit(cache=True)
def _crowd_mark_target(
    dim_buf: np.ndarray,
    c: int,
    in_target: np.ndarray,
    bucket_mode: int,
    split_q: float,
    side_high: int,
    n_buckets: int,
    bucket_idx0: int,
) -> None:
    """窗口内一次标记目标桶（避免 share/mean_ratio 内重复 sort）。"""
    if bucket_mode == 0:
        tmp = np.empty(c, dtype=np.float32)
        for k in range(c):
            tmp[k] = dim_buf[k]
        thr = _crowd_sorted_quantile(np.sort(tmp), c, split_q)
        for k in range(c):
            if side_high == 1:
                in_target[k] = dim_buf[k] >= thr
            else:
                in_target[k] = dim_buf[k] < thr
    else:
        ranks = np.empty(c, dtype=np.float32)
        _crowd_avg_rank(dim_buf, c, ranks)
        for k in range(c):
            b = _crowd_equal_freq_bucket(ranks[k], c, n_buckets)
            in_target[k] = b == bucket_idx0


@njit(cache=True)
def _roll_crowd_numba(
    dim: np.ndarray,
    attr: np.ndarray,
    weight: np.ndarray,
    window: int,
    op: int,
    bucket_mode: int,
    split_q: float,
    side_high: int,
    n_buckets: int,
    bucket_idx0: int,
    min_valid: int,
    use_attr: int,
    use_weight: int,
) -> np.ndarray:
    n = dim.shape[0]
    out = np.empty(n, dtype=np.float32)
    eps = 1e-12

    for i in range(n):
        w = window
        if w < 1:
            w = 1
        if w > i + 1:
            w = i + 1
        lo = i + 1 - w

        wlen = i - lo + 1
        dim_buf = np.empty(wlen, dtype=np.float32)
        attr_buf = np.empty(wlen, dtype=np.float32)
        wgt_buf = np.empty(wlen, dtype=np.float32)
        c = 0
        for j in range(lo, i + 1):
            d = dim[j]
            if d != d:
                continue
            if use_attr == 1:
                a = attr[j]
                if a != a:
                    continue
            else:
                a = 0.0
            if use_weight == 1:
                wg = weight[j]
                if wg != wg or wg <= 0.0:
                    continue
            else:
                wg = 1.0
            dim_buf[c] = d
            attr_buf[c] = a
            wgt_buf[c] = wg
            c += 1

        if c < min_valid:
            out[i] = np.nan
            continue

        if op == 3:
            if c < 2:
                out[i] = np.nan
                continue
            ranks = np.empty(c, dtype=np.float32)
            _crowd_avg_rank(dim_buf, c, ranks)
            denom = 0.0
            numer = 0.0
            for k in range(c):
                denom += wgt_buf[k]
            if denom <= eps:
                out[i] = np.nan
                continue
            inv_nm1 = 1.0 / float(c - 1)
            for k in range(c):
                rn = (ranks[k] - 1.0) * inv_nm1
                numer += rn * attr_buf[k] * wgt_buf[k]
            out[i] = numer / denom
            continue

        if op == 2:
            tmp = np.empty(c, dtype=np.float32)
            for k in range(c):
                tmp[k] = dim_buf[k]
            tmp_sorted = np.sort(tmp)
            thr = _crowd_sorted_quantile(tmp_sorted, c, split_q)
            sum_h = 0.0
            cnt_h = 0
            sum_l = 0.0
            cnt_l = 0
            for k in range(c):
                if dim_buf[k] >= thr:
                    sum_h += attr_buf[k]
                    cnt_h += 1
                else:
                    sum_l += attr_buf[k]
                    cnt_l += 1
            if cnt_h == 0 or cnt_l == 0:
                out[i] = np.nan
            else:
                out[i] = (sum_h / float(cnt_h)) - (sum_l / float(cnt_l))
            continue

        in_tgt = np.empty(c, dtype=np.int8)
        _crowd_mark_target(
            dim_buf,
            c,
            in_tgt,
            bucket_mode,
            split_q,
            side_high,
            n_buckets,
            bucket_idx0,
        )

        if op == 0:
            sum_all = 0.0
            sum_tgt = 0.0
            for k in range(c):
                sum_all += wgt_buf[k]
                if in_tgt[k]:
                    sum_tgt += wgt_buf[k]
            if sum_all <= eps or sum_tgt <= eps:
                out[i] = np.nan
            else:
                out[i] = sum_tgt / sum_all
            continue

        # mean_ratio
        sum_all_a = 0.0
        sum_tgt_a = 0.0
        cnt_all = 0
        cnt_tgt = 0
        for k in range(c):
            sum_all_a += attr_buf[k]
            cnt_all += 1
            if in_tgt[k]:
                sum_tgt_a += attr_buf[k]
                cnt_tgt += 1
        if cnt_all == 0 or cnt_tgt == 0:
            out[i] = np.nan
            continue
        mean_all = sum_all_a / float(cnt_all)
        mean_tgt = sum_tgt_a / float(cnt_tgt)
        if abs(mean_all) <= eps:
            out[i] = np.nan
        else:
            out[i] = mean_tgt / mean_all

    return out


def crowd_op_id(name: str) -> int:
    m = {
        "share": 0,
        "mean_ratio": 1,
        "contrast": 2,
        "rank_weighted": 3,
    }
    key = str(name).strip().lower()
    if key not in m:
        raise ValueError(f"unknown crowd op: {name!r}")
    return m[key]


def crowd_side_id(side: str) -> int:
    s = str(side).strip().lower()
    if s == "high":
        return 1
    if s == "low":
        return 0
    raise ValueError(f"side must be 'high' or 'low', got {side!r}")


def roll_crowd_fixed(
    dim: np.ndarray,
    attr: np.ndarray,
    weight: np.ndarray,
    window: int,
    op: str,
    *,
    bucket_mode: str = "quantile",
    side: str = "high",
    split: float = 0.5,
    n_buckets: int = 2,
    bucket_idx: int = 1,
    min_valid: int = 0,
    use_attr: bool = True,
    use_weight: bool = True,
    parallel: Optional[bool] = None,
) -> np.ndarray:
    """滚动广义拥挤度统计（Numba；无前视）。"""
    del parallel
    w = int(window)
    if w < 1:
        raise ValueError("window must be >= 1")
    op_id = crowd_op_id(op)
    mv = int(min_valid)
    if mv < 1:
        mv = max(3, w // 4)
    sq = float(split)
    if not (0.0 < sq < 1.0):
        raise ValueError("split must be in (0, 1)")
    bm = 0 if str(bucket_mode).strip().lower() == "quantile" else 1
    nb = int(n_buckets)
    bidx0 = int(bucket_idx) - 1
    if bm == 1:
        if nb < 2:
            raise ValueError("n_buckets must be >= 2")
        if bidx0 < 0 or bidx0 >= nb:
            raise ValueError("bucket_idx must be in [1, n_buckets]")
    sh = crowd_side_id(side)
    d = dim.astype(np.float32, copy=False)
    a = attr.astype(np.float32, copy=False) if use_attr else np.zeros_like(d)
    wg = weight.astype(np.float32, copy=False) if use_weight else np.ones_like(d)
    if not (d.shape[0] == a.shape[0] == wg.shape[0]):
        raise ValueError("dim, attr, weight must have the same length")
    return _roll_crowd_numba(
        d,
        a,
        wg,
        w,
        op_id,
        bm,
        sq,
        sh,
        nb,
        bidx0,
        mv,
        1 if use_attr else 0,
        1 if use_weight else 0,
    )
