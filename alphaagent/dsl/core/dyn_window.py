"""动态窗滚动、滞后与 ARG 极值加速：按品种分组算时间序列 → ``get_indexer`` 写回 → 可选 Numba(njit/prange)，否则 NumPy。"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .accel import (
    _CXX_ROLL_FIXED_MAX_OP,
    _use_cxx_backend,
)

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


def _use_cxx_for_dynamic_kernels() -> bool:
    """与 accel_backend 相同策略：可用且未强制 numba 时用 C++。"""
    return _use_cxx_backend() and _accel_backend() is not None


def _accel_backend():
    from . import accel

    return accel._fam_accel


def _numba_parallel_mode() -> Optional[bool]:
    """None = 按行数阈值自动；False = 强制串行内核；True = 强制并行内核。"""
    v = os.environ.get("FUTURE_ALPHA_MINER_NUMBA_PARALLEL", "").strip().lower()
    if v in ("", "auto"):
        return None
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on", "force"):
        return True
    return None


def _numba_parallel_min_rows() -> int:
    s = os.environ.get("FUTURE_ALPHA_MINER_NUMBA_PARALLEL_MIN_ROWS", "").strip()
    if not s:
        return 4096
    try:
        return max(1, int(s))
    except ValueError:
        return 4096


def _use_numba_parallel_for_length(n: int) -> bool:
    if not _HAS_NUMBA or n < 1:
        return False
    mode = _numba_parallel_mode()
    if mode is False:
        return False
    if mode is True:
        return True
    return n >= _numba_parallel_min_rows()


# --- Numba：变长窗口 min / max / sum / mean ---------------------------------


@njit(cache=True)
def _roll_dyn_mm_sm_numba(vals: np.ndarray, wvals: np.ndarray, op: int) -> np.ndarray:
    """op: 0=min, 1=max, 2=sum, 3=mean"""
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        wi = wvals[i]
        if wi < 1:
            wi = 1
        if wi > i + 1:
            wi = i + 1
        lo = i + 1 - wi
        if op == 0:
            m = np.inf
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    if v < m:
                        m = v
            out[i] = m if m != np.inf else np.nan
        elif op == 1:
            m = -np.inf
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    if v > m:
                        m = v
            out[i] = m if m != -np.inf else np.nan
        elif op == 2:
            s = 0.0
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    s += v
                    c += 1
            out[i] = s if c > 0 else np.nan
        else:
            s = 0.0
            c = 0
            for j in range(lo, i + 1):
                v = vals[j]
                if v == v:
                    s += v
                    c += 1
            out[i] = (s / c) if c > 0 else np.nan
    return out


if _HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _roll_dyn_mm_sm_numba_parallel(vals: np.ndarray, wvals: np.ndarray, op: int) -> np.ndarray:
        """与 _roll_dyn_mm_sm_numba 相同，外层 ``i`` 用 ``prange`` 并行。"""
        n = vals.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in prange(n):
            wi = wvals[i]
            if wi < 1:
                wi = 1
            if wi > i + 1:
                wi = i + 1
            lo = i + 1 - wi
            if op == 0:
                m = np.inf
                for j in range(lo, i + 1):
                    v = vals[j]
                    if v == v:
                        if v < m:
                            m = v
                out[i] = m if m != np.inf else np.nan
            elif op == 1:
                m = -np.inf
                for j in range(lo, i + 1):
                    v = vals[j]
                    if v == v:
                        if v > m:
                            m = v
                out[i] = m if m != -np.inf else np.nan
            elif op == 2:
                s = 0.0
                c = 0
                for j in range(lo, i + 1):
                    v = vals[j]
                    if v == v:
                        s += v
                        c += 1
                out[i] = s if c > 0 else np.nan
            else:
                s = 0.0
                c = 0
                for j in range(lo, i + 1):
                    v = vals[j]
                    if v == v:
                        s += v
                        c += 1
                out[i] = (s / c) if c > 0 else np.nan
        return out


@njit(cache=True)
def _delay_dyn_numba(vals: np.ndarray, lags: np.ndarray) -> np.ndarray:
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        k = lags[i]
        if k < 0:
            out[i] = np.nan
            continue
        j = i - k
        if j >= 0:
            out[i] = vals[j]
        else:
            out[i] = np.nan
    return out


if _HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _delay_dyn_numba_parallel(vals: np.ndarray, lags: np.ndarray) -> np.ndarray:
        n = vals.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in prange(n):
            k = lags[i]
            if k < 0:
                out[i] = np.nan
                continue
            j = i - k
            if j >= 0:
                out[i] = vals[j]
            else:
                out[i] = np.nan
        return out


@njit(cache=True)
def _arg_extreme_dyn_numba(vals: np.ndarray, wvals: np.ndarray, want_max: int) -> np.ndarray:
    n = vals.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        wi = wvals[i]
        if wi < 1:
            wi = 1
        if wi > i + 1:
            wi = i + 1
        lo = i + 1 - wi
        if want_max:
            best_j = lo
            best_v = vals[lo]
            for j in range(lo + 1, i + 1):
                v = vals[j]
                if v == v and (not (best_v == best_v) or v > best_v):
                    best_v = v
                    best_j = j
            if best_v == best_v:
                out[i] = float((i - best_j))
            else:
                out[i] = np.nan
        else:
            best_j = lo
            best_v = vals[lo]
            for j in range(lo + 1, i + 1):
                v = vals[j]
                if v == v and (not (best_v == best_v) or v < best_v):
                    best_v = v
                    best_j = j
            if best_v == best_v:
                out[i] = float((i - best_j))
            else:
                out[i] = np.nan
    return out


if _HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _arg_extreme_dyn_numba_parallel(
        vals: np.ndarray, wvals: np.ndarray, want_max: int
    ) -> np.ndarray:
        n = vals.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in prange(n):
            wi = wvals[i]
            if wi < 1:
                wi = 1
            if wi > i + 1:
                wi = i + 1
            lo = i + 1 - wi
            if want_max:
                best_j = lo
                best_v = vals[lo]
                for j in range(lo + 1, i + 1):
                    v = vals[j]
                    if v == v and (not (best_v == best_v) or v > best_v):
                        best_v = v
                        best_j = j
                if best_v == best_v:
                    out[i] = float((i - best_j))
                else:
                    out[i] = np.nan
            else:
                best_j = lo
                best_v = vals[lo]
                for j in range(lo + 1, i + 1):
                    v = vals[j]
                    if v == v and (not (best_v == best_v) or v < best_v):
                        best_v = v
                        best_j = j
                if best_v == best_v:
                    out[i] = float((i - best_j))
                else:
                    out[i] = np.nan
        return out


# 与 ``cpp/fam_accel.cpp`` 中 ``dyn_op_to_roll_fixed_op`` 一致：0=min … 8=skew, 9=kurt, 10=prod
_DYN_OP_MAP = {
    "min": 0,
    "max": 1,
    "sum": 2,
    "mean": 3,
    "std": 4,
    "var": 5,
    "median": 6,
    "rank_pct": 7,
    "skew": 8,
    "kurt": 9,
    "prod": 10,
}

# Numba 仅实现前四种；其余在无 C++ 时用 ``_roll_segment_py``
_DYN_NUMBA_KINDS = frozenset({"min", "max", "sum", "mean"})

# 兼容旧测试：仅含与 Numba 参考一致的四种动态 op
_OP_MAP = {"min": 0, "max": 1, "sum": 2, "mean": 3}


def _delay_segment_py(vals: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """逐元素动态滞后（与 ``shift_dynamic`` 无 Numba 分支同语义）；供测试与 C++ 对照。"""
    n = len(vals)
    o = np.empty(n, dtype=np.float32)
    for i in range(n):
        k = int(lags[i])
        if k < 0:
            o[i] = np.nan
            continue
        j = i - k
        o[i] = vals[j] if j >= 0 else np.nan
    return o


def _roll_segment_py(vals: np.ndarray, wvals: np.ndarray, kind: str, ddof: int) -> np.ndarray:
    n = len(vals)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        wi = int(wvals[i])
        wi = max(1, min(wi, i + 1))
        lo = i + 1 - wi
        sl = vals[lo : i + 1]
        if kind == "min":
            out[i] = float(np.nanmin(sl))
        elif kind == "max":
            out[i] = float(np.nanmax(sl))
        elif kind == "sum":
            out[i] = float(np.nansum(sl))
        elif kind == "mean":
            out[i] = float(np.nanmean(sl))
        elif kind == "std":
            out[i] = float(np.nanstd(sl, ddof=ddof)) if sl.size > 1 else 0.0
        elif kind == "var":
            out[i] = float(np.nanvar(sl, ddof=ddof))
        elif kind == "median":
            out[i] = float(np.nanmedian(sl))
        elif kind == "rank_pct":
            out[i] = float(pd.Series(sl).rank(pct=True).iloc[-1])
        elif kind == "skew":
            out[i] = float(pd.Series(sl).skew()) if len(sl) > 2 else np.nan
        elif kind == "kurt":
            out[i] = float(pd.Series(sl).kurt()) if len(sl) > 3 else np.nan
        elif kind == "prod":
            p = 1.0
            for v in sl:
                if v == v:
                    p *= float(v)
            out[i] = float(p)
        else:
            raise ValueError(kind)
    return out


def rolling_dynamic(
    df: pd.DataFrame,
    win_df: pd.DataFrame,
    wvals_fn: Callable[[pd.DataFrame, pd.Index], np.ndarray],
    kind: str,
    *,
    ddof: int = 1,
) -> pd.DataFrame:
    """动态滚动聚合；``wvals_fn`` 与 function_registry._dynamic_window_int_series 同类。"""
    result = np.full(len(df), np.nan, dtype=np.float32)
    use_numba = _HAS_NUMBA and kind in _DYN_NUMBA_KINDS

    for _, sub in df.groupby(level="instrument", sort=False):
        sub = sub.sort_index(level="datetime")
        wsub = win_df.reindex(sub.index)
        wvals = wvals_fn(wsub, sub.index).astype(np.int64, copy=False)
        vals = sub.iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
        pos = df.index.get_indexer(sub.index)
        op_map_val = _DYN_OP_MAP.get(kind)
        # 旧 C++ 扩展的 dyn_op_to_roll_fixed_op 仅支持到 _CXX_ROLL_FIXED_MAX_OP；
        # 超出范围（例如 kurt=9）时回退，避免版本不匹配抛异常。
        cxx_supports = (
            _use_cxx_for_dynamic_kernels()
            and op_map_val is not None
            and op_map_val <= _CXX_ROLL_FIXED_MAX_OP
        )
        if cxx_supports:
            par = _use_numba_parallel_for_length(len(vals))
            o = np.asarray(
                _accel_backend().roll_dyn_mm_sm(vals, wvals, op_map_val, par, ddof), dtype=np.float32
            ).reshape(-1)
        elif use_numba:
            op = _DYN_OP_MAP[kind]
            if _use_numba_parallel_for_length(len(vals)):
                o = _roll_dyn_mm_sm_numba_parallel(vals, wvals, op)
            else:
                o = _roll_dyn_mm_sm_numba(vals, wvals, op)
        else:
            o = _roll_segment_py(vals, wvals, kind, ddof)
        result[pos] = o

    return pd.DataFrame(result, index=df.index, columns=df.columns[:1])


def shift_dynamic(
    df: pd.DataFrame,
    lag_df: pd.DataFrame,
    lags_fn: Callable[[pd.DataFrame, pd.Index], np.ndarray],
) -> pd.DataFrame:
    """动态滞后；``lags_fn`` 应产生非负整数（含 0）。"""
    result = np.full(len(df), np.nan, dtype=np.float32)

    for _, sub in df.groupby(level="instrument", sort=False):
        sub = sub.sort_index(level="datetime")
        lsub = lag_df.reindex(sub.index)
        lags = lags_fn(lsub, sub.index).astype(np.int64, copy=False)
        vals = sub.iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
        pos = df.index.get_indexer(sub.index)
        if _use_cxx_for_dynamic_kernels():
            par = _use_numba_parallel_for_length(len(vals))
            o = np.asarray(_accel_backend().delay_dyn(vals, lags, par), dtype=np.float32).reshape(
                -1
            )
        elif _HAS_NUMBA:
            if _use_numba_parallel_for_length(len(vals)):
                o = _delay_dyn_numba_parallel(vals, lags)
            else:
                o = _delay_dyn_numba(vals, lags)
        else:
            n = len(vals)
            o = np.empty(n, dtype=np.float32)
            for i in range(n):
                k = int(lags[i])
                if k < 0:
                    o[i] = np.nan
                    continue
                j = i - k
                o[i] = vals[j] if j >= 0 else np.nan
        result[pos] = o

    return pd.DataFrame(result, index=df.index, columns=df.columns[:1])


def arg_extreme_dynamic(
    df: pd.DataFrame,
    win_df: pd.DataFrame,
    wvals_fn: Callable[[pd.DataFrame, pd.Index], np.ndarray],
    want_max: bool,
) -> pd.DataFrame:
    result = np.full(len(df), np.nan, dtype=np.float32)
    wm = 1 if want_max else 0

    for _, sub in df.groupby(level="instrument", sort=False):
        sub = sub.sort_index(level="datetime")
        wsub = win_df.reindex(sub.index)
        wvals = wvals_fn(wsub, sub.index).astype(np.int64, copy=False)
        vals = sub.iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
        pos = df.index.get_indexer(sub.index)
        if _use_cxx_for_dynamic_kernels():
            par = _use_numba_parallel_for_length(len(vals))
            o = np.asarray(
                _accel_backend().arg_extreme_dyn(vals, wvals, wm, par), dtype=np.float32
            ).reshape(-1)
        elif _HAS_NUMBA:
            if _use_numba_parallel_for_length(len(vals)):
                o = _arg_extreme_dyn_numba_parallel(vals, wvals, wm)
            else:
                o = _arg_extreme_dyn_numba(vals, wvals, wm)
        else:
            n = len(vals)
            o = np.empty(n, dtype=np.float32)
            for i in range(n):
                wi = int(wvals[i])
                wi = max(1, min(wi, i + 1))
                sl = vals[i - wi + 1 : i + 1]
                if want_max:
                    j = int(np.nanargmax(sl))
                else:
                    j = int(np.nanargmin(sl))
                o[i] = float(len(sl) - 1 - j)
        result[pos] = o

    return pd.DataFrame(result, index=df.index, columns=df.columns[:1])


def arg_extreme_fixed(df: pd.DataFrame, window: int, want_max: bool) -> pd.DataFrame:
    """固定窗 TS_ARGMAX / TS_ARGMIN：与 ``rolling(W).apply(nanargmax)`` 语义一致，走 Numba 而非 pandas apply。

    窗宽第 i 根为 ``min(W, i+1)``，与 ``rolling(W, min_periods=1)`` 一致。
    """
    W = max(1, int(window))
    wm = 1 if want_max else 0
    result = np.full(len(df), np.nan, dtype=np.float32)

    for _, sub in df.groupby(level="instrument", sort=False):
        sub = sub.sort_index(level="datetime")
        vals = sub.iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
        n = len(vals)
        wvals = np.minimum(W, np.arange(1, n + 1, dtype=np.int64))
        pos = df.index.get_indexer(sub.index)
        if _use_cxx_for_dynamic_kernels():
            par = _use_numba_parallel_for_length(n)
            o = np.asarray(
                _accel_backend().arg_extreme_dyn(vals, wvals, wm, par), dtype=np.float32
            ).reshape(-1)
        elif _HAS_NUMBA:
            if _use_numba_parallel_for_length(n):
                o = _arg_extreme_dyn_numba_parallel(vals, wvals, wm)
            else:
                o = _arg_extreme_dyn_numba(vals, wvals, wm)
        else:
            o = np.empty(n, dtype=np.float32)
            for i in range(n):
                wi = int(wvals[i])
                wi = max(1, min(wi, i + 1))
                sl = vals[i - wi + 1 : i + 1]
                if want_max:
                    j = int(np.nanargmax(sl))
                else:
                    j = int(np.nanargmin(sl))
                o[i] = float(len(sl) - 1 - j)
        result[pos] = o

    return pd.DataFrame(result, index=df.index, columns=df.columns[:1])


def numba_available() -> bool:
    return _HAS_NUMBA


def numba_parallel_config() -> dict:
    """当前 Numba 并行策略（便于排查性能）。"""
    return {
        "has_numba": _HAS_NUMBA,
        "parallel_mode": _numba_parallel_mode(),
        "parallel_min_rows_default": 4096,
        "parallel_min_rows_effective": _numba_parallel_min_rows(),
    }
