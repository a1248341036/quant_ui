"""截面（per-datetime）算子的 Numba 并行内核。

2026-10-08：``RANK`` / ``CS_WINSORIZE`` / ``CS_BUCKET`` / ``CS_NEUTRALIZE`` /
``CS_GROUP_RANK`` 原先按天走 Python 循环 + pandas 分组，单次调用 0.6~8 s（7.78M 行
面板，5,449 品种 × 1,636 日），实测占 DSL 求值累计耗时的 57%（
``artifacts/dsl_operator_profiling.jsonl``）。本模块把它们统一改成
「``ops_kit.datetime_group_bounds`` 取 datetime 组边界 + Numba ``prange`` 逐日并行」，
``nogil=True`` 让挖掘的 4 个并行 worker 真正同时跑内核（旧 pandas 路径持 GIL）。

**数值语义与 pandas 回落路径逐位一致**（``tests/test_cs_operator_accel.py`` 做新旧对照
门禁）。三处需要按 pandas 原样复刻的细节，改动前务必先读：

1. 分位数有**两套**互不相同的 pandas 实现，不能合并（实测各 4500 组对照标定）：
   - ``CS_BUCKET`` 走 ``pd.qcut`` → ``Series.quantile`` → ``np.percentile(qs * 100)``，
     而 ``np.percentile`` 内部又 ``q / 100``；这个 ×100/÷100 往返在浮点上不是恒等
     （如 ``q=1/3``），且插值用 numpy ``_lerp`` 的
     ``g >= 0.5 ? b - d*(1-g) : a + d*g``。对应 ``_quantile_np``。
   - ``CS_WINSORIZE`` 走 ``Series.groupby(...).quantile``（Cython），不做往返、
     恒用 ``a[lo] + g*(a[hi]-a[lo])``。对应 ``_quantile_gb``。
   写错任一处都会在浮点边界上错分档/错裁剪。
2. ``CS_WINSORIZE`` 的裁剪必须按 ``np.clip == minimum(maximum(v, lo), hi)`` 复刻：
   截面含 ±inf 时插值会算出 NaN 边界（``inf - inf`` / ``0 * inf``），此时整段结果
   必须为 NaN，不能退化成"不裁剪"。
3. CS_BUCKET 复刻 ``pd.qcut(labels=False, duplicates="drop")``：``linspace(0,1,n+1)``
   的第 k 个分位是 ``k * (1/n)``（末位强制 1.0），去重后 ``searchsorted(side="left")``，
   再把 ``== edges[0]`` 的 id 提到 1（``include_lowest=True``）。

面板未按 datetime 非递减排序（``bounds is None``）、numba 缺失、或
``ALPHA_DSL_CS_ACCEL=0/off/false/no`` 时，公共函数返回 ``None``，调用方回落原
pandas 路径。
"""
from __future__ import annotations

import os

import numpy as np

try:  # pragma: no cover - 导入分支按环境走
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
    prange = range  # type: ignore[misc, assignment]

    def njit(*args, **kwargs):
        def _wrap(f):
            return f

        return _wrap if not args else args[0]


_ENV_SWITCH = "ALPHA_DSL_CS_ACCEL"


def cs_accel_enabled() -> bool:
    """快路径是否启用（numba 可用且未被环境变量关闭）。"""
    if not _HAS_NUMBA:
        return False
    return os.environ.get(_ENV_SWITCH, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


# -----------------------------------------------------------------------------
# 内核
# -----------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _not_nan(v: float) -> bool:
    """``notna``：只有 NaN 无效（±inf 算有效，与 pandas rank / quantile 一致）。"""
    return v == v


@njit(cache=True, inline="always")
def _quantile_np(a: np.ndarray, q: float) -> float:
    """``np.percentile(a, q * 100, method="linear")`` 的逐位等价实现（a 已升序）。

    供 CS_BUCKET 使用（``pd.qcut`` → ``Series.quantile`` → ``np.percentile``）。
    """
    c = a.shape[0]
    if c == 1:
        return a[0]
    q_eff = (q * 100.0) / 100.0
    vi = q_eff * (c - 1)
    lo = int(np.floor(vi))
    if lo < 0:
        lo = 0
    elif lo > c - 1:
        lo = c - 1
    hi = lo + 1
    if hi > c - 1:
        hi = c - 1
    g = vi - lo
    d = a[hi] - a[lo]
    if g >= 0.5:
        return a[hi] - d * (1.0 - g)
    return a[lo] + d * g


@njit(cache=True, inline="always")
def _quantile_gb(a: np.ndarray, q: float) -> float:
    """``Series.groupby(...).quantile(q)`` 的逐位等价实现（a 已升序）。

    与 ``_quantile_np`` 有两处不同（实测标定，勿合并）：
    1. 不做 numpy ``q*100/100`` 往返（groupby 走 Cython，直接用 q）；
    2. 恒用 ``a[lo] + g*(a[hi]-a[lo])``，不用 numpy ``_lerp`` 的
       ``g>=0.5 → b - d*(1-g)`` 分支。
    供 CS_WINSORIZE 使用（旧实现调 ``grp.quantile``）。
    """
    c = a.shape[0]
    if c == 1:
        return a[0]
    vi = q * (c - 1)
    lo = int(np.floor(vi))
    if lo < 0:
        lo = 0
    elif lo > c - 1:
        lo = c - 1
    hi = lo + 1
    if hi > c - 1:
        hi = c - 1
    g = vi - lo
    return a[lo] + g * (a[hi] - a[lo])


@njit(cache=True, inline="always")
def _linspace_q(k: int, n: int) -> float:
    """``np.linspace(0, 1, n + 1)[k]``：``k * (1/n)``，末位强制 1.0。"""
    if k == n:
        return 1.0
    return k * (1.0 / n)


@njit(cache=True, parallel=True, nogil=True)
def _rank_pct_kernel(arr: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """逐日 ``rank(pct=True, method="average")``。"""
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    for d in prange(bounds.shape[0] - 1):
        st = bounds[d]
        en = bounds[d + 1]
        c = 0
        for i in range(st, en):
            if _not_nan(arr[i]):
                c += 1
        if c == 0:
            continue
        vals = np.empty(c, dtype=np.float64)
        pos = np.empty(c, dtype=np.int64)
        j = 0
        for i in range(st, en):
            v = arr[i]
            if _not_nan(v):
                vals[j] = v
                pos[j] = i
                j += 1
        order = np.argsort(vals)
        i0 = 0
        while i0 < c:
            i1 = i0 + 1
            v0 = vals[order[i0]]
            while i1 < c and vals[order[i1]] == v0:
                i1 += 1
            pct = np.float32((0.5 * (i0 + 1 + i1)) / c)
            for t in range(i0, i1):
                out[pos[order[t]]] = pct
            i0 = i1
    return out


@njit(cache=True, parallel=True, nogil=True)
def _neutralize_kernel(xv, gv, bounds):
    """逐日按 group 值分组，输出 ``x - group_mean(x)``（组内 1 个有效值 → 0）。"""
    out = np.full(xv.shape[0], np.nan, dtype=np.float32)
    for d in prange(bounds.shape[0] - 1):
        st = bounds[d]
        en = bounds[d + 1]
        cg = 0
        for i in range(st, en):
            if np.isfinite(gv[i]):
                cg += 1
        if cg == 0:
            continue
        gsel = np.empty(cg, dtype=np.float64)
        gpos = np.empty(cg, dtype=np.int64)
        j = 0
        for i in range(st, en):
            g = gv[i]
            if np.isfinite(g):
                gsel[j] = g
                gpos[j] = i
                j += 1
        order = np.argsort(gsel)
        i0 = 0
        while i0 < cg:
            i1 = i0 + 1
            g0 = gsel[order[i0]]
            while i1 < cg and gsel[order[i1]] == g0:
                i1 += 1
            total = 0.0
            n = 0
            for t in range(i0, i1):
                xi = xv[gpos[order[t]]]
                if np.isfinite(xi):
                    total += xi
                    n += 1
            if n == 1:
                for t in range(i0, i1):
                    p = gpos[order[t]]
                    if np.isfinite(xv[p]):
                        out[p] = np.float32(0.0)
            elif n > 1:
                mu = total / n
                for t in range(i0, i1):
                    p = gpos[order[t]]
                    xi = xv[p]
                    if np.isfinite(xi):
                        out[p] = np.float32(xi - mu)
            i0 = i1
    return out


@njit(cache=True, parallel=True, nogil=True)
def _group_rank_pct_kernel(xv, gv, bounds):
    """逐日按 group 值分组，组内对 x 做 ``rank(pct=True, method="average")``。"""
    out = np.full(xv.shape[0], np.nan, dtype=np.float32)
    for d in prange(bounds.shape[0] - 1):
        st = bounds[d]
        en = bounds[d + 1]
        cg = 0
        for i in range(st, en):
            if np.isfinite(gv[i]):
                cg += 1
        if cg == 0:
            continue
        gsel = np.empty(cg, dtype=np.float64)
        gpos = np.empty(cg, dtype=np.int64)
        j = 0
        for i in range(st, en):
            g = gv[i]
            if np.isfinite(g):
                gsel[j] = g
                gpos[j] = i
                j += 1
        order = np.argsort(gsel)
        i0 = 0
        while i0 < cg:
            i1 = i0 + 1
            g0 = gsel[order[i0]]
            while i1 < cg and gsel[order[i1]] == g0:
                i1 += 1
            c = 0
            for t in range(i0, i1):
                if np.isfinite(xv[gpos[order[t]]]):
                    c += 1
            if c == 0:
                i0 = i1
                continue
            vals = np.empty(c, dtype=np.float64)
            pos = np.empty(c, dtype=np.int64)
            jj = 0
            for t in range(i0, i1):
                p = gpos[order[t]]
                xi = xv[p]
                if np.isfinite(xi):
                    vals[jj] = xi
                    pos[jj] = p
                    jj += 1
            sub = np.argsort(vals)
            k0 = 0
            while k0 < c:
                k1 = k0 + 1
                v0 = vals[sub[k0]]
                while k1 < c and vals[sub[k1]] == v0:
                    k1 += 1
                pct = np.float32((0.5 * (k0 + 1 + k1)) / c)
                for t in range(k0, k1):
                    out[pos[sub[t]]] = pct
                k0 = k1
            i0 = i1
    return out


@njit(cache=True, parallel=True, nogil=True)
def _winsorize_kernel(arr, bounds, lower_pct: float, upper_pct: float):
    """逐日把有效值 clip 到 ``[lower_pct, upper_pct]`` 分位之间。

    有效值口径同 pandas ``groupby.quantile``：只丢 NaN，±inf 参与分位计算再被 clip。
    """
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    for d in prange(bounds.shape[0] - 1):
        st = bounds[d]
        en = bounds[d + 1]
        c = 0
        for i in range(st, en):
            if _not_nan(arr[i]):
                c += 1
        if c == 0:
            continue
        vals = np.empty(c, dtype=np.float64)
        j = 0
        for i in range(st, en):
            v = arr[i]
            if _not_nan(v):
                vals[j] = v
                j += 1
        sorted_vals = np.sort(vals)
        q_lo = _quantile_gb(sorted_vals, lower_pct)
        q_hi = _quantile_gb(sorted_vals, upper_pct)
        for i in range(st, en):
            v = arr[i]
            if _not_nan(v):
                # np.clip(v, q_lo, q_hi) == minimum(maximum(v, q_lo), q_hi)：
                # 边界为 NaN（含 inf 的截面做插值时会出现）时结果必须整体 NaN
                t = v if v > q_lo else q_lo
                out[i] = np.float32(t if t < q_hi else q_hi)
    return out


@njit(cache=True, parallel=True, nogil=True)
def _bucket_kernel(arr, bounds, n_bins: int):
    """逐日 ``pd.qcut(valid, n_bins, labels=False, duplicates="drop")`` 的等价实现。

    有效值口径同旧实现：``np.isfinite``（±inf 视作缺失，整行保持 NaN）。
    """
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    for d in prange(bounds.shape[0] - 1):
        st = bounds[d]
        en = bounds[d + 1]
        c = 0
        for i in range(st, en):
            if np.isfinite(arr[i]):
                c += 1
        if c < n_bins:
            continue  # 有效样本 < n_bins：整段 NaN（与旧实现一致）
        vals = np.empty(c, dtype=np.float64)
        j = 0
        for i in range(st, en):
            v = arr[i]
            if np.isfinite(v):
                vals[j] = v
                j += 1
        sorted_vals = np.sort(vals)

        nb = n_bins + 1
        edges = np.empty(nb, dtype=np.float64)
        for k in range(nb):
            edges[k] = _quantile_np(sorted_vals, _linspace_q(k, n_bins))
        # duplicates="drop"：去掉重复边界（分位单调不减 → 保留每组首个即等价），
        # 就地压缩到 edges 前 nu 位，避免切片重绑定（numba 布局类型不一致）。
        nu = 0
        for k in range(nb):
            if nu == 0 or edges[k] != edges[nu - 1]:
                edges[nu] = edges[k]
                nu += 1
        if nu < nb and nb != 2:
            nb = nu

        first = edges[0]
        for i in range(st, en):
            v = arr[i]
            if not np.isfinite(v):
                continue
            # searchsorted(edges, v, side="left")
            lo = 0
            hi = nb
            while lo < hi:
                mid = (lo + hi) // 2
                if edges[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            ids = lo
            if v == first:
                ids = 1
            if ids == 0 or ids == nb:
                continue
            out[i] = np.float32(ids - 1)
    return out


# -----------------------------------------------------------------------------
# 公共入口：不可用时返回 None，调用方回落 pandas 路径
# -----------------------------------------------------------------------------


def _prepare(arr) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _prepare_bounds(bounds) -> np.ndarray | None:
    if bounds is None or not cs_accel_enabled():
        return None
    return np.ascontiguousarray(bounds, dtype=np.int64)


def rank_pct(arr, bounds):
    """``RANK`` 快路径；不可用返回 None。"""
    bd = _prepare_bounds(bounds)
    if bd is None:
        return None
    return _rank_pct_kernel(_prepare(arr), bd)


def neutralize(x, group, bounds):
    """``CS_NEUTRALIZE`` 快路径；不可用返回 None。"""
    bd = _prepare_bounds(bounds)
    if bd is None:
        return None
    return _neutralize_kernel(_prepare(x), _prepare(group), bd)


def group_rank_pct(x, group, bounds):
    """``CS_GROUP_RANK`` 快路径；不可用返回 None。"""
    bd = _prepare_bounds(bounds)
    if bd is None:
        return None
    return _group_rank_pct_kernel(_prepare(x), _prepare(group), bd)


def winsorize(arr, bounds, lower_pct: float, upper_pct: float):
    """``CS_WINSORIZE`` 快路径；不可用返回 None。"""
    bd = _prepare_bounds(bounds)
    if bd is None:
        return None
    return _winsorize_kernel(_prepare(arr), bd, float(lower_pct), float(upper_pct))


def bucket(arr, bounds, n_bins: int):
    """``CS_BUCKET`` 快路径；不可用返回 None。"""
    bd = _prepare_bounds(bounds)
    if bd is None:
        return None
    return _bucket_kernel(_prepare(arr), bd, int(n_bins))
