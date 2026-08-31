"""metrics 子模块：纯函数基础设施。

coverage / pearson_ic / spearman_ic 是无状态的数学函数，供 ic、decile、
mls、portfolio 等子模块复用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coverage(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def pearson_ic(factor: np.ndarray, label: np.ndarray, *, min_pairs: int = 30) -> float:
    x = np.asarray(factor, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < min_pairs:
        return float("nan")
    xs = x[mask]
    ys = y[mask]
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = float(np.sqrt((xs * xs).sum() * (ys * ys).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((xs * ys).sum() / denom)


def spearman_ic(factor: np.ndarray, label: np.ndarray, *, min_pairs: int = 10) -> float:
    """Spearman 秩相关（截面或任意配对样本）。"""
    x = np.asarray(factor, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < min_pairs:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_ic(xr, yr, min_pairs=min_pairs)
