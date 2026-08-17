from __future__ import annotations

"""组合优化器：风险平价 / 均值方差权重求解。

输入历史收益矩阵，输出目标权重向量。约束：全仓、单票权重上限。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk_model import _shrink_cov


def shrink_covariance(rets: np.ndarray) -> np.ndarray:
    """收益矩阵 -> 收缩协方差（LedoitWolf 优先，退化时常数收缩近似）。"""
    mat = np.asarray(rets, dtype=float)
    if mat.ndim != 2 or mat.shape[1] < 2:
        return np.atleast_2d(np.cov(mat, rowvar=False))
    if mat.shape[1] == 2:
        # LedoitWolf 对单变量/奇异样本不稳定，直接退化常数收缩
        S = np.cov(mat, rowvar=False)
        S = np.nan_to_num(S, nan=0.0)
        alpha = max(0.1, min(1.0, 50.0 / (mat.shape[0] + 50)))
        return alpha * np.diag(np.diag(S)) + (1.0 - alpha) * S
    return _shrink_cov(mat)


def _project(w: np.ndarray, max_weight: float | None) -> np.ndarray:
    """把权重投影到可行域：非负、单票上限、归一化。"""
    w = np.maximum(w, 0.0)
    if max_weight is not None:
        w = np.minimum(w, float(max_weight))
    s = w.sum()
    if s <= 0:
        n = len(w)
        w = np.ones(n) / n
        if max_weight is not None:
            w = np.minimum(w, float(max_weight))
            w = w / w.sum()
    else:
        w = w / s
    return w


def risk_parity_weights(cov: np.ndarray,
                        max_weight: float | None = None) -> np.ndarray:
    """风险平价：各资产风险贡献相等。"""
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)

    def rc_sq(w: np.ndarray) -> np.ndarray:
        w = w / w.sum()
        vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        mrc = cov @ w / vol
        return (w * mrc) ** 2  # 用平方规避符号

    def obj(w: np.ndarray) -> float:
        r = rc_sq(w)
        return float(np.var(r))

    w0 = _project(np.ones(n) / n, max_weight)
    bounds = [(0.0, max_weight if max_weight else 1.0)] * n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 200, "ftol": 1e-8})
    return _project(res.x if res.success else w0, max_weight)


def mean_variance_weights(returns: np.ndarray,
                          gamma: float = 1.0,
                          max_weight: float | None = None) -> np.ndarray:
    """均值方差：最大化 w'μ - γ·w'Σw（γ 为风险厌恶系数）。"""
    n = returns.shape[1]
    mu = np.nanmean(returns, axis=0)
    mu = np.nan_to_num(mu, nan=0.0)
    cov = np.cov(returns, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    if n == 1:
        return np.ones(1)

    def neg_obj(w: np.ndarray) -> float:
        w = w / w.sum()
        return float(-(w @ mu - gamma * w @ cov @ w))

    w0 = _project(np.ones(n) / n, max_weight)
    bounds = [(0.0, max_weight if max_weight else 1.0)] * n
    res = minimize(neg_obj, w0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 200, "ftol": 1e-8})
    return _project(res.x if res.success else w0, max_weight)


def max_diversification_weights(cov: np.ndarray,
                                max_weight: float | None = None) -> np.ndarray:
    """最大化分散化比率：DR = Σ(w_i·σ_i) / sqrt(w'Σw)。"""
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    def neg_dr(w: np.ndarray) -> float:
        w = w / w.sum()
        port_var = max(float(w @ cov @ w), 1e-12)
        return -(float(w @ vol) / np.sqrt(port_var))

    w0 = _project(np.ones(n) / n, max_weight)
    bounds = [(0.0, max_weight if max_weight else 1.0)] * n
    res = minimize(neg_dr, w0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 200, "ftol": 1e-8})
    return _project(res.x if res.success else w0, max_weight)


def weights_from_returns(returns: pd.DataFrame,
                         method: str = "risk_parity",
                         gamma: float = 1.0,
                         max_weight: float | None = None,
                         cov_shrink: bool = True) -> dict[str, float]:
    """DataFrame 版本：列=股票，行=收益。返回 {列名: 权重}。"""
    if returns.empty or len(returns.columns) == 0:
        return {}
    mat = returns.values.astype(float)
    if method == "risk_parity":
        cov = shrink_covariance(mat) if cov_shrink else np.cov(mat, rowvar=False)
        cov = np.nan_to_num(cov, nan=0.0)
        w = risk_parity_weights(cov, max_weight=max_weight)
    elif method == "mean_variance":
        w = mean_variance_weights(mat, gamma=gamma, max_weight=max_weight)
    elif method == "max_diversification":
        cov = shrink_covariance(mat) if cov_shrink else np.cov(mat, rowvar=False)
        cov = np.nan_to_num(cov, nan=0.0)
        w = max_diversification_weights(cov, max_weight=max_weight)
    else:
        raise ValueError(f"未知组合优化方法: {method}")
    return {c: float(wi) for c, wi in zip(returns.columns, w) if wi > 1e-8}
