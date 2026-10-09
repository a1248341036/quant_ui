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


def portfolio_vol(close_wide: pd.DataFrame | None,
                  window: int = 60,
                  codes: list | None = None,
                  min_obs: int | None = None,
                  periods_per_year: int = 252) -> float:
    """持仓等权组合日收益的滚动已实现波动（年化）——vol targeting 的 σ_p 预测。

    close_wide 为收盘价宽表（index=交易日、columns=code、停牌/缺失为 NaN）；
    codes 为当前持仓票（None=全部列）。取最近 window+1 行，逐日对"当日有
    有效收益的票"等权平均得到组合日收益序列，样本 std（ddof=1）后按
    √periods_per_year 年化。

    选择滚动已实现波动而非 shrink_covariance：权重出口每个调仓日只需要一个
    标量 σ_p，等权组合日收益的滚动 std 为 O(window×N) 且无优化器不稳定问题；
    协方差法仍留给完整优化器（weights_from_returns）。

    安全退化（窗口不足/数据无效 → 返回 NaN，调用方将 vol scale 退化为 1）：
    - close_wide 为 None/空、codes 全部不在列中、window <= 0；
    - 有效收益观测数 < min_obs（默认 max(2, window//2)，即窗口不足）；
    - 全部收益非有限（NaN/inf，如 0 价格污染）。
    σ_p 恰为 0（收益恒为 0，如长期停牌价格不变）时返回 0.0，由调用方按
    "零波动=数据不可信"退化为 scale=1，而非放大仓位。
    """
    if close_wide is None or window <= 0:
        return float("nan")
    frame = close_wide
    if codes is not None:
        cols = [c for c in codes if c in frame.columns]
        if not cols:
            return float("nan")
        frame = frame[cols]
    if frame.empty or frame.shape[1] == 0:
        return float("nan")
    frame = frame.tail(int(window) + 1)
    # fill_method=None：停牌/缺失保持 NaN（不前向填充），停牌日被逐日等权
    # 均值剔除；复牌日收益跨停牌期结转（lump），与 pct_change 语义一致。
    rets = frame.pct_change(fill_method=None)
    port = rets.mean(axis=1)
    vals = port.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if min_obs is None:
        min_obs = max(2, int(window) // 2)
    if len(vals) < max(2, int(min_obs)):
        return float("nan")
    sigma_daily = float(np.std(vals, ddof=1))
    if not np.isfinite(sigma_daily):
        return float("nan")
    return float(sigma_daily * np.sqrt(periods_per_year))


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
