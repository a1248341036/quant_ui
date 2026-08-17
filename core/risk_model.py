from __future__ import annotations

"""轻量 Barra 风格风险模型 + 风险归因 + 因子中性化。

- 风格因子：流动性/动量/低波动/换手（日线），可选价值/盈利质量/成长（财务）
- 行业因子：二进制哑变量（去掉一个基准行业，与截距一起避免哑变量陷阱）
- 逐期横截面回归 r = Xβ + ε 得到因子收益，因子协方差用收缩估计
- 资产协方差 = X ΣF X' + diag(specific)
- neutralize：把因子得分对风险暴露回归取残差（剔除风格/行业暴露）

注意：本模块是"轻量"实现，风格暴露用代理变量（成交额代理规模/流动性），
行业分类来自本地 tech 缓存；数据完整后可替换为更细的 Barra 因子定义。
"""

import numpy as np


# ---------- 暴露矩阵 ----------

def _winsorize(mat: np.ndarray, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    return np.clip(mat, lo, hi)


def _standardize(mat: np.ndarray) -> np.ndarray:
    """横截面 z-score（按行=日期）。全 NaN 行保留 NaN。"""
    cnt = np.sum(~np.isnan(mat), axis=1, keepdims=True)
    s = np.nansum(np.where(np.isnan(mat), 0.0, mat), axis=1, keepdims=True)
    mu = np.divide(s, cnt, out=np.full_like(s, np.nan), where=cnt > 0)
    sq = np.nansum(np.where(np.isnan(mat), 0.0, (mat - mu) ** 2), axis=1, keepdims=True)
    sd = np.divide(np.sqrt(sq / np.maximum(cnt - 1, 1.0)),
                   cnt, out=np.full_like(sq, np.nan), where=cnt > 1)
    sd[sd < 1e-12] = 1.0
    return (mat - mu) / sd


def build_exposures(
    close: np.ndarray,
    am20: np.ndarray,
    turn20: np.ndarray,
    mom20: np.ndarray | None = None,
    vol20: np.ndarray | None = None,
    pb: np.ndarray | None = None,
    roe: np.ndarray | None = None,
    growth: np.ndarray | None = None,
    industry_map: dict[str, str] | None = None,
    codes: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """返回 (X, factor_names)。X: T x K x P。

    风格因子横截面 z-score 并截尾；行业哑变量二进制（去掉基准行业）。
    行业哑变量本身"非标准化"（0/1），回归时自带截距。
    """
    T, K = close.shape
    style_arrays: list[np.ndarray] = []
    style_names: list[str] = []

    if am20 is not None:
        style_arrays.append(np.log1p(np.maximum(am20, 0.0)))
        style_names.append("liquidity")
    if mom20 is not None:
        style_arrays.append(np.nan_to_num(mom20, nan=np.nan))
        style_names.append("momentum")
    if vol20 is not None:
        style_arrays.append(vol20)
        style_names.append("volatility")
    if turn20 is not None:
        style_arrays.append(turn20)
        style_names.append("turnover")
    if pb is not None:
        style_arrays.append(-np.log1p(np.maximum(pb, 0.0)))
        style_names.append("value")
    if roe is not None:
        style_arrays.append(roe)
        style_names.append("quality")
    if growth is not None:
        style_arrays.append(growth)
        style_names.append("growth")

    if not style_arrays:
        raise ValueError("至少需要一个风格因子")

    X_style = np.full((T, K, len(style_names)), np.nan)
    for i, arr in enumerate(style_arrays):
        X_style[:, :, i] = _winsorize(_standardize(np.asarray(arr, dtype=float)))

    names = list(style_names)
    if industry_map and codes:
        ind_arr = np.array([industry_map.get(str(c), "其他") for c in codes])
        inds = sorted({str(i) for i in ind_arr})
        ref = inds[0]
        ind_names = [f"ind_{i}" for i in inds if i != ref]
        X_ind = np.zeros((T, K, len(ind_names)))
        for j, name in enumerate(ind_names):
            mask = ind_arr == name[4:]
            X_ind[:, :, j] = mask.astype(float)
        X = np.concatenate([X_style, X_ind], axis=2)
        names += ind_names
        return X, names

    return X_style, names


# ---------- 因子收益 / 协方差 ----------

def factor_returns(rets: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐期横截面回归 r = Xβ + ε。

    返回 (beta: T x P, specific_var: T x K)。
    回归含截距，beta 只存 X 部分的系数。
    """
    T, K, P = X.shape
    beta = np.full((T, P), np.nan)
    specific = np.full((T, K), np.nan)
    for t in range(T):
        r = rets[t]
        Xt = X[t]
        valid = np.isfinite(r) & np.isfinite(Xt).all(axis=1)
        if int(valid.sum()) < P + 5:
            continue
        Xv = np.column_stack([np.ones(int(valid.sum())), Xt[valid]])
        rv = r[valid]
        try:
            coef, *_ = np.linalg.lstsq(Xv, rv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        beta[t] = coef[1:]
        resid = rv - Xv @ coef
        specific[t, valid] = resid ** 2
    return beta, specific


def _shrink_cov(mat: np.ndarray) -> np.ndarray:
    """因子协方差收缩。优先 sklearn LedoitWolf，否则常数收缩近似。"""
    mat = mat[np.isfinite(mat).all(axis=1)]
    n = mat.shape[1]
    if len(mat) < 3:
        return np.eye(n) * 1e-4
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(mat).covariance_
    except Exception:
        pass
    S = np.cov(mat, rowvar=False)
    alpha = max(0.1, min(1.0, 50.0 / (len(mat) + 50)))
    return alpha * np.diag(np.diag(S)) + (1.0 - alpha) * S


def covariance_from_exposures(
    X: np.ndarray,
    rets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (asset_cov, factor_cov, specific_var)。

    asset_cov 用因子暴露均值 + 收缩因子协方差 + 特异方差近似：
      Σ = X̄ ΣF X̄' + diag(specific)
    """
    beta, specific = factor_returns(rets, X)
    factor_cov = _shrink_cov(beta)
    spec_var = np.nanmean(specific, axis=0)
    spec_var = np.nan_to_num(spec_var, nan=1e-4)
    spec_var = np.maximum(spec_var, 1e-6)

    X_valid = np.where(np.isfinite(X), X, 0.0)
    X_mean = X_valid.mean(axis=0)  # K x P
    asset_cov = X_mean @ factor_cov @ X_mean.T + np.diag(spec_var)
    return asset_cov, factor_cov, spec_var


# ---------- 风险归因 ----------

def portfolio_risk_attribution(
    weights: np.ndarray,
    X: np.ndarray,
    factor_cov: np.ndarray,
    specific_var: np.ndarray,
    names: list[str],
) -> dict:
    """组合方差按因子分解（风险贡献）。

    b = X̄'w；组合因子方差 = b' ΣF b。每因子贡献 = b_p × (ΣF b)_p / total。
    返回 {因子名: 占比} + {"specific": 残差占比}。
    """
    w = np.asarray(weights, dtype=float)
    X_use = np.where(np.isfinite(X), X, 0.0)
    X_mean = X_use.mean(axis=0)
    b = X_mean.T @ w
    total_f = float(b @ factor_cov @ b)
    mrc = factor_cov @ b
    contrib = b * mrc
    spec = float(w @ (np.maximum(specific_var, 0.0) * w))
    total = total_f + spec
    out: dict[str, float] = {}
    if total > 1e-18:
        for i, name in enumerate(names):
            out[name] = float(contrib[i] / total)
        out["specific"] = float(spec / total)
    else:
        for name in names:
            out[name] = 0.0
        out["specific"] = 1.0
    return out


# ---------- 因子中性化 ----------

def neutralize(score: np.ndarray, X: np.ndarray) -> np.ndarray:
    """逐期把得分对风险暴露矩阵回归取残差。score: T x K, X: T x K x P。"""
    T, K = score.shape
    out = np.full_like(score, np.nan)
    for t in range(T):
        s = score[t]
        Xt = X[t]
        valid = np.isfinite(s) & np.isfinite(Xt).all(axis=1)
        n = int(valid.sum())
        if n < 5:
            continue
        Xv = np.column_stack([np.ones(n), Xt[valid]])
        sv = s[valid]
        try:
            coef, *_ = np.linalg.lstsq(Xv, sv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[t, valid] = sv - Xv @ coef
    return out
