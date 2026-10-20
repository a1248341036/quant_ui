# -*- coding: utf-8 -*-
"""组合增量计算（增量轨的数学内核，2026-10-20）。

**定位**：候选池的实际用途是 ML 样本源，lgbm 要的是"现有组合解释不掉的新信息"。
本模块给出**不重训模型**的增量度量（实测 0.5-1s/因子，对比重训法 3-10 分钟/因子）：

    candidate_t = α_t + β_t · combo_score_t + ε_t      # 按日截面单变量 OLS（闭式解）
    raw_ic    = corr(candidate, label)
    resid_ic  = corr(ε, label)                          # 剥掉组合后剩余 alpha
    increment = |resid_ic| / |raw_ic|                   # 组合解释不掉的比例

**combo_score 的硬约束（勿改）**
1. **禁止**用 `data/stock/pred_demo.parquet`：它只有 2025-01 起（盲测段），
   拿它做因子筛选 = 数据窥视。combo 必须由 **candidate 池因子在 train 段**合成。
2. 审 **库内** 因子要用 leave-one-out（剔除它自己），否则它作为控制变量把残差压扁、
   系统性低估自身增量；审 **库外** 新因子用完整候选池。
3. `increment` 是比值，**分母趋零会爆表**（实测 `x_vol_pe`：raw_ic +0.0000 /
   resid +0.0014 → increment 53.26）。因此判定必须同时要求 `raw_ic` 地板，
   见 `CandidateCriteria.min_raw_ic_floor`。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_ROWS_ROOT = Path(__file__).resolve().parents[4]  # alphaagent/factor/mining/eval -> 仓库根
CANDIDATE_REGISTRY = (
    _ROWS_ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main"
    / "mining_candidate_registry.json"
)
COMBO_CACHE = _ROWS_ROOT / "artifacts" / "alphaagent" / "combo" / "combo_zscore.parquet"


def load_candidate_exprs(registry_path: Path | None = None) -> list[dict[str, str]]:
    """读候选池因子表达式（combo 的原料）。"""
    path = registry_path or CANDIDATE_REGISTRY
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for name, rec in data.items():
        expr = rec.get("expr") or rec.get("expression")
        if expr:
            out.append({"name": name, "expr": expr})
    return out


def cs_zscore(s: pd.Series) -> pd.Series:
    """按日期截面 zscore（combo 与因子值都必须先过这一步才能相加）。"""
    d = s.index.get_level_values(0)
    g = s.groupby(d)
    mu = g.transform("mean")
    sd = g.transform("std")
    return ((s - mu) / sd.replace(0, np.nan)).astype("float32")


def xs_corr(a: pd.Series, b: pd.Series) -> float:
    """按日期截面 Pearson 相关，返回按日均值（全向量化，无 groupby.apply）。"""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 100:
        return float("nan")
    d = df.index.get_level_values(0)
    g = df.groupby(d)
    n = g["a"].size().astype(float)
    sa, sb = g["a"].sum(), g["b"].sum()
    sab = (df["a"] * df["b"]).groupby(d).sum()
    sa2 = (df["a"] ** 2).groupby(d).sum()
    sb2 = (df["b"] ** 2).groupby(d).sum()
    cov = sab - sa * sb / n
    va, vb = sa2 - sa ** 2 / n, sb2 - sb ** 2 / n
    corr = (cov / np.sqrt(va * vb)).replace([np.inf, -np.inf], np.nan)
    return float(corr.dropna().mean())


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """按日期截面做单变量 OLS 取残差（闭式解，全向量化）。"""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1)
    mask = df["y"].notna() & df["x"].notna()
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if not mask.any():
        return out
    sub = df[mask]
    d = sub.index.get_level_values(0)
    g = sub.groupby(d)
    n = g["y"].size().astype(float)
    sy, sx = g["y"].sum(), g["x"].sum()
    sxy = (sub["y"] * sub["x"]).groupby(d).sum()
    sxx = (sub["x"] ** 2).groupby(d).sum()
    beta = ((sxy - sy * sx / n) / (sxx - sx ** 2 / n)).replace([np.inf, -np.inf], np.nan)
    alpha = sy / n - beta * (sx / n)
    codes = sub.index.get_level_values(0)
    b = beta.reindex(codes).to_numpy()
    a_ = alpha.reindex(codes).to_numpy()
    out.loc[sub.index] = sub["y"].to_numpy() - a_ - b * sub["x"].to_numpy()
    return out


def increment_metrics(
    factor_z: pd.Series,
    combo: pd.Series,
    label: pd.Series,
) -> dict[str, Any]:
    """算一个因子的 raw_ic / resid_ic / increment / combo_corr。

    ``combo`` 由候选池合成（库内因子调用方应先做 leave-one-out）。
    """
    raw_ic = xs_corr(factor_z, label)
    combo_corr = xs_corr(factor_z, combo)
    if combo.notna().sum() < 1000:
        return {"raw_ic": raw_ic, "resid_ic": None, "increment": None,
                "combo_corr": combo_corr, "reason": "no_combo"}
    resid = residualize(factor_z, combo)
    resid_ic = xs_corr(resid, label)
    inc = None
    if (raw_ic is not None and not np.isnan(raw_ic) and raw_ic != 0
            and resid_ic is not None and not np.isnan(resid_ic)):
        inc = abs(resid_ic) / abs(raw_ic)
    return {"raw_ic": raw_ic, "resid_ic": resid_ic, "increment": inc,
            "combo_corr": combo_corr, "reason": None}


def save_combo(combo: pd.Series, path: Path | None = None) -> Path:
    """落盘 combo（供后续评估直接读，不必每次重算 47 个因子）。"""
    p = path or COMBO_CACHE
    p.parent.mkdir(parents=True, exist_ok=True)
    combo.rename("combo_score").reset_index().to_parquet(p, index=False)
    return p


def load_combo(path: Path | None = None) -> pd.Series:
    """读缓存 combo；缺失返回空 Series（调用方应降级为跳过增量轨）。"""
    p = path or COMBO_CACHE
    if not p.is_file():
        return pd.Series(dtype="float64")
    df = pd.read_parquet(p)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("datetime") or cols.get("date")
    ccol = cols.get("instrument") or cols.get("code") or cols.get("symbol")
    scol = cols.get("combo_score") or cols.get("score") or cols.get("value")
    if not (dcol and ccol and scol):
        return pd.Series(dtype="float64")
    df[dcol] = pd.to_datetime(df[dcol])
    s = df.set_index([dcol, ccol])[scol].astype("float64")
    s.index.names = ["datetime", "instrument"]
    return s.sort_index()
