# -*- coding: utf-8 -*-
"""AlphaMemo 校准工具：父本质量桶、Eq.7 置信和 APV 双门。"""

from __future__ import annotations

import math
from typing import Any

from .constants import APV_TAU_C_DEFAULT, APV_TAU_V_DEFAULT, EQ7_KAPPA_DEFAULT


def _parent_bucket(parent_ic: Any) -> str:
    """父本质量桶：|IC|<0.015 → low；[0.015, 0.025) → medium；≥0.025 → high。"""
    ic = _safe_float(parent_ic)
    if ic is None:
        return "low"
    a = abs(ic)
    if a < 0.015:
        return "low"
    if a < 0.025:
        return "medium"
    return "high"


def _eq7_confidence(residuals: list[float], *, kappa: float = EQ7_KAPPA_DEFAULT) -> float:
    """AlphaMemo Eq.7 置信度。

    ``c = n/(n+κ) · min(1, |μ|/(σ+ε))``

    - 单观测：n=1 → 约 0.11（κ=8）→ <0.3
    - 正负混杂（低 SNR）：|μ|/σ 小 → <0.3
    - 多次一致残差：|μ|/σ 大且 n 饱和 → >0.7 且渐近 1
    - 空序列 → 0.0
    """
    n = len(residuals)
    if n == 0:
        return 0.0
    vals = [_safe_float(r) for r in residuals]
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return 0.0
    mu = sum(vals) / n
    sigma = math.sqrt(sum((v - mu) ** 2 for v in vals) / n)
    eps = 1e-9
    snr = abs(mu) / (sigma + eps)
    return n / (n + kappa) * min(1.0, snr)


def _apv_gate(
    successes: float,
    failures: float,
    conf: float,
    *,
    tau_c: float = APV_TAU_C_DEFAULT,
    tau_v: float = APV_TAU_V_DEFAULT,
) -> tuple[bool, float, float]:
    """双门 APV 否决：``veto = c > τ_c ∧ π⁻ > τ_v``。

    - ``π⁻ = (1+f)/(2+n)``（Beta(1,1) 均匀先验下失败后验均值）
    - 正证据永不硬放行（只否决失败路径）
    """
    n = successes + failures
    if n <= 0:
        return False, 0.0, 0.0
    pi_neg = (1.0 + failures) / (2.0 + n)
    if conf > tau_c and pi_neg > tau_v:
        return True, pi_neg, pi_neg
    return False, pi_neg, pi_neg


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None