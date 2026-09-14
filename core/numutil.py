"""数值处理公共工具模块。"""
from __future__ import annotations

import math
from typing import Any


def to_float(
    v: Any,
    default: float | None = None,
    *,
    allow_inf: bool = False,
) -> float | None:
    """统一安全数值转换。

    - 正常数字/数值字符串 -> float
    - None / 非法字符串 / 格式错误 -> default (默认 None)
    - NaN -> default (默认 None)
    - inf / -inf:
        - 默认 allow_inf=False -> default (严格 JSON 安全)
        - allow_inf=True -> float('inf') / float('-inf')（用于统计/校准需保留极端符号的场景）
    """
    if v is None:
        return default
    try:
        x = float(v)
        if math.isnan(x):
            return default
        if not allow_inf and math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default
