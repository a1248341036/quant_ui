"""数值处理公共工具模块。"""
from __future__ import annotations

import math
from typing import Any


def to_float(v: Any, default: float | None = None) -> float | None:
    """统一安全数值转换。

    - 正常数字/数值字符串 -> float
    - None / 非法字符串 / 格式错误 -> default (默认 None)
    - NaN / inf / -inf -> default (默认 None)
    """
    if v is None:
        return default
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default
