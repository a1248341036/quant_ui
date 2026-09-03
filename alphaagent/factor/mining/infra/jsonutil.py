"""JSON 写出前的数值安全化。

Python ``json.dumps`` 默认 ``allow_nan=True``，会把 ``NaN``/``Infinity`` 写成
非法 JSON 字面量；严格解析器（浏览器 ``JSON.parse``、Starlette
``allow_nan=False`` 响应渲染）都会失败。事件轨迹、摘要、工具结果落盘前统一
经 :func:`json_safe` 清洗：非有限浮点（含 numpy 浮点）替换为 ``None``。
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def json_safe(value: Any) -> Any:
    """递归把非有限浮点（NaN/±Inf）替换为 None，其余原样返回。"""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    return value
