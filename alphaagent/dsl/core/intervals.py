"""周期归一化（混频 resample / evaluator 依赖）。"""
from __future__ import annotations

import re

_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*([mhd])(?:in)?\s*$", re.IGNORECASE)


def normalize_bar_interval(value: str | int) -> str:
    """把 ``5`` / ``5min`` / ``5m`` / ``1H`` 归一为 ``5m`` / ``1h``。"""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"bar_interval 必须为正整数，收到: {value!r}")
        return f"{value}m"
    s = str(value).strip().lower()
    if not s:
        raise ValueError("bar_interval 不能为空")
    m = _INTERVAL_RE.match(s)
    if m is None:
        raise ValueError(f"不支持的 bar_interval: {value!r}")
    num = int(m.group(1))
    unit = m.group(2).lower()
    return f"{num}{unit}"


def bar_interval_to_minutes(value: str | int) -> int:
    tag = normalize_bar_interval(value)
    m = _INTERVAL_RE.match(tag)
    assert m is not None
    num = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return num
    if unit == "h":
        return num * 60
    if unit == "d":
        return num * 24 * 60
    raise ValueError(f"不支持的 bar_interval: {value!r}")
