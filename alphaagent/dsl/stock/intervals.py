"""股票 DSL 周期归一化：仅支持日频 @1d 与日历周 @1w。"""

from __future__ import annotations

import re

import pandas as pd

# 1d / 1w 及常见别名
_STOCK_INTERVAL_RE = re.compile(
    r"^\s*(\d+)\s*(d|w|day|week)(?:s)?\s*$",
    re.IGNORECASE,
)

# 股票模式允许的归一化周期
STOCK_INTERVALS = frozenset({"1d", "1w"})


def normalize_bar_interval(value: str | int) -> str:
    """把 ``1d`` / ``1D`` / ``1week`` 等归一为 ``1d`` 或 ``1w``。"""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"bar_interval 必须为正整数，收到: {value!r}")
        # 整数仅用于分钟期货语义；股票模式不支持
        raise ValueError(f"股票 DSL 不支持整数周期 {value!r}，请使用 1d 或 1w")

    s = str(value).strip().lower()
    if not s:
        raise ValueError("bar_interval 不能为空")

    # 精确匹配已归一形式
    if s in STOCK_INTERVALS:
        return s

    m = _STOCK_INTERVAL_RE.match(s)
    if m is None:
        raise ValueError(
            f"不支持的 bar_interval: {value!r}（股票仅支持 1d / 1w）"
        )

    num = int(m.group(1))
    unit = m.group(2).lower()
    if unit in ("d", "day"):
        tag = f"{num}d"
    elif unit in ("w", "week"):
        tag = f"{num}w"
    else:
        raise ValueError(f"不支持的 bar_interval: {value!r}")

    if tag not in STOCK_INTERVALS:
        raise ValueError(f"股票 DSL 仅支持 1d / 1w，收到: {tag!r}")
    return tag


def bar_interval_to_timedelta(value: str | int) -> pd.Timedelta:
    """辅周期 bar 全长，用于无前视广播的「完成时刻 = 桶起点 + 全长」。"""
    tag = normalize_bar_interval(value)
    if tag == "1d":
        return pd.Timedelta(days=1)
    if tag == "1w":
        # 日历周：W-FRI 桶起点 + 7 天 = 该周 bar 完成
        return pd.Timedelta(days=7)
    raise ValueError(f"未知周期: {tag!r}")
