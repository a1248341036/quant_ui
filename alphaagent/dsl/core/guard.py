"""DSL 防未来函数守卫（列黑名单 + 时序算子负参数拦截）。

背景：LLM 生成的因子表达式可能无意引用收益标签列（``$label_*``），或给时序算子传
负窗口参数（如 ``TS_DELTA($close, -1)`` / ``DELAY($close, -1)`` 等价于引用 t+1 的
未来值），造成前视泄漏。本模块在 ``eval_multi_line_factor`` 求值入口统一拦截：

- **列黑名单**（源码扫描）：默认禁止 ``label_`` 前缀列作为因子输入。
  ``ALPHA_DSL_BLOCKED_COL_PREFIXES`` 可覆盖（逗号分隔，整体替换默认值）。
- **负参数守卫**（运行时包装）：对时序/滚动算子族（TS_*、DELAY/DELTA/SMA/EMA/WMA、
  SLOPE/RESI/REGBETA/REGRESI、CHIP_*/CROWD_*/PRICE_GAP_*、KLINE_GEOMETRY、
  WICK_EFFICIENCY、VOLUME_CLOCK_VPIN、MUTUAL_INFO_LAG）拦截负整数参数。
  算术/比较/CS_* 等负数合法的算子显式豁免（``_NEG_INT_EXEMPT``）；
  新增算子默认受保护，需要豁免时显式加入该集合。
- 只拦截**整数**负值（含 np.integer）：负浮点阈值（如 ``LT(x, -0.5)``）不受影响。

``ALPHA_DSL_GUARD=0`` 可整体关闭（仅诊断用，生产建议保持开启）。
"""
from __future__ import annotations

import functools
import os
import re
from typing import Any, Mapping, Optional, Tuple

import numpy as np

GUARD_DISABLE_ENV = "ALPHA_DSL_GUARD"
BLOCKED_PREFIXES_ENV = "ALPHA_DSL_BLOCKED_COL_PREFIXES"

_DEFAULT_BLOCKED_PREFIXES: Tuple[str, ...] = ("label_",)

# ``$name`` / ``$name@freq`` 引用（与 eval.py 的提取规则一致）
_DOLLAR_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)(?:@[A-Za-z0-9_]+)?")

# 负整数参数合法、不做守卫的算子（算术/比较/逻辑/截面等）。
# 注意是「豁免白名单」：新算子默认被守卫，出现误报时再显式加入。
_NEG_INT_EXEMPT = frozenset(
    {
        # 四则与逐元素数学（负标量合法，如 MULTIPLY(x, -1)、POW(x, -2)）
        "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "POW",
        "ABS", "SIGN", "NEG", "INV", "LOG", "EXP", "SQRT", "CAST",
        "MAX", "MIN", "MAXIMUM", "MINIMUM", "MEAN", "STD",
        # 比较/逻辑/条件（阈值可为负，如 LT(x, -1)）
        "EQ", "NE", "GT", "GE", "LT", "LE", "AND", "OR",
        "IF_THEN_ELSE", "GATED_SIGNAL", "PIECEWISE_STATE",
        # 排名/截面（无时间维度，负参数不会前视）
        "RANK", "DIVERGENCE_RANK", "SEQUENCE",
        "CS_BUCKET", "CS_DEMEAN", "CS_GROUP_RANK", "CS_NEUTRALIZE",
        "CS_RESIDUALIZE", "CS_WINSORIZE", "CS_ZSCORE",
        # 其他负数合法的工具
        "FILLNA",
    }
)


def guard_enabled() -> bool:
    return os.environ.get(GUARD_DISABLE_ENV, "1").strip().lower() not in {
        "0", "false", "off", "no", "",
    }


def blocked_column_prefixes() -> Tuple[str, ...]:
    raw = os.environ.get(BLOCKED_PREFIXES_ENV)
    if raw is None:
        return _DEFAULT_BLOCKED_PREFIXES
    items = tuple(p.strip() for p in raw.split(",") if p.strip())
    return items or _DEFAULT_BLOCKED_PREFIXES


def _strip_string_literals(s: str) -> str:
    out = re.sub(r"'[^']*'", "", s)
    out = re.sub(r'"[^"]*"', "", s)
    return out


def find_blocked_columns(expr: str) -> list[str]:
    """返回表达式中引用的被禁列名（如 ``label_*`` 前缀列），保序去重。"""
    if not expr:
        return []
    prefixes = blocked_column_prefixes()
    seen: set[str] = set()
    out: list[str] = []
    for m in _DOLLAR_REF_RE.finditer(_strip_string_literals(expr)):
        name = m.group(1)
        if name in seen:
            continue
        if any(name.startswith(p) for p in prefixes):
            seen.add(name)
            out.append(name)
    return out


def wrap_lookahead_guard(ns: Mapping[str, Any]) -> dict[str, Any]:
    """对命名空间做负参数守卫包装：豁免集以外的算子拦截负整数实参。

    与算子耗时监控包装（monitor.wrap_operator_namespace）正交：
    守卫在外层先做廉价的参数检查，再把调用透传给（可能被监控包装的）原算子。
    """
    wrapped: dict[str, Any] = {}
    for name, fn in ns.items():
        if (
            isinstance(name, str)
            and name[:1].isupper()
            and name not in _NEG_INT_EXEMPT
            and callable(fn)
        ):
            wrapped[name] = _make_guarded(name, fn)
        else:
            wrapped[name] = fn
    return wrapped


def _make_guarded(name: str, fn):
    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        bad = _find_negative_int_arg(args, kwargs)
        if bad is not None:
            key, value = bad
            raise ValueError(
                f"{name} 参数 {key}={value} 为负数：时序/滚动算子的窗口、滞后、位移参数"
                f"禁止为负（防未来函数）。如需反向取值请用负乘法（如 MULTIPLY(x, -1)）。"
            )
        return fn(*args, **kwargs)

    return guarded


def _find_negative_int_arg(args: tuple, kwargs: Mapping[str, Any]) -> Optional[Tuple[Any, int]]:
    for i, v in enumerate(args, start=1):
        if _is_negative_int(v):
            return i, v
    for k, v in kwargs.items():
        if _is_negative_int(v):
            return k, v
    return None


def _is_negative_int(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, np.integer)) and int(v) < 0
