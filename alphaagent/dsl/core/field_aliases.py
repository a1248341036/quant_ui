# -*- coding: utf-8 -*-
"""模块 · field_aliases：DSL 字段别名（提示词/题库用名 → 真实面板列）。

背景（2026-09-27 整夜 run 实测）：提示词与研报题库会写 ``$turnover``，但 CNE 面板
列名是 ``turnover_rate``；模型照提示书写就报「表达式引用了不可用字段」，只能退化成
``DIVIDE($amount, $float_cap)`` 代理。别名表把这类"自然写法"收敛到真实列，
**校验、编译与"字段是否可用"判定共用同一张表**，避免多处双写漂移。

约定：
- 别名只在**目标列确实存在**时生效——面板没载入该列族时仍照旧报「不可用字段」，
  绝不静默指向不存在的列；
- 改写只发生在编译期（不改持久表达式），历史因子重放走同一别名，语义等价；
- 改写带右边界（``(?![A-Za-z0-9_])``），``$turnover_rate`` 不会被 ``turnover`` 误伤。
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

# 别名（提示词/题库/schema 里出现的写法）→ 真实面板列名
FIELD_ALIASES: dict[str, str] = {
    "turnover": "turnover_rate",
}

_DOLLAR_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)(?:@[A-Za-z0-9_]+)?")


def bare_column(name: str) -> str:
    """列/字段名归一：去 ``$`` 前缀与 ``@频率`` 后缀（别名**不**在此步展开）。"""
    return str(name or "").lstrip("$").split("@", 1)[0].strip()


def canonical_column(name: str) -> str:
    """别名归一：``turnover`` → ``turnover_rate``；未登记的原样返回。"""
    return FIELD_ALIASES.get(bare_column(name), bare_column(name))


def available_columns(columns: Iterable[str] | None) -> set[str]:
    """把面板列（可带 ``$`` 前缀）归一成裸列名集合。"""
    if not columns:
        return set()
    return {bare_column(col) for col in columns if str(col).strip()}


def apply_field_aliases(
    expr: str, available: Iterable[str] | None
) -> tuple[str, tuple[str, ...]]:
    """把表达式中「目标列可用」的别名引用改写为真实列名。

    Args:
        expr: DSL 表达式（多行亦可）。
        available: 本次实际可用的面板列（裸名或带 ``$`` 均可）。

    Returns:
        ``(改写后表达式, 命中的别名元组)``；未命中时原样返回。
    """
    avail = available_columns(available)
    applied: list[str] = []
    out = str(expr or "")
    for alias, target in FIELD_ALIASES.items():
        if target not in avail:
            continue
        pattern = re.compile(r"\$" + re.escape(alias) + r"(?![A-Za-z0-9_])")
        out, n = pattern.subn("$" + target, out)
        if n:
            applied.append(alias)
    return out, tuple(applied)


def missing_fields(
    expr: str, available: Iterable[str] | None
) -> tuple[str, ...]:
    """列出表达式引用但面板未载入的字段（按别名归一后比较，去重且稳定排序）。

    ``available`` 为空（None/[]）表示**未提供可用列信息**，此时不判定、返回空元组，
    由调用方决定旧行为。
    """
    avail = available_columns(available)
    if not avail:
        return ()
    seen: set[str] = set()
    for match in _DOLLAR_REF_RE.finditer(str(expr or "")):
        raw = match.group(1)
        canon = FIELD_ALIASES.get(raw, raw)
        if canon in avail:
            continue
        if raw in avail:  # 别名原名本身也是真实列（如同时存在 turnover 与 turnover_rate）
            continue
        seen.add(raw)
    return tuple(sorted(seen))


def fields_available(expr: str, available: Iterable[str] | None) -> bool:
    """表达式引用的字段是否全部可用（``available`` 为空时不判定 → True）。"""
    return not missing_fields(expr, available)
