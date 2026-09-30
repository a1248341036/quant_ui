#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子表达式的算子树解析、别名修复与构造模板兜底（供 extract_factor_records 使用）。

- `parse_op_tree(expr)`：把本仓 DSL 表达式解析成缩进算子树 + 用到的算子集合
- `apply_alias(expr)`：把模型常见别名写法改写为本仓算子名（单一真源 ALIAS）
- `try_construct(name, definition)`：按关键词匹配"研报常见构造"的组合表达式模板
"""
from __future__ import annotations

import re
from functools import lru_cache

# 别名：模型常见写法 → 本仓算子名
ALIAS: dict[str, str] = {
    "SUM": "TS_SUM", "MEAN": "TS_MEAN", "STD": "TS_STD", "VAR": "TS_VAR",
    "MEDIAN": "TS_MEDIAN", "MAX": "TS_MAX", "MIN": "TS_MIN", "CORR": "TS_CORR",
    "COV": "TS_COV", "SKEW": "TS_SKEW", "KURT": "TS_KURT", "QUANTILE": "TS_QUANTILE",
    "PCTCHANGE": "TS_PCTCHANGE", "RETURN": "TS_PCTCHANGE", "RANKX": "RANK",
    "ZSCORE": "CS_ZSCORE", "NEUTRALIZE": "CS_NEUTRALIZE", "WINSORIZE": "CS_WINSORIZE",
    "BETA": "REGBETA", "REGRESSION": "REGRESI", "RESIDUAL": "RESI", "TREND": "SLOPE",
    "EWM": "EMA", "MOVINGAVG": "SMA", "ROLLINGMEAN": "TS_MEAN",
}

# 构造模板：研报常见但需组合表达的结构（关键词正则 → 表达式）
CONSTRUCTS: list[tuple[str, str]] = [
    ("illiq|amihud|非流动性",
     "TS_MEAN(DIVIDE(ABS(TS_PCTCHANGE($adj_close, 1)), $amount), 20)"),
    ("cgo|处置效应|参考成本|cost_basis",
     "DIVIDE(SUBTRACT($adj_close, DIVIDE(TS_SUM(MULTIPLY($adj_close, $volume), 60), "
     "TS_SUM($volume, 60))), $adj_close)"),
    ("i?vr?ff|rff|特异度|特质波动|残差波动|idiosyncratic|ff三因子|ff_regression",
     "TS_STD(RESI($adj_close, CS_NEUTRALIZE($adj_close)), 20)"),
    ("换手率|turnover",
     "TS_MEAN($turnover_rate, 20)"),
]


# 字段别名：研报/模型口径 → 本仓面板字段（只收语义明确等价的；口径不同的不换）
FIELD_ALIAS: dict[str, str] = {
    "dividend_ttm": "dv_ttm", "dividend_yield": "dv_ttm", "dv_ratio": "dv_ttm",
    "cash_dividend": "div_cash_div", "cash_div": "div_cash_div",
    "days_to_ex": "div_days_to_ex",
    "fsi_operate_profit": "funda_operate_profit", "operate_profit": "funda_operate_profit",
    "net_profit": "funda_net_profit", "netprofit": "funda_net_profit",
    "total_revenue": "funda_total_revenue", "revenue": "funda_total_revenue",
    "total_equity": "funda_total_equity", "equity": "funda_total_equity",
    "total_assets": "funda_total_assets", "assets": "funda_total_assets",
    "roe": "funda_roe", "roa": "funda_roa", "eps": "funda_eps",
    "total_mv": "tot_cap", "mktcap": "tot_cap", "market_cap": "tot_cap",
    "circ_mv": "float_cap", "float_mv": "float_cap",
}


@lru_cache(maxsize=8192)
def apply_field_alias(expr: str) -> tuple[str, tuple[str, ...]]:
    """把字段别名改写为本仓字段名（避免"名字不同"被误判为缺字段）。"""
    notes: list[str] = []
    out = expr
    for bad, good in FIELD_ALIAS.items():
        pat = "\\$" + bad + "\\b"
        if re.search(pat, out):
            out = re.sub(pat, "$" + good, out)
            notes.append(bad + "->" + good)
    return out, tuple(notes)


@lru_cache(maxsize=8192)
def apply_alias(expr: str) -> tuple[str, tuple[str, ...]]:
    """把别名写法改写为本仓算子名，返回 (新表达式, 改写记录)。"""
    notes: list[str] = []
    out = expr
    for bad, good in ALIAS.items():
        pat = r"\b" + bad + r"\s*\("
        if re.search(pat, out):
            out = re.sub(pat, good + "(", out)
            notes.append(bad + "->" + good)
    return out, tuple(notes)


def try_construct(name: str, definition: str = "") -> tuple[str | None, str | None]:
    """按因子名/原文定义匹配构造模板（组合兜底）。"""
    text = (str(name) + " " + str(definition or "")).lower()
    for kw, tmpl in CONSTRUCTS:
        if re.search(kw, text):
            return tmpl, kw.split("|")[0]
    return None, None


@lru_cache(maxsize=8192)
def parse_op_tree(expr: str) -> tuple[str, tuple[str, ...]]:
    """解析表达式为算子树（缩进文本）+ 用到的算子列表。"""
    ops_used: list[str] = []
    n = len(expr)

    def skip(j: int) -> int:
        while j < n and expr[j] in " \t":
            j += 1
        return j

    def parse(j: int, depth: int) -> tuple[str, int]:
        j = skip(j)
        start = j
        if j < n and expr[j] == "$":          # 字段引用（$field）
            j += 1
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
        elif j < n and (expr[j].isdigit() or expr[j] == "."):   # 数字字面量
            while j < n and (expr[j].isdigit() or expr[j] in "._"):
                j += 1
        else:
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
        tok = expr[start:j]
        if j <= start:                        # 保证推进，避免死循环
            j = start + 1
            tok = expr[start:j]
        k = skip(j)
        if k < n and expr[k] == "(":
            ops_used.append(tok)
            lines = ["  " * depth + tok + "("]
            j = k + 1
            while j < n and expr[j] != ")":
                sub, j = parse(j, depth + 1)
                lines.append(sub)
                j = skip(j)
                if j < n and expr[j] == ",":
                    j += 1
            lines.append("  " * depth + ")")
            return "\n".join(lines), min(j + 1, n)
        return "  " * depth + (tok or "?"), j

    tree, _ = parse(0, 0)
    return tree, tuple(ops_used)
