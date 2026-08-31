# -*- coding: utf-8 -*-
"""迷你 query DSL: get_fundamentals(query(valuation.x, income.y).filter(...))。

支持聚宽常用子集:
- 列: valuation(code/market_cap/circulating_market_cap/turnover_ratio/
      pe_ratio/pb_ratio)
      income(np_parent_company_owners/net_profit/operating_revenue/end_date)
- 过滤: ==/!=/</<=/>/>=, in_(), between(), &(与), |(或), ~(非)
- 排序: order_by(col.asc() / col.desc()); 截断: limit(n)
单位: market_cap 亿元(与聚宽一致), pe_ratio/pb_ratio 倍(与聚宽一致),
财务字段元(与聚宽一致)。

_Expr 携带其引用列集合(_Query.columns), 供 finance.run_query 等通用表查询
判定数据源。
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


class _Expr:
    def __init__(self, fn: Callable[[pd.DataFrame], Any],
                 columns: tuple[str, ...] = ()):
        self.fn = fn
        self.columns = columns          # 本表达式引用的列名(JQ key)

    def __and__(self, other):
        return _Expr(lambda df: self.fn(df) & other.fn(df),
                     self.columns + other.columns)

    def __or__(self, other):
        return _Expr(lambda df: self.fn(df) | other.fn(df),
                     self.columns + other.columns)

    def __invert__(self):
        return _Expr(lambda df: ~self.fn(df), self.columns)


_MISSING_WARNED: set[str] = set()


def norm_code(s) -> str:
    """聚宽风格代码 -> 6 位裸码: '601988.XSHG'/'601988' -> '601988'。"""
    return str(s).split(".")[0].strip().zfill(6)


def _norm_codes(seq):
    """in_ 序列归一化: 形如证券代码的元素取 6 位裸码, 其余原样。"""
    out = []
    for x in seq:
        t = str(x).strip()
        if "." in t and t.split(".", 1)[0].isdigit():
            t = t.split(".", 1)[0]
        out.append(t.zfill(6) if t.isdigit() else t)
    return out


class _Col:
    def __init__(self, key: str):
        self.key = key

    def _s(self, df: pd.DataFrame) -> pd.Series:
        if self.key in df.columns:
            return df[self.key]
        if self.key not in _MISSING_WARNED:
            _MISSING_WARNED.add(self.key)
            print(f"[query] 未支持的列 {self.key}, 相关过滤条件恒为 False",
                  flush=True)
        return pd.Series(np.nan, index=df.index)

    def _e(self, fn: Callable[[pd.DataFrame], Any]) -> _Expr:
        return _Expr(fn, (self.key,))

    def __gt__(self, v):
        return self._e(lambda df: self._s(df) > v)

    def __lt__(self, v):
        return self._e(lambda df: self._s(df) < v)

    def __ge__(self, v):
        return self._e(lambda df: self._s(df) >= v)

    def __le__(self, v):
        return self._e(lambda df: self._s(df) <= v)

    def __eq__(self, v):  # type: ignore[override]
        return self._e(lambda df: self._s(df) == v)

    def __ne__(self, v):  # type: ignore[override]
        return self._e(lambda df: self._s(df) != v)

    def in_(self, seq):
        return self._e(lambda df: self._s(df).isin(_norm_codes(seq)))

    def between(self, a, b):
        return self._e(lambda df: self._s(df).between(a, b, inclusive="both"))

    def asc(self):
        return ("asc", self)

    def desc(self):
        return ("desc", self)


class valuation:
    code = _Col("__code__")
    market_cap = _Col("market_cap")                     # 亿元
    circulating_market_cap = _Col("circulating_market_cap")
    turnover_ratio = _Col("turnover")
    pe_ratio = _Col("pe_ratio")                         # 市盈率(倍, 年化归母净利口径)
    pb_ratio = _Col("pb_ratio")                         # 市净率(倍, 归母净资产口径)


class income:
    np_parent_company_owners = _Col("np_parent_company_owners")  # 元
    net_profit = _Col("net_profit")                                # 元
    operating_revenue = _Col("operating_revenue")                  # 元
    end_date = _Col("income_end_date")


class _Query:
    def __init__(self, cols):
        self.cols = list(cols)
        self.exprs: list[_Expr] = []
        self.order: tuple | None = None
        self._limit: int | None = None

    def filter(self, *exprs):
        self.exprs.extend(exprs)
        return self

    def order_by(self, key):
        self.order = key
        return self

    def limit(self, n):
        self._limit = int(n)
        return self

    @property
    def columns(self) -> tuple[str, ...]:
        """本查询引用的全部列名(选列 + 过滤条件)。"""
        out = [c.key if isinstance(c, _Col) else str(c) for c in self.cols]
        for expr in self.exprs:
            out.extend(expr.columns)
        return tuple(out)


def query(*cols) -> _Query:
    return _Query(cols)
