# -*- coding: utf-8 -*-
"""迷你 query DSL: get_fundamentals(query(valuation.x, income.y).filter(...))。

支持聚宽常用子集:
- 列: valuation(code/market_cap/circulating_market_cap/turnover_ratio)
      income(np_parent_company_owners/net_profit/operating_revenue/end_date)
- 过滤: ==/!=/</<=/>/>=, in_(), between(), &(与), |(或), ~(非)
- 排序: order_by(col.asc() / col.desc()); 截断: limit(n)
单位: market_cap 亿元(与聚宽一致), 财务字段元(与聚宽一致)。
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


class _Expr:
    def __init__(self, fn: Callable[[pd.DataFrame], Any]):
        self.fn = fn

    def __and__(self, other):
        return _Expr(lambda df: self.fn(df) & other.fn(df))

    def __or__(self, other):
        return _Expr(lambda df: self.fn(df) | other.fn(df))

    def __invert__(self):
        return _Expr(lambda df: ~self.fn(df))


_MISSING_WARNED: set[str] = set()


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

    def __gt__(self, v):
        return _Expr(lambda df: self._s(df) > v)

    def __lt__(self, v):
        return _Expr(lambda df: self._s(df) < v)

    def __ge__(self, v):
        return _Expr(lambda df: self._s(df) >= v)

    def __le__(self, v):
        return _Expr(lambda df: self._s(df) <= v)

    def __eq__(self, v):  # type: ignore[override]
        return _Expr(lambda df: self._s(df) == v)

    def __ne__(self, v):  # type: ignore[override]
        return _Expr(lambda df: self._s(df) != v)

    def in_(self, seq):
        return _Expr(lambda df: self._s(df).isin(list(seq)))

    def between(self, a, b):
        return _Expr(lambda df: self._s(df).between(a, b, inclusive="both"))

    def asc(self):
        return ("asc", self)

    def desc(self):
        return ("desc", self)


class valuation:
    code = _Col("__code__")
    market_cap = _Col("market_cap")                     # 亿元
    circulating_market_cap = _Col("circulating_market_cap")
    turnover_ratio = _Col("turnover")


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


def query(*cols) -> _Query:
    return _Query(cols)
