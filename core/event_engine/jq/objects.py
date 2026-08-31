# -*- coding: utf-8 -*-
"""壳对象: 聚宽策略代码看到的 context/portfolio/positions/current_data 视图。

全部是只读视图, 底层状态在事件引擎 ctx + JQRuntime 的成本表里。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class _G:
    """聚宽 g 全局对象。"""


class _Log:
    def __init__(self):
        self.buffer: list[str] = []

    def _add(self, level, msg):
        line = f"[{level}] {msg}"
        self.buffer.append(line)
        if level in ("warn", "error"):
            print(line, flush=True)

    def info(self, msg):
        self._add("info", msg)

    def debug(self, msg):
        self._add("debug", msg)

    def warn(self, msg):
        self._add("warn", msg)

    warning = warn

    def error(self, msg):
        self._add("error", msg)

    def set_level(self, *a, **k):
        pass


class _OrderStatus:
    """聚宽 OrderStatus 子集(字符串枚举, == 比较即可用)."""

    created = "created"
    opened = "opened"
    held = "held"
    canceled = "canceled"
    rejected = "rejected"


class _Order:
    """下单回执(日线引擎: 提交时未成交, T+1 开盘撮合)."""

    def __init__(self, security, amount, filled=0.0, price=0.0,
                 status=_OrderStatus.opened, is_buy=True, add_time=None):
        self.security = security
        self.amount = float(amount)
        self.filled = float(filled)
        self.price = float(price)
        self.status = status
        self.is_buy = bool(is_buy)
        self.add_time = add_time
        self.avg_cost = float(price)
        self.side = "buy" if is_buy else "sell"


class _OrderCost:
    """聚宽 OrderCost 桩: 费率由运行参数控制, 这里仅承接参数不报错。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FixedSlippage:
    def __init__(self, v=0.0):
        self.value = v


class _CodeData:
    """get_current_data()[code] 的视图。is_st 兼容名字带 ST 但标记缺失的股票。

    日线口径: last_price 取当日开盘(盘中"现价"的无泄漏代理)。
    """

    def __init__(self, row: pd.Series, name_map: dict):
        self.paused = bool(row.get("paused", False))
        nm = name_map.get(row.name if hasattr(row, "name") else "", "")
        self.is_st = bool(row.get("st", False)) or ("ST" in nm.upper())
        self.name = nm
        self.high_limit = row.get("limit_price", np.nan)
        self.low_limit = row.get("low_limit", np.nan)
        self.day_open = row.get("open_raw", np.nan)
        op = self.day_open
        self.last_price = (float(op) if op is not None and np.isfinite(op)
                           else row.get("close_raw", np.nan))
        self.last_price_closed = row.get("close_raw", np.nan)


class _CurrentData:
    def __init__(self, snap: pd.DataFrame, name_map: dict):
        self._snap = snap
        self._name_map = name_map

    def __getitem__(self, code):
        if code not in self._snap.index:
            # 停牌/无数据股票: 返回"不可交易"视图
            return _CodeData(pd.Series({"paused": True, "st": False}),
                             self._name_map)
        return _CodeData(self._snap.loc[code], self._name_map)

    def get(self, code, default=None):
        return self[code] if code in self._snap.index else default


class _Position:
    def __init__(self, code, shares, avg_cost, price):
        self.security = code
        self.total_amount = float(shares)
        self.avg_cost = float(avg_cost)
        self.price = float(price)
        self.value = float(shares) * float(price)

    @property
    def closable(self):
        return self.total_amount > 0


class _PositionsDict(dict):
    """聚宽语义: 访问未持仓代码返回空仓视图(临时对象, 不落库)."""

    def __missing__(self, key):
        return _Position(key, 0, 0.0, 0.0)


class _Portfolio:
    def __init__(self, rt):
        self._rt = rt

    @property
    def cash(self):
        return self._rt._engine_ctx.cash

    @property
    def total_value(self):
        return self._rt._engine_ctx.portfolio_value or self._rt.capital

    @property
    def positions_value(self):
        ctx = self._rt._engine_ctx
        return sum(ctx.position_value(c) for c in ctx.positions)

    @property
    def positions(self):
        ctx = self._rt._engine_ctx
        out = _PositionsDict()
        for code, shares in ctx.positions.items():
            if shares:
                px = ctx.last_close(code) or 0.0
                out[code] = _Position(code, shares,
                                      self._rt._cost.get(code, 0.0), px)
        return out


class _Context:
    def __init__(self, rt):
        self._rt = rt
        self.portfolio = _Portfolio(rt)
        self.current_dt: pd.Timestamp = pd.Timestamp.today()
        self.previous_date: pd.Timestamp = pd.Timestamp.today()

    @property
    def run_params(self):
        return {"capital": self._rt.capital}


class _SecurityInfo:
    def __init__(self, code, name, start_date):
        self.code = code
        self.display_name = name
        self.start_date = (start_date if start_date is not None
                           else pd.Timestamp("1990-01-01"))
