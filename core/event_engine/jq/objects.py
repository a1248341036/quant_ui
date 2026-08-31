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

    def _add(self, level, *args):
        # JQ log.info('a', b, c) 多参数空格拼接(DataFrame 等对象 str 化)
        msg = " ".join(str(a) for a in args)
        line = f"[{level}] {msg}"
        self.buffer.append(line)
        if level in ("warn", "error"):
            print(line, flush=True)

    def info(self, *args):
        self._add("info", *args)

    def debug(self, *args):
        self._add("debug", *args)

    def warn(self, *args):
        self._add("warn", *args)

    warning = warn

    def error(self, *args):
        self._add("error", *args)

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
    """下单回执(日线引擎: 提交时未成交, flush 后按执行日开盘撮合)."""

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
        self.order_id: int | None = None


class _Trade:
    """聚宽 Trade 对象近似: 逐笔成交(amount 正=买 负=卖)。"""

    def __init__(self, security, price, amount, time=None, fee=0.0):
        self.security = security
        self.price = float(price)
        self.amount = float(amount)      # 带方向: 买正卖负
        self.time = time
        self.fee = float(fee)
        self.is_buy = self.amount > 0

    def __repr__(self):
        return (f"<Trade {self.security} {self.amount:+.0f}股 "
                f"@{self.price:.3f} {self.time}>")


class _OrderCost:
    """聚宽 OrderCost 桩: 费率由运行参数控制, 这里仅承接参数不报错。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FixedSlippage:
    """聚宽 FixedSlippage: 绝对价差(元)。引擎按万分比近似(见 set_slippage)。"""

    def __init__(self, v=0.0):
        self.value = v


class _PriceRelatedSlippage:
    """聚宽 PriceRelatedSlippage: 比例滑点, 买卖各偏 value/2。"""

    def __init__(self, v=0.00246):
        self.value = v


class _CodeData:
    """get_current_data()[code] / handle_data(data)[code] 的视图。

    is_st 兼容名字带 ST 但标记缺失的股票;
    日线口径: last_price 取当日开盘(盘中"现价"的无泄漏代理)。
    """

    def __init__(self, row: pd.Series, name_map: dict, rt=None):
        self._rt = rt
        self.code = (str(row.name) if hasattr(row, "name") else "")
        nm = name_map.get(self.code, "")
        self.paused = bool(row.get("paused", False))
        self.is_st = bool(row.get("st", False)) or ("ST" in nm.upper())
        self.name = nm
        self.high_limit = row.get("limit_price", np.nan)
        self.low_limit = row.get("low_limit", np.nan)
        self.day_open = row.get("open_raw", np.nan)
        op = self.day_open
        self.last_price = (float(op) if op is not None and np.isfinite(op)
                           else row.get("close_raw", np.nan))
        self.last_price_closed = row.get("close_raw", np.nan)
        # SecurityUnitData 兼容字段(快照缺列给 NaN)
        self.open = self.day_open
        self.close = self.last_price_closed
        self.high = row.get("high_raw", np.nan)
        self.low = row.get("low_raw", np.nan)
        self.volume = np.nan
        self.money = np.nan

    def mavg(self, n: int) -> float:
        """前 n 个交易日收盘均价(不含当日, 无未来泄漏)。"""
        if self._rt is None or not self.code:
            return np.nan
        h = self._rt.history(int(n), unit="1d", field="close",
                             security_list=[self.code]).get(self.code) or []
        vals = [v for v in h if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    def __getattr__(self, key):
        # vwap/paused 兼容等未知 SecurityUnitData 属性 -> NaN
        if key.startswith("_"):
            raise AttributeError(key)
        return np.nan


class _CurrentData:
    def __init__(self, snap: pd.DataFrame, name_map: dict, rt=None):
        self._snap = snap
        self._name_map = name_map
        self._rt = rt

    @staticmethod
    def _norm(code):
        # '601988.XSHG'/'601988' -> '601988' (用户常写聚宽风格后缀码)
        return str(code).split(".")[0].strip().zfill(6)

    def __getitem__(self, code):
        code = self._norm(code)
        if code not in self._snap.index:
            # 停牌/无数据股票: 返回"不可交易"视图
            return _CodeData(pd.Series({"paused": True, "st": False}),
                             self._name_map, self._rt)
        return _CodeData(self._snap.loc[code], self._name_map, self._rt)

    def get(self, code, default=None):
        code = self._norm(code)
        return self[code] if code in self._snap.index else default

    def __iter__(self):
        return iter(self._snap.index)

    def __len__(self):
        return len(self._snap)

    def __contains__(self, code):
        return self._norm(code) in self._snap.index


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

    @property
    def closeable_amount(self):
        # 引擎持仓在 T+1 开盘结算后才可见 -> 全部可卖
        return self.total_amount


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
    def available_cash(self):
        return self._rt._engine_ctx.cash

    @property
    def total_value(self):
        return self._rt._engine_ctx.portfolio_value or self._rt.capital

    @property
    def returns(self):
        cap = self._rt.capital or 1.0
        return self.total_value / cap - 1.0

    @property
    def locked_cash(self):
        return 0.0

    @property
    def in_out_cash(self):
        return 0.0

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
    def subportfolios(self):
        # 单账户兼容: id=0 主账户
        return [_SubPortfolio(self._rt)]

    @property
    def run_params(self):
        return {"capital": self._rt.capital,
                "type": "simple_backtest"}


class _SubPortfolio:
    def __init__(self, rt):
        self._rt = rt
        self.id = 0
        self.type = "stock"

    @property
    def inout_cash(self):
        return 0.0

    @property
    def available_cash(self):
        return self._rt._engine_ctx.cash

    @property
    def transferable_cash(self):
        return self._rt._engine_ctx.cash

    @property
    def locked_cash(self):
        return 0.0

    @property
    def positions(self):
        return self._rt.context.portfolio.positions

    @property
    def total_value(self):
        return self._rt._engine_ctx.portfolio_value or self._rt.capital


class _SecurityInfo:
    def __init__(self, code, name, start_date):
        self.code = code
        self.display_name = name
        self.start_date = (start_date if start_date is not None
                           else pd.Timestamp("1990-01-01"))
        self.end_date = pd.Timestamp("2200-01-01")
        self.type = "stock"
