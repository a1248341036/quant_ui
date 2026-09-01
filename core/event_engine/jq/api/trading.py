# -*- coding: utf-8 -*-
"""交易函数(聚宽文档类别): order 系列 + 订单/成交查询 + 撤单。

撮合口径: 用户函数运行期间订单进入 pending 队列(回执 status=opened),
函数跑完后统一 flush 到事件引擎, 按下单时点分钟价撮合(T+1 语义);
flush 后回执标记 held(受理), 逐笔成交明细经 get_trades() 查询。

style 支持: MarketOrderStyle(limit_price)/LimitOrderStyle(limit_price)——
limit_price 作为保护价传引擎(买入不高于/卖出不低于该价才成交)。
"""
from __future__ import annotations

from core.event_engine.jq.objects import (_LimitOrderStyle, _MarketOrderStyle,
                                          _OrderStatus)


def install(ns: dict, rt) -> None:
    def _style_limit(style):
        lp = getattr(style, "limit_price", None) if style is not None else None
        return (float(lp) if lp is not None and lp == lp else None)

    def _mk_and_enqueue(kind, code, value=None, shares=None,
                        limit_price=None):
        code = str(code).zfill(6)
        o = rt._mk_order(code, value=value, shares=shares)
        if o is not None:
            arg = float(value if value is not None else shares)
            o.limit_price = limit_price
            rt.pending_orders.append((kind, code, arg, o, limit_price))
        return o

    def order_target_value(code, value, style=None):
        return _mk_and_enqueue("tv", code, value=float(value),
                               limit_price=_style_limit(style))

    def order_value(code, value, style=None):
        return _mk_and_enqueue("ov", code, value=float(value),
                               limit_price=_style_limit(style))

    def order_shares(code, shares, style=None):
        return _mk_and_enqueue("os", code, shares=float(shares),
                               limit_price=_style_limit(style))

    def order(code, amount, style=None):
        return order_shares(code, amount, style=style)

    def order_target(code, amount, style=None):
        return _mk_and_enqueue("ot", code, shares=float(amount),
                               limit_price=_style_limit(style))

    def order_target_percent(code, pct, style=None):
        # 回执按市值估算, 排队存 pct(引擎按占比撮合, flush 端不再换算)
        code = str(code).zfill(6)
        pv = (rt._engine_ctx.portfolio_value
              if rt._engine_ctx is not None else None) or rt.capital
        o = rt._mk_order(code, value=float(pct) * float(pv))
        if o is not None:
            lp = _style_limit(style)
            o.limit_price = lp
            rt.pending_orders.append(("op", code, float(pct), o, lp))
        return o

    def cancel_order(order):
        if order is None:
            return None
        for i, entry in enumerate(list(rt.pending_orders)):
            if entry[3] is order:
                rt.pending_orders.pop(i)
                order.status = _OrderStatus.canceled
                break
        return order

    def get_orders():
        # 当日全部订单 {order_id: Order}
        return dict(rt._day_orders)

    def get_open_orders():
        # 未成交(尚未 flush)订单
        return {oid: o for oid, o in rt._day_orders.items()
                if o.status == _OrderStatus.opened}

    def get_trades():
        # 当日成交(引擎撮合在 run_day 后执行, 此处为上一执行日开盘成交)
        return list(rt._day_trades)

    ns.update({
        "order": order,
        "order_target": order_target,
        "order_value": order_value,
        "order_target_value": order_target_value,
        "order_shares": order_shares,
        "order_target_percent": order_target_percent,
        "MarketOrderStyle": _MarketOrderStyle,
        "LimitOrderStyle": _LimitOrderStyle,
        "cancel_order": cancel_order,
        "get_orders": get_orders,
        "get_open_orders": get_open_orders,
        "get_trades": get_trades,
    })
