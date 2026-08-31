# -*- coding: utf-8 -*-
"""策略设置函数(聚宽文档类别): set_option/set_benchmark/set_universe/
set_slippage/set_order_cost + FixedSlippage/PriceRelatedSlippage/OrderCost。

费率语义:
- set_order_cost 把 open/close 佣金+税费折算为引擎 buy_cost/sell_cost;
  min_commission(每笔最低佣金, 元)传入引擎, 单笔费用 = max(金额×费率, min_commission)
- set_slippage(PriceRelatedSlippage(r)): JQ 买卖各偏 r/2 -> 引擎单边 bps=r/2*1e4
- set_slippage(FixedSlippage(v)): JQ 为绝对价差(元); 引擎仅支持万分比,
  按社区惯例(v=3/10000 意图即 3bp)作比例解释
"""
from __future__ import annotations

from core.event_engine.jq.objects import (_FixedSlippage, _OrderCost,
                                          _PriceRelatedSlippage)


def install(ns: dict, rt) -> None:
    def set_option(option, value=True, **kwargs):
        # use_real_price/avoid_future_data 本引擎恒成立; order_volume_ratio
        # 近似对应引擎 max_participation(20日均额口径), 运行参数侧控制
        return None

    def set_benchmark(security, **kwargs):
        # 基准仅作记录; CNE index_bars 齐备, 超额输出由回测结果侧计算
        rt.benchmark = security
        return None

    def set_universe(universe, **kwargs):
        return None                       # 旧 API, 兼容保留

    def set_slippage(s, **kwargs):
        if isinstance(s, _PriceRelatedSlippage):
            # JQ: 比例滑点, 买卖各偏 value/2 -> 引擎单边 bps = value/2*1e4
            rt.cost_cfg["slippage_bps"] = float(s.value) / 2 * 1e4
            return
        v = getattr(s, "value", s if isinstance(s, (int, float)) else None)
        if v:
            # FixedSlippage 聚宽语义为绝对价差(元); 引擎仅支持万分比,
            # 按社区惯例(v=3/10000 意图即 3bp)作比例解释
            rt.cost_cfg["slippage_bps"] = float(v) * 1e4

    def set_order_cost(cost, type="stock", **kwargs):
        k = getattr(cost, "kwargs", None) or {}
        try:
            oc = (float(k.get("open_commission") or 0)
                  + float(k.get("open_tax") or 0))
            sc = (float(k.get("close_commission") or 0)
                  + float(k.get("close_tax") or 0))
            if oc > 0:
                rt.cost_cfg["buy_cost"] = oc
            if sc > 0:
                rt.cost_cfg["sell_cost"] = sc
            mc = k.get("min_commission")
            if mc:
                rt.cost_cfg["min_commission"] = float(mc)
        except (TypeError, ValueError):
            pass

    ns.update({
        "set_option": set_option,
        "set_benchmark": set_benchmark,
        "set_universe": set_universe,
        "set_slippage": set_slippage,
        "set_order_cost": set_order_cost,
        "FixedSlippage": _FixedSlippage,
        "PriceRelatedSlippage": _PriceRelatedSlippage,
        "OrderCost": _OrderCost,
    })
