# -*- coding: utf-8 -*-
"""策略设置函数(聚宽文档类别): set_option/set_benchmark/set_universe/
set_slippage/set_order_cost + FixedSlippage/PriceRelatedSlippage/OrderCost。

费率语义(与聚宽逐项一致, 经聚宽成交明细校验):
- set_order_cost: buy_cost/sell_cost = 佣金费率, buy_tax/sell_tax = 税率,
  min_commission 只作用于佣金单项: 单笔费用 = max(金额×佣金, 最低佣金)
  + 金额×税 (实证: 聚宽卖出 18198 元收 23.20 = 5 + 18.198)
- set_slippage(PriceRelatedSlippage(r)): JQ 买卖各偏 r/2 -> 引擎单边 bps=r*1e4/2
- set_slippage(FixedSlippage(v)): JQ 语义 = 绝对价差 v 元(买卖各偏 v/2),
  引擎侧 slippage_bps=0, 由运行时在 fill_price 上加绝对偏移
"""
from __future__ import annotations

from core.event_engine.jq.objects import (_FixedSlippage, _OrderCost,
                                          _PriceRelatedSlippage)


def install(ns: dict, rt) -> None:
    def set_option(option, value=True, **kwargs):
        # avoid_future_data: 真生效——数据 API 以 current_dt/previous_date 为界,
        # 显式请求未来数据时抛错(聚宽同语义: 检测到未来函数报错);
        # use_real_price 本引擎恒为真实价; order_volume_ratio 近似对应
        # 引擎 max_participation(20日均额口径), 运行参数侧控制
        if str(option) == "avoid_future_data":
            rt.avoid_future_data = bool(value)
            rt.log.info(f"[runtime] avoid_future_data = {bool(value)} "
                        f"(数据 API 将拒绝未来日期请求)")
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
            rt.cost_cfg.pop("fixed_slippage", None)
            return
        v = getattr(s, "value", s if isinstance(s, (int, float)) else None)
        if v:
            # FixedSlippage: 绝对价差 v 元, 买卖各偏 v/2(元)
            rt.cost_cfg["fixed_slippage"] = float(v)
            rt.cost_cfg.pop("slippage_bps", None)

    def set_order_cost(cost, type="stock", **kwargs):
        k = getattr(cost, "kwargs", None) or {}
        try:
            oc = float(k.get("open_commission") or 0)
            sc = float(k.get("close_commission") or 0)
            ot = float(k.get("open_tax") or 0)
            st_ = float(k.get("close_tax") or 0)
            if oc > 0:
                rt.cost_cfg["buy_cost"] = oc
            if sc > 0:
                rt.cost_cfg["sell_cost"] = sc
            if ot:
                rt.cost_cfg["buy_tax"] = ot
            if st_:
                rt.cost_cfg["sell_tax"] = st_
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
