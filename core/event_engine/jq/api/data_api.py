# -*- coding: utf-8 -*-
"""数据获取函数(聚宽文档类别): 行情/基本面/证券信息/交易日历。

重型矩阵实现(get_price/history/attribute_history/get_current_data/
get_fundamentals/get_snapshot/get_factor)是 JQRuntime 的有状态方法
(依赖行情矩阵/截面/财务值缓存), 本模块负责:
1) 把运行时方法装配进策略命名空间;
2) 实现轻量的证券信息/交易日历类 API;
3) 注入 query DSL(valuation/income)。
指数代码(399101.XSHE 等)行情走 CNE curated index_bars。
"""
from __future__ import annotations

import pandas as pd

from core.event_engine.jq.objects import _SecurityInfo
from core.event_engine.jq.query import _Col, income, query, valuation


def install(ns: dict, rt) -> None:
    def get_index_stocks(index_symbol, date=None):
        # 全量池(域内全部股票), 点时口径: 剔除信号日尚未上市的代码;
        # 已退市代码无可靠退市标记, 保留(由 paused 过滤兜底)
        d = (pd.Timestamp(date) if date is not None
             else rt.context.previous_date)
        ldm = rt.ctx.list_date_map
        return [c for c in rt.ctx.codes
                if c not in ldm or ldm[c] <= d]

    def get_security_info(code):
        code = str(code).zfill(6)
        return _SecurityInfo(code, rt.ctx.name_map.get(code, ""),
                             rt.ctx.list_date_map.get(code))

    def get_all_securities(types="stock", date=None):
        codes = list(rt.ctx.codes)
        return pd.DataFrame({
            "display_name": [rt.ctx.name_map.get(c, "") for c in codes],
            "start_date": [rt.ctx.list_date_map.get(c)
                           or pd.Timestamp("1990-01-01") for c in codes],
            "end_date": pd.Timestamp("2200-01-01"),
            "type": "stock",
        }, index=codes)

    def get_trade_days(start_date=None, end_date=None, count=None):
        return rt._get_trade_days(start_date, end_date, count)

    def get_all_trade_days():
        return rt.ctx.tables.dates

    def normalize_code(code):
        s = str(code).strip().upper()
        if "." in s:
            return s
        c = s.zfill(6)
        return c + (".XSHG" if c.startswith(("5", "6", "9", "11", "13"))
                    else ".XSHE")

    ns.update({
        # 有状态实现: JQRuntime 方法(矩阵/截面/财务缓存)
        "get_price": rt.get_price,
        "get_snapshot": rt.get_snapshot,
        "history": rt.history,
        "attribute_history": rt.attribute_history,
        "get_current_data": rt.get_current_data,
        "get_fundamentals": rt.get_fundamentals,
        "get_factor": rt.get_factor,
        # 本模块轻量实现
        "get_index_stocks": get_index_stocks,
        "get_security_info": get_security_info,
        "get_all_securities": get_all_securities,
        "get_trade_days": get_trade_days,
        "get_all_trade_days": get_all_trade_days,
        "normalize_code": normalize_code,
        # query DSL(get_fundamentals 的查询构造器)
        "query": query,
        "valuation": valuation,
        "income": income,
    })
