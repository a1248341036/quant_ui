# -*- coding: utf-8 -*-
"""聚宽 API 兼容层 —— 按官方文档类别拆分的命名空间装配。

对应聚宽文档《策略API介绍》的类别划分(与 docs/jq_api_snapshot 快照一致):

- 策略程序架构 / 运行时间  -> framework.py (run_daily/run_weekly/run_monthly/
                              unschedule_all + 定时任务时间排序)
- 策略设置函数            -> settings.py (set_option/set_benchmark/set_universe/
                              set_slippage/set_order_cost + 滑点/费用类)
- 数据获取函数 (+jqlib)   -> data_api.py (get_price/history/attribute_history/
                              get_current_data/get_fundamentals/get_index_stocks/
                              get_all_securities/get_security_info/get_trade_days/
                              normalize_code/get_factor); 重型矩阵实现为
                              JQRuntime 方法, 本模块负责装配与轻量实现
- 数据获取函数 -> 财务数据 -> finance.py (finance.run_query + STK_XR_XD
                              分红送转表)
- 交易函数                -> trading.py (order 四件套/order_shares/
                              order_target_percent/cancel_order/get_orders/
                              get_open_orders/get_trades)
- 对象                    -> core/event_engine/jq/objects.py (Context/Portfolio/
                              Position/Order/Trade/SecurityUnitData 壳对象)
- 策略组合操作            -> portfolio.py (set_subportfolios/transfer_cash,
                              单账户兼容)
- 其他函数                -> misc.py (jqdata/jqfactor 模块桩, write_file 等占位)

每个模块暴露 install(ns, rt); install_all 按序装配(顺序: misc 的模块桩先于
finance 的 jqdata.finance 升级)。
"""
from . import data_api, finance, framework, misc, portfolio, settings, trading

_MODULES = (misc, finance, framework, settings, trading, data_api, portfolio)


def install_all(ns: dict, rt) -> None:
    """把全部类别的聚宽 API 装配进策略命名空间。"""
    for mod in _MODULES:
        mod.install(ns, rt)
