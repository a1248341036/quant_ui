# -*- coding: utf-8 -*-
"""聚宽兼容层 —— 在本平台跑聚宽风格的策略代码。

包结构(core/event_engine/jq/):
- runtime.py   JQRuntime: exec 用户代码 + 任务注册 + 数据/下单 API + 日循环适配
- query.py     迷你 query DSL: query(valuation.x).filter(...).order_by().limit()
- objects.py   壳对象: g/log/context/portfolio/positions/get_current_data 视图
- entry.py     run_jq_backtest 入口(exec -> 适配器 -> run_event_backtest)

用户代码按聚宽写法: initialize(context) + run_daily/run_weekly 注册 +
get_price / get_fundamentals(query(...)) / get_current_data / order_target_value。

与聚宽的差异(日线近似, 详见 scripts/jq_repro/checklist.md):
- 函数时序: 注册的函数每个交易日按注册顺序各执行一次(时间只保序不区分);
  信号=昨日收盘数据, 成交=今日开盘价(T+1 开盘), 涨停开盘买不进/跌停卖不出
- get_price/history 的 '1m' 频率按日线近似(最新收盘)
- close/open 返回真实价(未复权, 对齐聚宽 use_real_price 语义); 撮合层用前复权
  价格处理分红除权, 净值含分红
- get_fundamentals 支持 query DSL 子集(见 query.py 的 valuation/income 列)
- get_index_stocks: 399101(中小板综) -> 沪深主板域近似; 其余指数未接入
- finance.run_query(审计意见等) 未接入
"""
from core.event_engine.jq.entry import run_jq_backtest  # noqa: F401
