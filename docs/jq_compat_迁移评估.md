# 聚宽 API → 本平台迁移评估表

> 依据: 官方 API 文档全文快照 `docs/jq_api_snapshot/api_full.html`（2026-08-31 抓取）。
> 判定档位: ✅ 已支持 | 🟢 可直迁（低成本）| 🟡 可迁需扩展（中等）| 🔴 不建议/不可行
> 数据前提: 股票日线=CNE quant_dataset；指数=CNE curated index_bars；财务=pg_parquet（CNE 注册外部集）；ST= CNE curated trading_status。
>
> **兼容层实现按聚宽文档类别拆分**（`core/event_engine/jq/api/`，各模块 `install(ns, rt)` 装配）:
>
> | 聚宽文档类别 | 模块 |
|---|---|
| 策略程序架构 / 运行时间 | `api/framework.py` |
| 策略设置函数 | `api/settings.py` |
| 数据获取函数 (+jqlib 因子桥) | `api/data_api.py` + `factor_bridge.py`（重型矩阵实现为 JQRuntime 方法） |
| 交易函数 | `api/trading.py` |
| 对象 | `objects.py` |
| 策略组合操作 | `api/portfolio.py` |
| 其他函数 | `api/misc.py` |
| 生命周期编排 | `runtime.py`（run_day 生命周期驱动 + 撮合对接，非 ns 装配） |


## 一、生命周期 / 框架

| 聚宽 API | 判定 | 说明 |
|---|---|---|
| initialize(context) | ✅ | 已支持 |
| process_initialize(context) | 🟢 | 每日 init 后调一次空钩子即可（兼容重入型策略） |
| before_trading_start(context, data) | 🟢 | 映射 'before_open' 时间槽（schedule 排序已就绪） |
| handle_data(context, data) | 🟢 | 映射 'every_bar'；data 需 SecurityUnitData 轻量对象 |
| after_trading_end(context) | 🟢 | 映射 'after_close' 时间槽 |
| run_daily / run_weekly / run_monthly | ✅ | 已支持：时间排序、kwargs 忽略、月内第 N 交易日 |
| unschedule_all() | 🟢 | 清空 scheduled 列表 |
| run_interval / Tick 级专用函数 | 🔴 | 日线引擎无意义（数据湖有分钟线，属另一档工程） |

## 二、设置函数

| 聚宽 API | 判定 | 说明 |
|---|---|---|
| set_option('use_real_price'/'avoid_future_data') | ✅ | no-op（本引擎恒为真实价+点时数据） |
| set_option('order_volume_ratio') | 🟢 | 可映射引擎 max_participation（口径差：JQ 按当根 bar 量，本引擎按 20 日均额） |
| set_benchmark | 🟢→ | 现 no-op；CNE index_bars 齐备，可升级为真实基准并输出超额 |
| set_slippage(FixedSlippage(v)) | 🟡 | JQ 语义是**绝对价差**（买卖各偏 v/2 元）；现 cost_cfg 当 bps 处理，小价差近似可用，精确需按元实现 |
| set_slippage(PriceRelatedSlippage(r)) | 🟢 | 比例滑点 → slippage_bps 直接换算 |
| set_order_cost | ✅ | 已生效：佣金/印花税折算 buy_cost/sell_cost；min_commission 每笔最低佣金已接入引擎（单笔费用 = max(金额×费率, min_commission)，买卖均生效） |
| set_universe | ✅ | no-op（旧 API） |
| set_subportfolios / SubPortfolio / transfer_cash | 🟡 | 单账户可兼容（id=0），多账户资金划转无引擎支持 |

## 三、交易函数与对象

| 聚宽 API | 判定 | 说明 |
|---|---|---|
| order / order_target / order_value / order_target_value | ✅ | 已支持（T+1 开盘撮合、100 股整数手、涨跌停/停牌拒单） |
| order(..., style=MarketOrderStyle/LimitOrderStyle) | 🟢 | 接受忽略；LimitOrder 可近似（开盘价越过限价则拒） |
| cancel_order(order) | 🟢 | 从当日 pending 队列移除 |
| get_orders() / get_open_orders() | 🟢 | 引擎 fills / pending 已有数据，包一层 Order 对象 |
| Order 对象(status/filled/amount/security/avg_cost) | ✅ | 已支持（提交时 opened，T+1 撮合后 held） |
| Trade 对象 | 🟢 | 引擎 fills（date/price/amount/side）可直接映射 |
| Position(.closeable_amount) | 🟢 | 已有 security/total_amount/price/avg_cost/value；closeable_amount（T+1 可卖）引擎状态可直接给 |
| Portfolio(.returns/.locked_cash/.start_date) | 🟢 | cash/total_value/positions_value 已有；其余属性直接补 |
| Context(.subportfolio/.previous_date/.current_dt/.run_params) | ✅ | 已支持 |
| SecurityUnitData | 🟢 | handle_data 用；open/close/high/low/mavg(n) 可包 |
| 融资融券/期货专用下单 | 🔴 | 引擎不支持（get_mtss 等数据类查询可迁，见下） |

## 四、数据获取函数

| 聚宽 API | 判定 | 说明 / 数据来源 |
|---|---|---|
| get_price（daily） | ✅ | 已支持 close/open/high_limit/low_limit/close_adj；🟢 补 high/low/volume/money（stock_daily 均有，meta 已读 high/low） |
| get_price（'1m'/'5m'/'60m'） | 🟡 | 现 1m≈日线近似；数据湖有 1/5/15/30/60min 逐股 parquet，60m 可真支持（工程中等） |
| history(count, unit, field) | ✅ | 已支持 1d/1m（信号日为界，无未来泄漏）；'60m' 同上可真支持 |
| attribute_history | ✅ | 已支持 |
| get_current_data() | ✅ | 已支持（T 日涨跌停/ST/停牌，last_price≈开盘防泄漏） |
| get_current_tick / get_ticks | 🔴 | Tick 级不做 |
| get_fundamentals（valuation） | ✅ | market_cap 已有；**pe_ratio/pb_ratio 已补**（市值/年化归母净利、市值/归母净资产，balancesheet.parquet 点时口径，亏损/负净资产→NaN） |
| get_fundamentals（income） | ✅ | 已支持三字段（ann_date 点时）；balancesheet/cashflow.parquet 在库，可扩 |
| finance.run_query + STK_XR_XD（分红送转） | ✅ | **已补**（api/finance.py）：dividend.parquet 映射（code/company_name/report_date/board_plan_pub_date/dividend_ratio/bonus_ratio_rmb/bonus_amount_rmb(=每股派息×总股本, 万元)/record_date/ex_date/pay_date）；同报告期 预案/实施 多行去重；日期列为 datetime.date（与聚宽一致，可直接与 dt.date 比较） |
| 代码归一化 | ✅ | 聚宽风格 '601988.XSHG' 与裸码 '601988' 全链路等价（in_/get_current_data/get_price/get_security_info/下单/持仓） |
| numpy 全局别名 | ✅ | zeros/ones/array/arange/mean/nanmean/std/where 等 28 个预载（不覆盖 max/min/sum/abs/round 内建；`log` 让位给策略日志对象） |
| get_fundamentals（indicator） | 🟢 | fina_indicator.parquet 在库（ROE/EPS/毛利率等 JQ indicator 常用字段基本齐） |
| get_fundamentals_continuously | 🟢 | 循环调用封装 |
| get_index_stocks | ✅→ | 已支持（全量池点时）；**CNE curated/index_constituents 在库 → 可升级为真实指数成分**（国九策略用 399101 成分才是原意） |
| get_all_securities(['stock']) | ✅ | 已支持；etf/index 类型可扩（CNE fund_bars/index_bars） |
| get_security_info | ✅→ | start_date/display_name 已有；end_date/type 可由 namechange.parquet 补 |
| get_trade_days / get_all_trade_days | 🟢 | tables.dates / CNE trade_cal 直接给 |
| get_extras('is_st') | 🟢 | tables.is_st 直接给 |
| get_extras('unit_net_value'/'acc_net_value') | 🟡 | CNE fund_nav（ETF/货币基金净值） |
| get_locked_shares（限售解禁） | 🟢 | pg_parquet/share_float.parquet 在库 |
| get_billboard_list（龙虎榜） | 🔴 | CNE 暂无对应数据集 |
| get_money_fund | 🟡 | CNE fund_nav/fund_bars（货币基金） |
| get_mtss / get_marginsec_stocks（两融） | 🟡 | CNE 有 margin_trading 数据集，可接 |
| get_industry_stocks / get_industry | 🟡 | CNE industry_members / sector_members / derived/industry_index |
| get_concept_stocks / get_concepts | 🟡 | CNE sector_members 待确认是否含概念板块 |
| get_factor_values（jqfactor） | 🟡 | AlphaAgent 因子库/DSL 算子可覆盖部分自定义因子 |
| jqlib（alpha101/alpha191/technical_analysis） | 🟡 | 本平台 DSL 算子库可自算大部分技术指标 |
| finance.run_query（利润表等） | 🟡 | income/balancesheet/cashflow 可映射；STK_AUDIT_OPINION（审计意见）无数据 |
| macro.run_query / get_yield_curve | 🔴 | 宏观/债券数据无 |
| normalize_code | 🟢 | 纯函数 |

## 五、其他

| 聚宽 API | 判定 | 说明 |
|---|---|---|
| log（info/debug/warn/error/set_level） | ✅ | 已支持（buffer + 前端展示） |
| write_file / read_file | 🟢 | 映射到 artifacts 目录（回测沙箱文件） |
| send_message / 微信通知 | 🟢 | no-op 即可 |
| 风险指标（回测结果页） | ✅ | 平台 metrics 自有（收益/回撤/夏普/换手/胜率） |
| 归因分析（Brinson/因子） | 🔴 | 平台另有 AlphaAgent 因子分析体系，不按 JQ 形态迁 |

## 建议落地顺序

> **P0 已全部落地（2026-08-31）**：生命周期四钩子（process_initialize/before_trading_start/
> handle_data/after_trading_end，handle_data 注入 9:30 槽、data 为当日截面视图支持 .mavg(n)）、
> get_trade_days/get_all_trade_days、get_orders/get_open_orders/cancel_order、get_trades+
> Trade 对象、Position.closeable_amount、Portfolio.returns、normalize_code、unschedule_all、
> get_price 全字段（high/low/pre_close/volume/money，并行会话完成数据层）、
> PriceRelatedSlippage 精确映射（r/2→单边 bps）。回归：小盘三正与国九 2024 结果逐位一致。

1. ~~**P0（几十行级，立刻解锁大量社区策略）**~~：已实现，见上
2. **P1（数据已在库，每个半天级）**：get_index_stocks 接 CNE index_constituents（真实成分）、get_fundamentals 扩 indicator/pe/pb、get_extras(is_st)、get_locked_shares、get_security_info.end_date、set_benchmark 真实基准+超额输出
3. **P2（工程较大或受限）**：60 分钟真数据 history、get_industry_stocks、get_mtss、get_money_fund、多账户、get_fundamentals_continuously
4. **不做**：Tick/分钟撮合、期货/两融交易、宏观/债券、龙虎榜（待 CNE 扩数据源）
