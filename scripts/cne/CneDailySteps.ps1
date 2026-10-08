# CNE 每日同步预置步骤清单（纯数据，无 WPF 依赖）。
# GUI 看板（CneDailyGui.psm1）与 CLI worker（run_cne_daily.ps1 -NoGui 子进程）共用：
# worker 没有它就无法在波次失败时把受阻数据集标记进对账清单（2026-10-08 看板清单为空即此因）。

# ── 预置全量同步计划清单（固定 56 项全集顺序，无一遗漏）──────────────
$Global:CnePredefinedSteps = @(
    # 核心行情与参考基准 (core)
    [pscustomobject]@{ Dataset = "instruments";                  CnName = "全A标的清单";               Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "trading_calendar";             CnName = "交易日历";                 Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "trading_status";               CnName = "停复牌与ST状态";           Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "stock_st";                     CnName = "风险警示板ST名单";         Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "trading_status_st";            CnName = "ST证据日更续签";           Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "corporate_actions";            CnName = "除权除息与送转";           Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "tushare_wide_daily";           CnName = "Tushare宽表行情";          Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "daily_bars";                   CnName = "A股日K线";                 Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "index_bars";                   CnName = "主要指数日K线";             Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "fund_bars";                    CnName = "场内基金与ETF行情";        Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "derive_adj_factors";           CnName = "后复权因子计算";           Category = "核心行情"; Wave = "core" }
    [pscustomobject]@{ Dataset = "derive_industry_index";        CnName = "申万行业收益推导";         Category = "核心行情"; Wave = "core" }

    # 财务基本面与公告披露 (fundamentals)
    [pscustomobject]@{ Dataset = "financial_statement_items";    CnName = "财报科目长表";             Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "earnings_disclosure_schedule"; CnName = "预约披露时间表";           Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "index_constituents";           CnName = "指数成分股快照";           Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "industry_members";             CnName = "申万行业分类快照";         Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "share_structure";              CnName = "股本结构变动";             Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "shareholder_counts";           CnName = "股东户数数据";             Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "balancesheet";                 CnName = "资产负债表";               Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "income";                       CnName = "利润表";                   Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "cashflow";                     CnName = "现金流量表";               Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "fina_indicator";               CnName = "财务核心指标";             Category = "财务基本面"; Wave = "fundamentals" }
    [pscustomobject]@{ Dataset = "report_rc";                    CnName = "研报盈利预测";             Category = "财务基本面"; Wave = "fundamentals" }

    # 资金流与估值分析 (capital)
    [pscustomobject]@{ Dataset = "fund_flow";                    CnName = "个股资金流向";             Category = "资金估值"; Wave = "capital" }
    [pscustomobject]@{ Dataset = "valuation_metrics";            CnName = "估值指标(PE/PB/市值)";     Category = "资金估值"; Wave = "capital" }
    [pscustomobject]@{ Dataset = "sector_members";               CnName = "板块概念成分股";           Category = "资金估值"; Wave = "capital" }
    [pscustomobject]@{ Dataset = "announcement_index";           CnName = "巨潮公告索引";             Category = "资金估值"; Wave = "capital" }
    [pscustomobject]@{ Dataset = "fund_nav";                     CnName = "公募基金净值";             Category = "资金估值"; Wave = "capital" }
    [pscustomobject]@{ Dataset = "index_bars_external";          CnName = "基准指数行情";             Category = "资金估值"; Wave = "capital" }
    [pscustomobject]@{ Dataset = "margin_trading";               CnName = "融资融券余额";             Category = "资金估值"; Wave = "capital" }
    [pscustomobject]@{ Dataset = "northbound_holdings";          CnName = "陆股通持股季报";           Category = "资金估值"; Wave = "capital" }
    [pscustomobject]@{ Dataset = "northbound_flows";             CnName = "北向资金流(已停产)";       Category = "资金估值"; Wave = "capital" }

    # 宏观指标与大宗风险 (macro_risk)
    [pscustomobject]@{ Dataset = "macro_indicators";             CnName = "宏观经济指标";             Category = "宏观风险"; Wave = "macro_risk" }
    [pscustomobject]@{ Dataset = "market_breadth";               CnName = "市场宽度(涨跌家数)";       Category = "宏观风险"; Wave = "macro_risk" }
    [pscustomobject]@{ Dataset = "share_unlock_schedule";        CnName = "限售解禁计划";             Category = "宏观风险"; Wave = "macro_risk" }
    [pscustomobject]@{ Dataset = "regulatory_events";            CnName = "监管处罚公告";             Category = "宏观风险"; Wave = "macro_risk" }
    [pscustomobject]@{ Dataset = "commodity_bars";               CnName = "商品期货主连K线";         Category = "宏观风险"; Wave = "macro_risk" }

    # 事件披露与异动信号 (signals)
    [pscustomobject]@{ Dataset = "dragon_tiger";                 CnName = "龙虎榜交易明细";           Category = "事件信号"; Wave = "signals" }
    [pscustomobject]@{ Dataset = "block_trades";                 CnName = "大宗交易明细";             Category = "事件信号"; Wave = "signals" }
    [pscustomobject]@{ Dataset = "dividend";                     CnName = "分红送转披露";             Category = "事件信号"; Wave = "signals" }
    [pscustomobject]@{ Dataset = "namechange";                   CnName = "股票曾用名变更";           Category = "事件信号" ; Wave = "signals" }
    [pscustomobject]@{ Dataset = "share_float_external";         CnName = "限售解禁数据";             Category = "事件信号"; Wave = "signals" }
    [pscustomobject]@{ Dataset = "stk_surv";                     CnName = "机构调研活动";             Category = "事件信号"; Wave = "signals" }
    [pscustomobject]@{ Dataset = "fund_fees";                    CnName = "公募基金费率参考";         Category = "事件信号"; Wave = "signals" }

    # 舆情分析与前沿研究 (research)
    [pscustomobject]@{ Dataset = "institutional_holdings";       CnName = "机构持股汇总";             Category = "舆情研究"; Wave = "research" }
    [pscustomobject]@{ Dataset = "analyst_consensus";            CnName = "分析师一致预期";           Category = "舆情研究"; Wave = "research" }
    [pscustomobject]@{ Dataset = "hot_rank";                     CnName = "东财人气热榜";             Category = "舆情研究"; Wave = "research" }
    [pscustomobject]@{ Dataset = "sector_bars";                  CnName = "板块K线(同花顺)";         Category = "舆情研究"; Wave = "research" }
    [pscustomobject]@{ Dataset = "sector_fund_flow";             CnName = "板块资金流向";             Category = "舆情研究"; Wave = "research" }
    [pscustomobject]@{ Dataset = "news_headlines";               CnName = "新闻电报提要";             Category = "舆情研究"; Wave = "research" }
    [pscustomobject]@{ Dataset = "flash_news_wire";              CnName = "7x24快讯";                 Category = "舆情研究"; Wave = "research" }
    [pscustomobject]@{ Dataset = "sentiment_articles";           CnName = "个股新闻舆情打分";         Category = "舆情研究"; Wave = "research" }
    [pscustomobject]@{ Dataset = "sentiment_scores";             CnName = "市场情绪综合得分";         Category = "舆情研究"; Wave = "research" }

    # 综合入库与数据湖维护 (maintenance)
    [pscustomobject]@{ Dataset = "etf_fund_refresh";             CnName = "ETF/基金面板刷新";          Category = "入库维护"; Wave = "maintenance" }
    [pscustomobject]@{ Dataset = "stale_retry";                  CnName = "滞后补抓重试";             Category = "入库维护"; Wave = "maintenance" }
    [pscustomobject]@{ Dataset = "meta_backup";                  CnName = "元数据备份与清理";         Category = "入库维护"; Wave = "maintenance" }
)

$Global:CneDatasetNames = @{}
foreach ($item in $Global:CnePredefinedSteps) {
    $Global:CneDatasetNames[$item.Dataset] = $item.CnName
}
