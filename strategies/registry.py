from __future__ import annotations

# 策略 = 因子 + 排序方向
# ascending=True 表示买因子值最小的一批（如低波动/低换手/超跌反转）
# ascending=False 表示买因子值最大的一批（如高动量/高成交额）
# group: 前端分组；desc: 前端展示的一句话说明
STRATEGIES = {
    "低换手冷门": {"factor": "turn20", "ascending": True,
                   "group": "冷门/价值", "desc": "换手率最低的一批股票"},
    "高成交领涨": {"factor": "am20", "ascending": False,
                   "group": "动量/趋势", "desc": "20 日平均成交额最高的资金龙头"},
    "低成交冷门": {"factor": "am20", "ascending": True,
                   "group": "冷门/价值", "desc": "成交额最低的冷门股"},
    "冷门+行业分散": {"factor": "am20", "ascending": True, "industry_cap": 1,
                       "group": "冷门/价值", "desc": "低成交 + 每行业最多 1 只分散持仓"},
    "动量 20 日": {"factor": "mom20", "ascending": False,
                   "group": "动量/趋势", "desc": "20 日涨幅最大"},
    "动量 60 日": {"factor": "mom60", "ascending": False,
                   "group": "动量/趋势", "desc": "60 日涨幅最大"},
    "每半年动量 Top1": {"factor": "mom60", "ascending": False,
                        "group": "动量/趋势",
                        "desc": "每年 3月/9月 调仓，60日动量最强 1 只持有半年（SPMO 策略 ETF/股票池代理）"},
    "每半年动量 Top3": {"factor": "mom60", "ascending": False,
                        "group": "动量/趋势",
                        "desc": "每年 3月/9月 调仓，60日动量最强 3 只持有半年（SPMO 策略分散版）"},
    "反转 20 日": {"factor": "mom20", "ascending": True,
                   "group": "反转/均值回归", "desc": "20 日跌幅最大，博超跌反弹"},
    "双均线多头 5/10": {"factor": "ma_cross5_10", "ascending": False,
                         "group": "动量/趋势", "desc": "MA5 相对 MA10 乖离最大，超短线多头"},
    "双均线反转 5/10": {"factor": "ma_cross5_10", "ascending": True,
                         "group": "反转/均值回归", "desc": "MA5 相对 MA10 乖离最小，超短线超跌"},
    "双均线多头 5/20": {"factor": "ma_cross5_20", "ascending": False,
                         "group": "动量/趋势", "desc": "MA5 相对 MA20 乖离最大，金叉/多头优先"},
    "双均线反转 5/20": {"factor": "ma_cross5_20", "ascending": True,
                         "group": "反转/均值回归", "desc": "MA5 相对 MA20 乖离最小，死叉/超跌反弹"},
    "双均线多头 10/30": {"factor": "ma_cross10_30", "ascending": False,
                          "group": "动量/趋势", "desc": "MA10 相对 MA30 乖离最大，波段多头"},
    "双均线反转 10/30": {"factor": "ma_cross10_30", "ascending": True,
                          "group": "反转/均值回归", "desc": "MA10 相对 MA30 乖离最小，波段超跌"},
    "双均线多头 20/60": {"factor": "ma_cross20_60", "ascending": False,
                          "group": "动量/趋势", "desc": "MA20 相对 MA60 乖离最大，中期多头优先"},
    "双均线多头 20/30": {"factor": "ma_cross20_30", "ascending": False,
                          "group": "动量/趋势", "desc": "MA20 相对 MA30 乖离最大，参数稳定性验证优选"},
    "双均线多头 20/30 ADX25": {"factor": "ma_cross20_30", "ascending": False,
                                "group": "动量/趋势", "desc": "MA20/MA30 乖离多头 + ADX≥25 趋势强度过滤",
                                "adx_filter": 25.0},
    "双均线反转 20/60": {"factor": "ma_cross20_60", "ascending": True,
                          "group": "反转/均值回归", "desc": "MA20 相对 MA60 乖离最小，中期超跌反弹"},
    "低波动": {"factor": "vol20", "ascending": True,
               "group": "波动/风控", "desc": "20 日波动率最低"},
    "复合因子": {"factor": "composite", "ascending": True,
                 "group": "复合/综合", "desc": "低成交 + 低波动复合打分"},
    "多空动量 20 日": {"factor": "mom20", "ascending": False,
                        "long_short": True, "short_n": 3, "short_cost_rate": 0.086,
                        "group": "多空/对冲", "desc": "动量 Top3 多头 + 最弱 3 只空头，净敞口 0"},
    "多空低换手": {"factor": "turn20", "ascending": True,
                   "long_short": True, "short_n": 3, "short_cost_rate": 0.086,
                   "group": "多空/对冲", "desc": "低换手 Top3 多头 + 高换手 3 只空头，净敞口 0"},
}


def list_strategies() -> list[str]:
    return list(STRATEGIES.keys())


def get_strategy(name: str) -> dict:
    return STRATEGIES[name]
