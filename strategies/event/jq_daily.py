# -*- coding: utf-8 -*-
"""日线模板示例 —— 风控类 + 策略类，代码优先、显式开启。

演示 strategies/template/daily.py 的最终用法：
- 稳健风控(RiskPolicy): __init__ 里配置止损/止盈/冷却/底线，def 了挂上就触发
- 小盘三正(DailyStrategy): select() 只写选股，挂 risk 一行即生效

回测:  python strategies/event/_run_backtest.py jq_daily 2025-01-01
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根

from _runtime import build_context  # noqa: E402
from strategies.template.daily import DailyStrategy, RiskPolicy  # noqa: E402

_CTX = build_context()


# ── 风控类：__init__ 里配置，def 了挂上就触发；不写 = 无风控 ──
class 稳健风控(RiskPolicy):
    def __init__(self):
        self.stoploss = 0.07           # 个股止损 -7%
        self.take_profit = 2.0         # 止盈 100%
        self.cooldown_days = 1         # 止损后 1 天禁买
        self.market_crash = 0.05       # 大盘惨跌 5% 清仓
        self.pass_months = (4,)        # 4 月空仓
        self.max_single_nav = 0.25     # 底线: 单票占净值上限
        self.daily_loss_limit = 0.05   # 底线: 单日亏损熔断


# ── 策略类：select 只写选股，其余全是属性配置 ──
class 小盘三正(DailyStrategy):
    data_ctx = _CTX
    risk = 稳健风控
    stock_num = 7
    rebalance_weekday = 1              # 周二调仓
    position_mode = "cash_equal"       # 剩余现金均分
    top_keep = 50
    highest = 60.0                     # 股价上限(持仓豁免)

    def select(self, snap):
        df = snap
        ok = (df.mv.between(3e4, 1e7)
              & df.fin_三正
              & ~df.st
              & ~df.paused
              & df.listed_ok)
        return df[ok].sort_values("mv").index.tolist()


EVENT_STRATEGIES = {"日线小盘三正": 小盘三正}
