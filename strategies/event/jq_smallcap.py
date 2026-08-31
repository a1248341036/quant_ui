# -*- coding: utf-8 -*-
"""小盘三正(聚宽致敬市场15) —— 聚宽风格单文件策略。

改因子/调参只动这个文件:
- select(): 选股逻辑(过滤+排序), 返回按优先级排序的候选代码
- PARAMS:  仓位/风控/调仓日/择时
骨架(调仓执行/豁免/止损/清仓)在 _runtime.py, 一般不用动。

模拟盘: 创建事件账户选本模块 + 事件策略 "小盘三正JQ" + 股票池 "全A主板"。
快速回测: python strategies/event/_run_backtest.py jq_smallcap 2025-01-01
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runtime import build_context, make_event_strategy  # noqa: E402

# ── 参数(聚宽原版 v2.0) ──────────────────────────────────────────
PARAMS = dict(
    stock_num=7,               # 基准持仓数
    max_single_weight=0.12,    # 单票上限 12%
    max_exposure=0.70,         # 小票总暴露上限 70%
    stoploss=0.07,             # 个股止损 -7%
    take_profit=2.0,           # 止盈 100%
    market_crash=0.05,         # 域平均(1-收/开)>=5% 清仓
    highest=60.0,              # 信号日收盘价上限(元)
    rebalance_weekday=1,       # 周二调仓(0=周一)
    pass_months=(4,),          # 4月空仓(原版避年报雷)
    top_keep=50,               # 候选保留数(原版 stock_num*3=21)
    # MA10 连续仓位: 7 -> 4(sigmoid, 相对阈值 2.5%)
    ma_timing=dict(window=10, base=7, span=3, scale=0.025, lo=4),
)

MIN_MV, MAX_MV = 3e4, 1e7     # 万元: 3亿 ~ 1000亿
LISTED_MIN = 375              # 上市满 375 天


def select(snap):
    """选股: 主板微盘 + 国九条三正, 按市值升序。

    snap 列: close/close_raw/mv/turnover/st/paused/listed_ok/hl/
             limit_price/fin_三正 (index=code)
    """
    df = snap
    ok = (df.mv.between(MIN_MV, MAX_MV)
          & df.fin_三正
          & ~df.st
          & ~df.paused
          & df.listed_ok)
    return df[ok].sort_values("mv").head(PARAMS["top_keep"]).index.tolist()


# ── 运行时装配(不用动) ───────────────────────────────────────────
_CTX = build_context()
EVENT_STRATEGIES = {"小盘三正JQ": make_event_strategy(PARAMS, select, _CTX)}
