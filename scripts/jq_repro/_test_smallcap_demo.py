# -*- coding: utf-8 -*-
"""小盘三正示例(用户 Code tab 版本, 裸码 000300 指数) 端到端验证。

崩溃点: get_price('000300', ...) 裸码无后缀被当作股票 -> 空数据 ->
adjust_stock_num 的 iloc[-1] 越界。修复: 裸 000xxx 不在股票面板时按上证系
指数解析(000300 -> CNE index_bars 真实沪深300)。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse  # noqa: E402

CODE = '''
# 聚宽风格策略示例 —— 小盘三正
# 改选股逻辑直接改 get_stock_list / select 部分, 参数在 initialize 里
from jqdata import *
import numpy as np
import pandas as pd
from datetime import timedelta

def initialize(context):
    g.stock_num = 7            # 基准持仓数
    g.min_mv, g.max_mv = 3, 1000   # 市值区间(亿)
    g.highest = 60             # 收盘价上限(元)
    g.stoploss = 0.07
    g.pass_months = [4]
    g.hold_list = []
    g.yesterday_HL_list = []
    g.target_list = []
    run_daily(prepare_stock_list, time='9:05')
    run_daily(sell_stocks, time='10:00')
    run_daily(trade_afternoon, time='14:00')
    run_weekly(weekly_adjustment, 2, time='10:00')

def prepare_stock_list(context):
    g.hold_list = [p.security for p in context.portfolio.positions.values()]
    if g.hold_list:
        df = get_price(g.hold_list, end_date=context.previous_date,
                       fields=['close', 'high_limit'], count=1, panel=False)
        g.yesterday_HL_list = list(df[df['close'] == df['high_limit']]['code'])
    else:
        g.yesterday_HL_list = []

def get_stock_list(context):
    stocks = get_index_stocks('000300')   # 全量池
    cur = get_current_data()
    initial = [s for s in stocks
               if not cur[s].paused and not cur[s].is_st and '退' not in cur[s].name
               and not s.startswith(('30', '68', '8', '4'))]
    q = query(valuation.code, valuation.market_cap,
              income.np_parent_company_owners, income.net_profit,
              income.operating_revenue).filter(
        valuation.code.in_(initial),
        valuation.market_cap.between(g.min_mv, g.max_mv),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
    ).order_by(valuation.market_cap.asc()).limit(21)
    df = get_fundamentals(q, date=context.previous_date)
    last = history(1, unit='1d', field='close', security_list=list(df['code']))
    return [s for s in df['code'] if s in g.hold_list
            or (s in last and last[s][-1] <= g.highest)]

def weekly_adjustment(context):
    if context.current_dt.month in g.pass_months:
        for s in list(context.portfolio.positions):
            order_target_value(s, 0)
        return
    num = adjust_stock_num(context)
    g.target_list = get_stock_list(context)[:num]
    for s in g.hold_list:
        if s not in g.target_list and s not in g.yesterday_HL_list \\
                and s in context.portfolio.positions:
            order_target_value(s, 0)
    buy_list = [s for s in g.target_list if s not in g.hold_list]
    if not buy_list:
        return
    pv = context.portfolio.total_value
    exposure = sum(p.value for p in context.portfolio.positions.values()) / pv
    avail = 0.70 - exposure
    per = min(0.12, max(avail, 0.0) / len(buy_list))
    for s in buy_list:
        order_target_value(s, per * pv)

def adjust_stock_num(context):
    idx = get_price('000300', end_date=context.previous_date, count=30)
    ma = idx['close'].rolling(10).mean()
    if pd.isna(ma.iloc[-1]):
        return g.stock_num
    diff = idx['close'].iloc[-1] - ma.iloc[-1]
    frac = 1.0 / (1.0 + np.exp(-diff / (idx['close'].iloc[-1] * 0.025)))
    return max(4, min(g.stock_num, int(round(g.stock_num - 3 * frac))))

def sell_stocks(context):
    pos = context.portfolio.positions
    for s in list(pos.keys()):
        if pos[s].price >= pos[s].avg_cost * 2:
            order_target_value(s, 0)
        elif pos[s].price < pos[s].avg_cost * (1 - g.stoploss):
            order_target_value(s, 0)
    df = get_price(get_index_stocks('000300'), end_date=context.previous_date,
                   fields=['close', 'open'], count=1, panel=False)
    if not df.empty and (1 - df['close'] / df['open']).mean() >= 0.05:
        for s in list(pos.keys()):
            order_target_value(s, 0)

def trade_afternoon(context):
    # 涨停开板近似: 昨收涨停 且 今开 <+9.5% (未封死) -> 开盘卖
    for s in g.yesterday_HL_list:
        if s in context.portfolio.positions:
            cur = get_current_data()[s]
            op = get_price(s, end_date=context.current_dt, fields=['open'], count=1)
            if not op.empty and op['open'].iloc[-1] < cur.last_price * 1.095:
                order_target_value(s, 0)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-08")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--capital", type=float, default=100_000.0)
    args = ap.parse_args()

    from core.event_engine.jq.entry import run_jq_backtest

    res = run_jq_backtest(CODE, start=args.start, end=args.end,
                          capital=args.capital)
    print("\n===== 结果 =====")
    print("ok:", res.get("ok"))
    print("metrics:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in res.get("metrics", {}).items()
                       if isinstance(v, (int, float))})
    logs = res.get("logs", [])
    warn = [x for x in logs if "[warn]" in x]
    print("warn:", warn[:5])
    hold = res.get("holdings", [])
    print("末期持仓:", [(h["code"], h.get("name", ""),
                         round(h["weight"], 3)) for h in hold])
    print("成交笔数:", len(res.get("trades", [])))
    assert res["ok"], "回测失败"


if __name__ == "__main__":
    main()
