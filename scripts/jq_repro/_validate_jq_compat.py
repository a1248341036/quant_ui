# -*- coding: utf-8 -*-
"""端到端验证: 小盘三正策略用聚宽风格代码 + jq_compat 回测, 对齐已验证结果。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.event_engine.jq import run_jq_backtest  # noqa: E402

CODE = '''
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
from datetime import time, timedelta, datetime

def initialize(context):
    set_option('avoid_future_data', True)
    set_benchmark('399101.XSHE')
    set_option('use_real_price', True)
    set_slippage(FixedSlippage(0.0003))
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             close_commission=0.0001, open_commission=0.0001,
                             close_today_commission=0, min_commission=0.1), type='stock')
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'debug')

    g.trading_signal = True
    g.run_stoploss = True
    g.adjust_num = True
    g.hold_list = []
    g.yesterday_HL_list = []
    g.target_list = []
    g.pass_months = [4]
    g.limitup_stocks = []
    g.min_mv = 3
    g.max_mv = 1000
    g.stock_num = 7
    g.max_single_weight = 0.12
    g.max_smallcap_exposure = 0.70
    g.stoploss_limit = 0.07
    g.highest = 60
    g.reason_to_sell = {}

    run_daily(prepare_stock_list, time='9:05')
    run_daily(trade_afternoon, time='14:00')
    run_daily(sell_stocks, time='10:00')
    run_daily(close_account, time='14:50')
    run_weekly(weekly_adjustment, 2, time='10:00')

def prepare_stock_list(context):
    g.hold_list = [p.security for p in context.portfolio.positions.values()]
    g.limitup_stocks = []
    if g.hold_list:
        df = get_price(g.hold_list, end_date=context.previous_date,
                       frequency='daily', fields=['close', 'high_limit'],
                       count=1, panel=False, fill_paused=False)
        g.yesterday_HL_list = list(df[df['close'] == df['high_limit']]['code'])
    else:
        g.yesterday_HL_list = []
    g.trading_signal = today_is_between(context)

def get_stock_list(context):
    MKT_index = '399101.XSHE'
    idx_stocks = get_index_stocks(MKT_index, date=context.previous_date)
    initial_list = filter_stocks(context, idx_stocks)
    q = query(valuation.code, valuation.market_cap,
              income.np_parent_company_owners,
              income.net_profit,
              income.operating_revenue,
              ).filter(
        valuation.code.in_(initial_list),
        valuation.market_cap.between(g.min_mv, g.max_mv),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
    ).order_by(valuation.market_cap.asc()).limit(g.stock_num * 3)
    df = get_fundamentals(q, date=context.previous_date)
    final_list = list(df['code'])
    if not final_list:
        return []
    last_prices = history(1, unit='1d', field='close', security_list=final_list)
    return [s for s in final_list
            if s in g.hold_list or (s in last_prices and last_prices[s][-1] <= g.highest)]

def filter_stocks(context, stock_list):
    cur = get_current_data()
    last_prices = history(1, unit='1d', field='close', security_list=stock_list)
    out = []
    prev = context.previous_date
    for s in stock_list:
        if cur[s].paused or cur[s].is_st or '退' in cur[s].name:
            continue
        if s.startswith(('30', '68', '8', '4')):
            continue
        if s not in context.portfolio.positions:
            seq = last_prices.get(s)
            if not seq:
                continue
            lp = seq[-1]
            if lp >= cur[s].high_limit or lp <= cur[s].low_limit:
                continue
        info = get_security_info(s)
        if info.start_date is not None and (prev - info.start_date).days < 375:
            continue
        out.append(s)
    return out

def weekly_adjustment(context):
    if not g.trading_signal:
        log.info('空仓月份, 持币')
        return
    if g.adjust_num:
        new_num = adjust_stock_num(context)
        if g.stock_num != new_num:
            g.stock_num = new_num
            log.info('持仓数量调整为 %d' % new_num)
    g.target_list = get_stock_list(context)[:g.stock_num]
    sell_list = [s for s in g.hold_list
                 if s not in g.target_list and s not in g.yesterday_HL_list]
    for s in sell_list:
        if s in context.portfolio.positions:
            close_position(context.portfolio.positions[s])
    buy_list = [s for s in g.target_list if s not in g.hold_list]
    if not buy_list:
        return
    # 单票12%上限 + 70%暴露约束(与已验证复现口径一致)
    pv = context.portfolio.total_value
    exposure = sum(p.value for p in context.portfolio.positions.values()) / pv
    sold = sum(p.value for s, p in context.portfolio.positions.items()
               if s not in g.target_list) / pv
    avail = g.max_smallcap_exposure - (exposure - sold)
    per = min(g.max_single_weight, max(avail, 0.0) / len(buy_list))
    for s in buy_list:
        order_target_value(s, per * pv)

def check_limit_up(context):
    if not g.yesterday_HL_list:
        return
    for stock in g.yesterday_HL_list:
        if stock not in context.portfolio.positions:
            continue
        cur = get_price(stock, end_date=context.current_dt, frequency='1m',
                        fields=['close', 'high_limit'], skip_paused=False,
                        count=1, panel=False, fill_paused=True)
        if cur.iloc[0, 0] < cur.iloc[0, 1]:
            close_position(context.portfolio.positions[stock])
            g.limitup_stocks.append(stock)

def trade_afternoon(context):
    if g.trading_signal:
        check_limit_up(context)

def sell_stocks(context):
    if not g.run_stoploss:
        return
    pos = context.portfolio.positions
    for stock in list(pos.keys()):
        p = pos[stock].price
        ac = pos[stock].avg_cost
        if p >= ac * 2:
            order_target_value(stock, 0)
        elif p < ac * (1 - g.stoploss_limit):
            order_target_value(stock, 0)
            g.reason_to_sell[stock] = 'stoploss'
    df = get_price(get_index_stocks('399101.XSHE', date=context.previous_date),
                   end_date=context.previous_date, frequency='daily',
                   fields=['close', 'open'], count=1, panel=False)
    if not df.empty:
        down = (1 - df['close'] / df['open']).mean()
        if down >= 0.05:
            for stock in list(pos.keys()):
                order_target_value(stock, 0)

def adjust_stock_num(context):
    ma_para = 10
    today = context.previous_date
    start = today - timedelta(days=ma_para * 2)
    idx = get_price('399101.XSHE', start_date=start, end_date=today, frequency='daily')
    idx['ma'] = idx['close'].rolling(ma_para).mean()
    if len(idx) < ma_para or pd.isna(idx['ma'].iloc[-1]):
        return g.stock_num
    diff = idx['close'].iloc[-1] - idx['ma'].iloc[-1]
    scale = 500.0
    frac = 1.0 / (1.0 + np.exp(-diff / scale))
    num = int(round(g.stock_num - 3 * frac))
    num = max(4, min(g.stock_num, num))
    return num

def close_position(position):
    order_target_value(position.security, 0)

def today_is_between(context):
    return context.current_dt.month not in g.pass_months

def close_account(context):
    if not g.trading_signal and g.hold_list:
        for s in g.hold_list:
            if s in context.portfolio.positions:
                close_position(context.portfolio.positions[s])
'''


def main() -> int:
    out = run_jq_backtest(CODE, start="2025-09-01", capital=100_000.0)
    print("ok:", out["ok"])
    print("metrics:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in out["metrics"].items()})
    print("codes:", out["codes_count"], "窗口:", out["start"], "~", out["end"])
    print("期末持仓:")
    for h in out["holdings"]:
        print(f"  {h['code']} {h.get('name','')} 权重 {h['weight']:.2%}")
    print("最近 5 次调仓:")
    for t in out["trades"][-5:]:
        print(f"  {t['date']} 持仓 {t['num_hold']} 换手 {t['turnover']:.1%}")
    nav = out["nav"]
    apr = [p for p in nav if p["date"].startswith("2026-04")]
    if apr:
        vals = [p["value"] for p in apr]
        print(f"2026-04 净值波动: {max(vals)-min(vals):.6f} (应≈0)")
    print("日志尾部:")
    for line in out["logs"][-6:]:
        print(" ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
