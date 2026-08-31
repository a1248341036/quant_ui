# -*- coding: utf-8 -*-
"""红利搬砖策略(聚宽 post/27994) 端到端验证。

覆盖点: valuation.pe_ratio/pb_ratio(估值指标) + finance.STK_XR_XD(分红送转)
+ finance.run_query 通用表查询 + numpy 预载别名(zeros) + log 多参数。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse  # noqa: E402

CODE = '''
# 克隆自聚宽文章：https://www.joinquant.com/post/27994
# 标题：红利搬砖
# 作者：Gyro

# 导入库函数
import numpy as np
import pandas as pd
import datetime as dt
from jqdata import *

def initialize(context):
    # 设置系统
    set_option('use_real_price', True)
    # 设置信息格式
    log.set_level('order', 'error')
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 500)
    # 设置策略
    run_monthly(handle_trader, 1, '9:45')
    # 设置参数
    g.stock_pool = [
        '601988.XSHG', # 中国银行
        '601288.XSHG', # 农业银行
        '601398.XSHG', # 工商银行
        '601939.XSHG', # 建设银行
        ]
    g.num = 1 #选股数
    g.stocks = [] #股票池

def handle_trader(context):
    # 按年更新
    if context.current_dt.month in [5]:
        g.stocks = choice_stocks(context, g.stock_pool, g.num)
    # 卖出
    cdata = get_current_data()
    for s in context.portfolio.positions:
        if s not in g.stocks and not cdata[s].paused:
            log.info('sell', s, cdata[s].name)
            order_target(s, 0)
    # 买进
    position = 0.99*context.portfolio.total_value / max(1, len(g.stocks))
    for s in g.stocks:
        if s not in context.portfolio.positions and not cdata[s].paused and \\
            context.portfolio.available_cash > position:
            log.info('buy', s, cdata[s].name)
            order_value(s, position)

def choice_stocks(context, stocks, num):
    # 提取市值，基本面过滤
    sdf = get_fundamentals(query(
            valuation.code,
            valuation.market_cap, #单位，亿元
        ).filter(
            valuation.code.in_(stocks),
            valuation.pb_ratio > 0,
            valuation.pe_ratio > 0,
        )).dropna().set_index('code')
    stocks = list(sdf.index)
    # 最近三年的股息
    dt_now = context.current_dt.date()
    dt_3y = dt_now - dt.timedelta(days=3*365)
    ddf = finance.run_query(query(
            finance.STK_XR_XD.code,
            finance.STK_XR_XD.company_name,
            finance.STK_XR_XD.board_plan_pub_date,
            finance.STK_XR_XD.bonus_amount_rmb, #单位，万元
        ).filter(
            finance.STK_XR_XD.code.in_(stocks),
            finance.STK_XR_XD.board_plan_pub_date > dt_3y,
            finance.STK_XR_XD.board_plan_pub_date < dt_now,
            finance.STK_XR_XD.bonus_amount_rmb > 0
        ))
    log.info('\\n', ddf)
    stocks = list(set(ddf.code))
    # 累计分红
    divy = pd.Series(data=zeros(len(stocks)), index=stocks)
    for k in ddf.index:
        s = ddf.code[k]
        divy[s] += ddf.bonus_amount_rmb[k]
    # 建立数据表
    sdf = sdf.reindex(stocks)
    sdf['div_3y'] = divy
    # 计算股息率
    sdf['div_ratio'] = 0.33 * 1e-2 * sdf.div_3y / sdf.market_cap
    # report
    sdf['name'] = [get_security_info(s).display_name for s in sdf.index]
    sdf = sdf.sort_values(by='div_ratio', ascending=False)
    log.info('\\n', sdf[:10])
    return list(sdf.head(num).index)

# end
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end", default="2025-08-29")
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
    warn = [x for x in logs if "[warn]" in x or "Error" in x]
    print("warn/err:", warn[:8])
    # 选股输出(sdf 表)
    sel = [x for x in logs if "div_ratio" in x]
    for x in sel[:4]:
        print("选股表:", x[:300])
    buys = [x for x in logs if "buy" in x]
    print("买入日志:", buys[:4])
    hold = res.get("holdings", [])
    print("末期持仓:", [(h["code"], h.get("name", ""),
                         round(h["weight"], 3)) for h in hold])
    print("成交笔数:", len(res.get("trades", [])))
    assert res["ok"], "回测失败"


if __name__ == "__main__":
    main()
