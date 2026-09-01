# -*- coding: utf-8 -*-
"""get_billboard_list(龙虎榜) 全链路验证 —— CNE 懒注册插件 -> API 封装。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CODE = '''
from jqdata import *
import pandas as pd

def initialize(context):
    set_benchmark('399101.XSHE')
    run_daily(billboard_probe, '10:00')

def billboard_probe(context):
    # 只在首个非 warmup 交易日跑一次
    if context.previous_date < pd.Timestamp('2025-01-06'):
        return
    if getattr(g, 'done', False):
        return
    g.done = True
    df = get_billboard_list('total', start_date='2024-12-01',
                            end_date=context.previous_date)
    if len(df) == 0:
        log.warn('龙虎榜为空')
        return
    codes = df['code'].unique()
    log.info('龙虎榜 %d 行, %d 只, 日期 %s~%s' % (
        len(df), len(codes),
        df.index.min().date(), df.index.max().date()))
    assert df.index.max() <= context.previous_date, '点时纪律: 混入未来数据'
    px = get_price(list(codes)[:5], end_date=context.previous_date,
                   frequency='daily', fields=['close'], count=6, panel=False)
    ret = px.groupby('code')['close'].agg(lambda s: s.iloc[-1]/s.iloc[0]-1)
    log.info('热门股5日涨幅: %s' % dict(ret.round(4)))
    unschedule_all()
'''

from core.event_engine.jq.entry import run_jq_backtest  # noqa: E402
res = run_jq_backtest(CODE, start="2025-01-06", end="2025-02-28",
                      capital=100_000.0)
print("ok:", res["ok"])
warn = [x for x in res["logs"] if "[warn]" in x]
info = [x for x in res["logs"] if "龙虎榜" in x or "热门股" in x]
for x in info[:4]:
    print(" ", x)
assert res["ok"]
assert not warn, warn
assert info, "billboard_probe 未执行"
print("get_billboard_list 全链路验证通过")
