# -*- coding: utf-8 -*-
"""avoid_future_data 生效验证: 开关后显式未来请求报错, 默认路径不受影响。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CODE = '''
from jqdata import *
import pandas as pd

def initialize(context):
    set_option('avoid_future_data', True)
    set_benchmark('399101.XSHE')
    g.results = []
    run_daily(probe, '10:00')

def probe(context):
    if getattr(g, 'done', False):
        return
    g.done = True
    # 1) 默认路径(不显式传未来)必须正常
    df = get_price('000001', end_date=context.previous_date, count=5,
                   frequency='daily', fields=['close'], panel=False)
    assert len(df) == 5 and df['close'].notna().all()
    g.results.append('默认路径OK')
    # 2) 显式请求未来 -> 必须报错(用 try 捕获转为记录)
    try:
        get_price('000001', start_date='2030-01-01', end_date='2030-12-31',
                  frequency='daily', fields=['close'], panel=False)
        g.results.append('!! 未来K线未被拦截')
    except ValueError as exc:
        g.results.append('未来K线已拦截: ' + str(exc)[:38])
    # 3) 未来截面 -> 拦截
    try:
        get_fundamentals(query(valuation.code).limit(3),
                         date=context.current_dt + pd.Timedelta(days=365))
        g.results.append('!! 未来截面未被拦截')
    except ValueError:
        g.results.append('未来截面已拦截')
    # 4) 未来龙虎榜 -> 拦截
    try:
        get_billboard_list('total', start_date='2030-01-01',
                           end_date='2030-12-31')
        g.results.append('!! 未来榜单未被拦截')
    except ValueError:
        g.results.append('未来榜单已拦截')
    log.info('PROBE|' + ' || '.join(g.results))
    unschedule_all()
'''

import pandas as pd  # noqa: E402
from core.event_engine.jq.entry import run_jq_backtest  # noqa: E402

res = run_jq_backtest(CODE, start="2025-01-06", end="2025-02-28",
                      capital=100_000.0)
print("ok:", res["ok"])
line = [x for x in res["logs"] if "PROBE|" in x]
assert line, "probe 未执行"
payload = line[0].split("PROBE|", 1)[1]
print(payload)
assert "默认路径OK" in payload
assert payload.count("已拦截") == 3, "应有 3 项未来请求被拦截"
assert "未被拦截" not in payload
print("avoid_future_data 验证通过")
