# -*- coding: utf-8 -*-
"""验证 run_jq_backtest 返回 benchmark 曲线(聚宽基点口径) + bench_code。

聚宽口径: 基点 = 窗口首日之前的收盘, 首点 = 1+首日基准涨跌(≠1)。
用 _LAST_CTX 的原始指数日线独立复算核对。
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\Quant\quant_ui')

from _code_block import extract_jq_code
from core.event_engine.jq import run_jq_backtest, entry

src = open('scripts/jq_repro/_validate_jq_compat.py', encoding='utf-8').read()
code = extract_jq_code(src)
assert 'def initialize' in code and 'set_benchmark' in code, '策略代码提取失败'

out = run_jq_backtest(code, start='2025-09-01', capital=100_000.0, smoke=True)
assert out.get('ok'), out.get('error')
bench = out.get('benchmark')
print('bench_code:', out.get('bench_code'))
print('benchmark points:', len(bench) if bench else 0)
assert bench and len(bench) == len(out['nav']), 'benchmark/nav 长度不一致'
assert all(b['value'] == b['value'] for b in bench), 'benchmark 含 NaN'

# 独立核对: 首点 = 首日收盘 / 前一交易日收盘(聚宽基点=窗口前)
ctx = entry._LAST_CTX
ix = ctx.index_frame(out['bench_code'])
b_all = ix['close'].astype(float).dropna().sort_index()
import pandas as pd
d0 = pd.Timestamp(bench[0]['date'])
loc = int(b_all.index.searchsorted(d0))
expected_first = float(b_all.iloc[loc]) / float(b_all.iloc[loc - 1])
print('curve first:', bench[0]['value'], '| independent:', expected_first)
assert abs(bench[0]['value'] - expected_first) < 1e-9, '首点口径不符(应=首日收盘/前日收盘)'

panel_b = out['metrics'].get('基准收益')
print('panel 基准收益:', panel_b, '| curve last - 1:', bench[-1]['value'] - 1)
assert abs(panel_b - (bench[-1]['value'] - 1)) < 1e-9, '面板基准收益与曲线终点不一致'

print('SMOKE PASS')
