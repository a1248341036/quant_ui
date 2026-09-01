# -*- coding: utf-8 -*-
"""验证 run_jq_backtest 返回 benchmark 曲线 + bench_code。"""
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\Quant\quant_ui')

from core.event_engine.jq import run_jq_backtest

from _code_block import extract_jq_code

src = open('scripts/jq_repro/_validate_jq_compat.py', encoding='utf-8').read()
code = extract_jq_code(src)
assert 'def initialize' in code and 'set_benchmark' in code, '策略代码提取失败'

out = run_jq_backtest(code, start='2025-09-01', capital=100_000.0, smoke=True)
assert out.get('ok'), out.get('error')
bench = out.get('benchmark')
print('bench_code:', out.get('bench_code'))
print('benchmark points:', len(bench) if bench else 0)
if bench:
    print('first:', bench[0], 'last:', bench[-1])
    assert len(bench) == len(out['nav']), 'benchmark/nav 长度不一致'
    assert abs(bench[0]['value'] - 1.0) < 1e-9
print('SMOKE PASS')
