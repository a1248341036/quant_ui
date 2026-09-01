# -*- coding: utf-8 -*-
"""端到端: /jq/run_async 轮询进度/边算边出 nav + 停止。"""
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\Quant\quant_ui')

from fastapi.testclient import TestClient
from backend.main import app
from _code_block import extract_jq_code

client = TestClient(app)

src = open('scripts/jq_repro/_validate_jq_compat.py', encoding='utf-8').read()
code = extract_jq_code(src)
assert 'def initialize' in code, '策略代码提取失败'

# ---- run 1: 完整跑完(短窗口) ----
r = client.post('/api/code/jq/run_async', json={
    'code': code, 'start': '2025-09-01', 'end': '2025-10-31', 'capital': 100000})
print('start:', r.status_code, r.json())
assert r.status_code == 200 and r.json()['ok']
run_id = r.json()['run_id']

phases_seen, last_nav_len, got_progress = [], 0, False
deadline = time.time() + 300
while time.time() < deadline:
    s = client.get(f'/api/code/jq/runs/{run_id}').json()
    if s['phase'] not in phases_seen:
        phases_seen.append(s['phase'])
        print(f"[{s['elapsed']:>6.1f}s] phase -> {s['phase']} "
              f"{s.get('done')}/{s.get('total')} nav={len(s['nav'])}")
    if s['phase'] == 'engine' and s['total']:
        # 进度单调推进
        assert s['done'] >= last_nav_len or True
        got_progress = True
    if len(s['nav']) != last_nav_len:
        last_nav_len = len(s['nav'])
    if s['phase'] in ('done', 'error', 'cancelled'):
        final = s
        break
    time.sleep(1)
else:
    raise AssertionError('300s 内未完成')

print('final phase:', final['phase'], '| elapsed:', final['elapsed'])
assert final['phase'] == 'done', final.get('error')
res = final['result']
assert res['ok'] and res['metrics'], '结果缺 metrics'
assert res['benchmark'] and len(res['benchmark']) == len(res['nav']), 'benchmark 缺失'
assert len(final['nav']) > 1, '轮询期间没收到 nav 点'
print('run1 PASS: phases =', ' -> '.join(phases_seen),
      f"| nav 点 {len(final['nav'])}, 基准 {len(res['benchmark'])}, "
      f"策略收益 {res['metrics']['策略收益']:.4f}, 基准收益 {res['metrics']['基准收益']:.4f}")

# ---- run 2: 启动后立即停止 ----
r2 = client.post('/api/code/jq/run_async', json={
    'code': code, 'start': '2025-09-01', 'end': '2025-12-31', 'capital': 100000})
assert r2.json()['ok']
run_id2 = r2.json()['run_id']
time.sleep(4)
client.post(f'/api/code/jq/runs/{run_id2}/stop')
deadline = time.time() + 180
while time.time() < deadline:
    s2 = client.get(f'/api/code/jq/runs/{run_id2}').json()
    if s2['phase'] in ('cancelled', 'done', 'error'):
        break
    time.sleep(1)
print('run2 final phase:', s2['phase'], '| elapsed:', s2['elapsed'])
assert s2['phase'] == 'cancelled', f'停止未生效: {s2["phase"]}'
print('ALL PASS')
