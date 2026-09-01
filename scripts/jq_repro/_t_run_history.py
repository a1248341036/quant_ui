# -*- coding: utf-8 -*-
"""端到端: run_async -> 列表/详情(内存+磁盘回落) -> 历史留存。"""
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

r = client.post('/api/code/jq/run_async', json={
    'code': code, 'start': '2025-09-01', 'end': '2025-10-31', 'capital': 100000})
assert r.json()['ok'], r.json()
run_id = r.json()['run_id']
print('started:', run_id)

deadline = time.time() + 300
while time.time() < deadline:
    s = client.get(f'/api/code/jq/runs/{run_id}').json()
    if s['phase'] in ('done', 'error', 'cancelled'):
        break
    time.sleep(1)
assert s['phase'] == 'done', s.get('error')
print('run finished, elapsed', s['elapsed'])

# 列表: 应包含该运行, 带摘要指标
lst = client.get('/api/code/jq/runs?limit=20').json()
assert lst['ok'] and lst['items'], '列表为空'
item = next((x for x in lst['items'] if x['run_id'] == run_id), None)
assert item, '列表缺少刚完成的运行'
assert item['phase'] == 'done'
assert item['策略收益'] is not None and item['基准收益'] is not None
print('list item:', {k: item[k] for k in ('run_id', 'phase', 'start', 'end',
                                          '策略收益', '基准收益', 'code_head')})

# 磁盘文件已落盘(等 worker 线程写完, 最多 10s)
from backend.routers.code import JQ_RUNS_DIR
persisted = None
for _ in range(20):
    persisted = (JQ_RUNS_DIR / f'{run_id}.json').exists()
    if persisted:
        break
    time.sleep(0.5)
assert persisted, '落盘缺失'

# 详情: 内存可能已被清理模拟 -> 直接验证记录读取路径
rec = client.get(f'/api/code/jq/runs/{run_id}').json()
assert rec['phase'] == 'done' and rec['result']['metrics']
assert rec.get('start') == '2025-09-01', rec.get('start')
print('detail OK: result nav', len(rec['result']['nav']),
      'benchmark', len(rec['result']['benchmark']))
print('ALL PASS')
