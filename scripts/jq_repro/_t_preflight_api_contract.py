# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\Quant\quant_ui')
import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# 失败路径: 缺失 API
bad = 'def initialize(context):\n    get_mtss()\n'
r = client.post('/api/code/jq/preflight', json={'code': bad})
print('bad-code status:', r.status_code, 'ok:', r.json()['ok'], 'missing:', r.json()['missing'])

# /jq/run 契约不变: 缺 start 仍应 422
r2 = client.post('/api/code/jq/run', json={'code': bad})
print('run-no-start status:', r2.status_code)
assert r2.status_code == 422

# 用户实际报错场景: 只发 code, 不带 start/end/capital
code = open('scripts/jq_repro/_validate_jq_compat.py', encoding='utf-8').read()
m = code.find("CODE = r'''")
if m >= 0:
    code = code[m + len("CODE = r'''"):code.find("'''", m + len("CODE = r'''"))]
r3 = client.post('/api/code/jq/preflight', json={'code': code})
print('user-scenario status:', r3.status_code, 'ok:', r3.json()['ok'])
print('message:', r3.json()['message'])
assert r3.status_code == 200
print('ALL PASS')
