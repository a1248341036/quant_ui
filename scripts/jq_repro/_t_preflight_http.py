# -*- coding: utf-8 -*-
"""对真实 HTTP 服务验证 /api/code/jq/preflight 只发 code 的场景。"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

BASE = 'http://127.0.0.1:17891'

src = open('scripts/jq_repro/_validate_jq_compat.py', encoding='utf-8').read()
m = src.find("CODE = '''")
code = src[m + len("CODE = '''"):src.find("'''", m)]

def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

st, body = post('/api/code/jq/preflight', {'code': code})
print('user-scenario status:', st)
print('ok:', body.get('ok'), '| missing:', body.get('missing'))
print('message:', body.get('message'))
assert st == 200 and body['ok'] and not body['missing']

st2, body2 = post('/api/code/jq/run', {'code': code})
print('run-no-start status:', st2, '(expect 422)')
assert st2 == 422

print('ALL PASS')
