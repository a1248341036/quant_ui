"""quant-data-status FastAPI service (port 8001).

Provides:
  GET  /api/status            full data status snapshot
  POST /api/status/refresh    force collector run
  GET  /api/tasks             script task history
  POST /api/tasks             submit a script task
  GET  /api/tasks/{id}        task detail + log
  POST /api/tasks/{id}/cancel cancel queued/running task
  GET  /api/query?sql=...     read-only SQL on PostgreSQL
  GET  /                      minimal HTML dashboard
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .collector import collect_all, ensure_logs_dir
from .config import TABLE_DATE_COLUMN, PG_TABLES
from .state import load_state
from .tasks import TASK_DEFS, get_runner


COLLECT_INTERVAL = int(os.getenv("QUANT_DATA_STATUS_INTERVAL", "300"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_logs_dir()
    get_runner()

    def _loop():
        while True:
            try:
                collect_all(alert=True)
            except Exception as exc:
                print(f"[collector] error: {exc}", flush=True)
            time.sleep(COLLECT_INTERVAL)

    thread = threading.Thread(target=_loop, daemon=True, name="collector")
    thread.start()
    yield


app = FastAPI(title="quant-data-status", version="0.1.0", lifespan=lifespan)


class TaskRequest(BaseModel):
    type: str
    params: dict = Field(default_factory=dict)


@app.get("/api/status")
def status() -> dict:
    state = load_state()
    if not state:
        return {"overall": "unknown", "detail": "collector has not run yet"}
    return state


@app.post("/api/status/refresh")
def refresh_status() -> dict:
    try:
        return collect_all(alert=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/tasks")
def tasks_list() -> dict:
    return {"tasks": get_runner().list()}


@app.post("/api/tasks")
def task_submit(req: TaskRequest) -> dict:
    try:
        task = get_runner().submit(req.type, req.params)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return task


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str):
    task = get_runner().get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    log = ""
    try:
        log = open(task.get("log", ""), encoding="utf-8", errors="replace").read()[-20000:]
    except OSError:
        pass
    return {**task, "log_tail": log}


@app.post("/api/tasks/{task_id}/cancel")
def task_cancel(task_id: str) -> dict:
    return get_runner().cancel(task_id)


@app.get("/api/query")
def query(sql: str):
    sql = sql.strip().rstrip(";")
    up = sql.upper()
    if not (up.startswith("SELECT") or up.startswith("WITH")):
        raise HTTPException(status_code=400, detail="only SELECT/WITH allowed")
    if ";" in sql:
        raise HTTPException(status_code=400, detail="multiple statements not allowed")
    if " LIMIT " not in up:
        sql += " LIMIT 1000"
    try:
        import psycopg
        dsn = os.getenv("PG_DSN", "").strip()
        if not dsn:
            raise RuntimeError("PG_DSN not configured")
        with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [d.name for d in cur.description] if cur.description else []
                rows = cur.fetchall()
        return {"columns": cols, "rows": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/defs")
def defs() -> dict:
    return {"task_types": sorted(TASK_DEFS), "tables": PG_TABLES,
            "date_columns": TABLE_DATE_COLUMN}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots() -> str:
    return "User-agent: *\nDisallow: /\n"


_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>quant-data-status</title>
<style>
body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0f1115;color:#e6e6e6;margin:0;padding:24px}
h1{font-size:18px;margin:0 0 4px}
h2{font-size:14px;margin:24px 0 8px;color:#9aa0aa}
.meta{color:#7d8590;font-size:12px;margin-bottom:16px}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:bold}
.ok{background:#1b4332;color:#6ee7a8}
.warn{background:#5b3a10;color:#fbbf24}
.critical,.error{background:#5c1414;color:#ff7b7b}
.unknown{background:#333;color:#aaa}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border:1px solid #2a2f3a;padding:5px 8px;text-align:left}
th{background:#161b24;color:#9aa0aa;position:sticky;top:0}
tr:nth-child(even){background:#12161e}
pre{background:#0a0c10;border:1px solid #2a2f3a;padding:12px;font-size:12px;overflow:auto;max-height:300px}
.card{border:1px solid #2a2f3a;border-radius:8px;padding:12px 16px;margin-bottom:16px}
button{background:#1d4ed8;color:#fff;border:0;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:13px}
input,select{padding:6px 8px;background:#0a0c10;border:1px solid #2a2f3a;color:#e6e6e6;border-radius:6px;font-size:13px}
.flex{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
</style>
</head>
<body>
<h1>quant-data-status</h1>
<div class="meta">状态快照 + 脚本任务调度 · 自动采集每 <span id="interval"></span>s</div>
<div class="flex"><button onclick="refresh()">手动刷新状态</button><span id="updated"></span></div>
<div id="root">loading...</div>

<h2>提交脚本任务</h2>
<div class="card">
<div class="flex">
<select id="task-type"></select>
<button onclick="submitTask()">提交</button>
</div>
<pre id="task-result" style="display:none"></pre>
</div>

<h2>任务列表</h2>
<div id="tasks">loading...</div>

<script>
const API='/api';
async function j(url, opts){const r=await fetch(url,opts);if(!r.ok)throw new Error((await r.text()).slice(0,300));return r.json();}
function esc(s){const d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;}
function badge(s){const c=['ok','warn','critical','error','unknown'].includes(s)?s:'unknown';return `<span class="badge ${c}">${esc(s||'unknown')}</span>`;}
async function refresh(){
  document.getElementById('root').innerHTML='刷新中...';
  const s=await j(API+'/status/refresh',{method:'POST'});
  render(s);
}
function render(s){
  document.getElementById('updated').textContent='采集时间 '+s.updated_at+' · 耗时 '+s.elapsed_s+'s · 状态 '+badge(s.overall);
  let html='<div class="card">'+badge(s.overall)+' <span class="meta">更新时间 '+esc(s.updated_at)+'，耗时 '+esc(s.elapsed_s)+'s</span></div>';
  const g=s.groups||{};
  // resources
  const res=g.resources||{};
  if(res.memory){html+='<div class="card"><b>内存</b> 已用 '+esc(res.memory.used_pct)+'% · 可用 '+esc(Math.round(res.memory.available_bytes/1024/1024/1024*100)/100)+'GB</div>';}
  if(res.disk){html+='<div class="card"><b>磁盘</b> 可用 '+esc(res.disk.free_pct)+'% · '+esc(Math.round(res.disk.free_bytes/1024/1024/1024*100)/100)+'GB</div>';}
  if(res.pg_container){html+='<div class="card"><b>quant-pg</b> '+esc(res.pg_container)+'</div>';}
  // pg tables
  const pg=g.pg||{};
  html+='<h2>PostgreSQL ('+badge(pg.status)+')</h2><table><tr><th>表</th><th>行数</th><th>最早</th><th>最新</th><th>大小</th></tr>';
  for(const [name,t] of Object.entries(pg.tables||{})){
    html+=`<tr><td>${esc(name)}</td><td>${esc(t.rows)}</td><td>${esc(t.min_date)}</td><td>${esc(t.max_date)}</td><td>${t.size_bytes?esc(Math.round(t.size_bytes/1048576))+'MB':'—'}</td></tr>`;
  }
  html+='</table>';
  if(pg.daily_coverage&&pg.daily_coverage.length){
    html+='<h2>最近8个交易日覆盖</h2><table><tr><th>日期</th><th>股票数</th></tr>';
    for(const d of pg.daily_coverage)html+=`<tr><td>${esc(d.date)}</td><td>${esc(d.stocks)}</td></tr>`;
    html+='</table>';
  }
  // parquet
  const pq=g.parquet||{};
  html+='<h2>Parquet ('+badge(pq.status)+')</h2><table><tr><th>表</th><th>行数</th><th>最新</th><th>大小</th><th>mtime</th></tr>';
  for(const [name,f] of Object.entries(pq.files||{})){
    html+=`<tr><td>${esc(name)}</td><td>${esc(f.rows)}</td><td>${esc(f.max_date)}</td><td>${f.size_bytes?esc(Math.round(f.size_bytes/1048576))+'MB':'—'}</td><td>${esc(f.mtime)}</td></tr>`;
  }
  html+='</table>';
  // sources
  const src=g.sources||{};
  html+='<h2>数据源同步 ('+badge(src.status)+')</h2><div class="card">日历最新 '+esc(src.calendar_max)+' · stock_daily 最新 '+esc(src.stock_daily_max)+' · 滞后 '+esc(src.lag_days)+' 天</div>';
  // services
  const svc=g.services||{};
  html+='<h2>服务/定时器 ('+badge(svc.status)+')</h2><table><tr><th>项目</th><th>状态</th><th>详情</th></tr>';
  for(const [name,c] of Object.entries(svc.checks||{})){
    html+=`<tr><td>${esc(name)}</td><td>${c.ok!==false?badge('ok'):badge('error')}</td><td>${esc(JSON.stringify(c))}</td></tr>`;
  }
  html+='</table>';
  document.getElementById('root').innerHTML=html;
}
async function loadTasks(){
  const r=await j(API+'/tasks');
  document.getElementById('tasks').innerHTML='<table><tr><th>id</th><th>类型</th><th>状态</th><th>开始</th><th>结束</th><th>退出码</th><th></th></tr>'+
    r.tasks.map(t=>`<tr><td>${esc(t.id)}</td><td>${esc(t.type)}</td><td>${badge(t.status)}</td><td>${esc(t.started_at||'')}</td><td>${esc(t.finished_at||'')}</td><td>${esc(t.exit_code==null?'':t.exit_code)}</td><td><a href="#" onclick="showTask('${esc(t.id)}')">日志</a></td></tr>`).join('')+'</table>';
}
async function showTask(id){
  const t=await j(API+'/tasks/'+id);
  alert('['+t.status+'] '+t.argv.join(' ')+'\n\n'+(t.log_tail||'(no log)'));
}
async function submitTask(){
  const type=document.getElementById('task-type').value;
  const params={};
  if(type==='sync-daily'){const d=prompt('起始日期 YYYYMMDD');if(!d)return;params.date=d;}
  const r=await j(API+'/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type,params})});
  const pre=document.getElementById('task-result');pre.style.display='block';
  pre.textContent='已提交 '+r.id+' / '+r.type+' -> '+r.argv.join(' ');
  loadTasks();
}
async function init(){
  document.getElementById('interval').textContent='300';
  const d=await j(API+'/defs');
  document.getElementById('task-type').innerHTML=d.task_types.map(t=>`<option value="${t}">${t}</option>`).join('');
  const s=await j(API+'/status');
  render(s);
  loadTasks();
  setInterval(async()=>{try{render(await j(API+'/status'));loadTasks();}catch(e){}},15000);
}
init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("data_status.app:app", host="0.0.0.0", port=8001)
