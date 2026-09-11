from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.logging_config import api_logger, main_logger, setup_root_logger
from backend.routers import (alphaagent, backtest, code, data, ledger, paper,
                             sentiment, stock, strategy_pool)


# 设置根日志器
setup_root_logger()

app = FastAPI(title="quant_ui API", version="0.2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(backtest.router, prefix="/api", tags=["backtest"])
app.include_router(ledger.router, prefix="/api/ledger", tags=["ledger"])
app.include_router(paper.router, prefix="/api", tags=["paper"])
app.include_router(code.router, tags=["code"])
app.include_router(sentiment.router)
app.include_router(stock.router, prefix="/api/stock", tags=["stock"])
app.include_router(strategy_pool.router, tags=["strategy-pool"])
app.include_router(alphaagent.router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录 HTTP 请求日志，包含请求 ID、耗时、状态码。"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # 注入 request_id 到 headers，供下游使用
    request.state.request_id = request_id
    
    # 跳过健康检查和静态文件
    path = request.url.path
    if path not in ["/api/health", "/"] and not path.startswith("/"):
        api_logger.info(f"Incoming request: {request.method} {path}")
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # 记录响应日志
        if path not in ["/api/health", "/"] and not path.startswith("/"):
            status = response.status_code
            if status >= 400:
                api_logger.warning(
                    f"Request {request_id} {request.method} {path} "
                    f"completed with status {status} in {duration_ms:.0f}ms"
                )
            else:
                api_logger.info(
                    f"Request {request_id} {request.method} {path} "
                    f"completed with status {status} in {duration_ms:.0f}ms"
                )
        
        # 在响应头中添加 request_id
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        main_logger.error(
            f"Request {request_id} {request.method} {path} failed: {e} "
            f"in {duration_ms:.0f}ms",
            exc_info=True
        )
        raise


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ---------- 公网访问门禁（经 Cloudflare Tunnel 暴露时使用） ----------
# 浏览器访问：未授权渲染内置登录页，输对密码种 HttpOnly Cookie（一年免输）；
# 非 HTML 客户端（API/curl）：维持 401 JSON。?key= 查询参数兼容保留。
_ACCESS_KEY = os.environ.get("QUANT_UI_ACCESS_KEY", "")
_ACCESS_COOKIE = "qk"
_GATE_LOGIN_PATH = "/_gate/login"


def _client_host(request: Request) -> str:
    # 经 Tunnel 到达的请求 TCP 对端是本机 cloudflared，须以边缘注入的
    # CF-Connecting-IP 判定真实来源，否则回环直通会放行全部公网流量。
    return request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else ""
    )


_GATE_LOGIN_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>访问验证 - Quant UI</title>
<style>
  body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         background:#0f1115; color:#e6e6e6; font-family:system-ui,-apple-system,'Segoe UI',sans-serif; }
  .card { width:min(360px,90vw); padding:32px 28px; border-radius:14px;
          background:#171a21; box-shadow:0 8px 30px rgba(0,0,0,.45); }
  h1 { margin:0 0 6px; font-size:20px; }
  p  { margin:0 0 20px; font-size:13px; color:#8b93a1; }
  input[type=password] { width:100%; box-sizing:border-box; padding:11px 12px; border-radius:8px;
          border:1px solid #2c3340; background:#0f1115; color:#e6e6e6; font-size:15px; outline:none; }
  input[type=password]:focus { border-color:#4c8dff; }
  button { width:100%; margin-top:14px; padding:11px; border:0; border-radius:8px;
           background:#4c8dff; color:#fff; font-size:15px; cursor:pointer; }
  button:hover { background:#3d7bf0; }
  .err { margin:0 0 12px; font-size:13px; color:#ff6b6b; }
</style>
</head>
<body>
<form class="card" method="post" action="/_gate/login">
  <h1>🔒 访问验证</h1>
  <p>请输入访问密码后继续</p>
  __ERR__
  <input type="password" name="password" placeholder="访问密码" autocomplete="current-password" autofocus required>
  <input type="hidden" name="next" value="__NEXT__">
  <button type="submit">进 入</button>
</form>
</body>
</html>"""


def _gate_render(next_path: str, error: bool) -> HTMLResponse:
    esc = next_path.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    body = _GATE_LOGIN_HTML.replace("__NEXT__", esc)
    body = body.replace("__ERR__", '<p class="err">密码错误，请重试</p>' if error else "")
    return HTMLResponse(body, status_code=200)


def _safe_next(next_path: str) -> str:
    # 只接受站内相对路径，防开放重定向
    return next_path if next_path.startswith("/") and not next_path.startswith("//") else "/"


@app.middleware("http")
async def access_gate(request: Request, call_next):
    if not _ACCESS_KEY:
        return await call_next(request)  # 未配置密钥 = 不启用门禁
    if _client_host(request) in ("127.0.0.1", "::1"):
        return await call_next(request)
    if request.url.path == "/api/health":
        return await call_next(request)
    if request.cookies.get(_ACCESS_COOKIE) == _ACCESS_KEY:
        return await call_next(request)
    provided = request.query_params.get("key", "")
    if provided == _ACCESS_KEY:
        response = await call_next(request)
        response.set_cookie(
            _ACCESS_COOKIE,
            _ACCESS_KEY,
            max_age=365 * 24 * 3600,
            httponly=True,
            samesite="lax",
        )
        return response
    # 登录表单提交在此短路处理（不进入业务路由）
    if request.method == "POST" and request.url.path == _GATE_LOGIN_PATH:
        try:
            form = await request.form()
            password = str(form.get("password", ""))
            nxt = _safe_next(str(form.get("next") or "/"))
        except Exception:
            password, nxt = "", "/"
        if password == _ACCESS_KEY:
            resp = RedirectResponse(nxt, status_code=303)
            resp.set_cookie(
                _ACCESS_COOKIE,
                _ACCESS_KEY,
                max_age=365 * 24 * 3600,
                httponly=True,
                samesite="lax",
            )
            return resp
        return _gate_render(nxt, error=True)
    # 浏览器访问 → 登录页（记住原地址，登录后跳回）；其余 → 401 JSON
    if "text/html" in request.headers.get("accept", ""):
        nxt = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        return _gate_render(nxt, error=False)
    return JSONResponse({"detail": "unauthorized"}, status_code=401)


@app.on_event("startup")
def _ensure_db_schema() -> None:
    """确保本地 DuckDB 归档表（backtest_runs/ledger/paper_*）存在，幂等。"""
    from core import sqldb as duck_store
    duck_store.create_schema()
    from backend.alphaagent_service import bootstrap_research_memory
    from core.backtest_archive import backfill_excess_metrics
    bootstrap_research_memory()
    backfill_excess_metrics()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/trading-defaults")
def trading_defaults() -> dict:
    """返回统一交易参数默认值，供前端初始化表单。"""
    from core.trading_config import defaults
    return defaults()


static_dir = Path(__file__).resolve().parent.parent / "static" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
