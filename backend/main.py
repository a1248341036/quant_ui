from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
# 首次访问带 ?key=<QUANT_UI_ACCESS_KEY> 校验通过后种 HttpOnly Cookie，
# 页面内后续 API 请求自动携带 Cookie 放行；本机回环来源直接放行。
_ACCESS_KEY = os.environ.get("QUANT_UI_ACCESS_KEY", "")
_ACCESS_COOKIE = "qk"


def _client_host(request: Request) -> str:
    # 经 Tunnel 到达的请求 TCP 对端是本机 cloudflared，须以边缘注入的
    # CF-Connecting-IP 判定真实来源，否则回环直通会放行全部公网流量。
    return request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else ""
    )


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
