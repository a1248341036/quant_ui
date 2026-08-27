from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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
