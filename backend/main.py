from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import (alphaagent, backtest, code, data, ledger, paper,
                             sentiment, stock, strategy_pool)


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
