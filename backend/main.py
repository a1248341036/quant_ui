from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import (backtest, code, data, ledger, paper, sentiment,
                             stock, strategy_pool)


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


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
