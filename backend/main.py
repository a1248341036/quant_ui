from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import auth
from backend.routers import backtest, data, ledger

OPEN_API_PATHS = {"/api/health", "/api/auth/login"}


app = FastAPI(title="quant_ui API", version="0.2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(backtest.router, prefix="/api", tags=["backtest"])
app.include_router(ledger.router, prefix="/api/ledger", tags=["ledger"])


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.url.path not in OPEN_API_PATHS:
        if not auth.verify_token(request.cookies.get(auth.COOKIE, "")):
            return JSONResponse({"detail": "未登录"}, status_code=401)
    return await call_next(request)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
