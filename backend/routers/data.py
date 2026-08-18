from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from backend import services
from core.data import data_status, load_index


router = APIRouter()

_panel_info_cache: dict = {"ts": 0.0, "value": None}


class UpdateRequest(BaseModel):
    mode: str = "incremental"
    end: str | None = None


@router.get("/status")
def status():
    return data_status()


@router.get("/panel-info")
def panel_info():
    """面板统计走 pg_parquet/DuckDB 轻量聚合，不加载整张面板。"""
    now = time.time()
    if now - _panel_info_cache["ts"] < 300 and _panel_info_cache["value"] is not None:
        return _panel_info_cache["value"]
    try:
        from core.data import PG_PARQUET_DIR
        from core.db import query as duck_query
        stock_path = PG_PARQUET_DIR / "stock_daily.parquet"
        if stock_path.exists():
            df = duck_query(
                "SELECT (SELECT count(*) FROM stock_daily) AS n_rows, "
                "(SELECT count(*) FROM stock_basic) AS n_codes, "
                "(SELECT min(trade_date) FROM stock_daily) AS first_date, "
                "(SELECT max(trade_date) FROM stock_daily) AS last_date ",
                [],
            ).iloc[0]
            value = {
                "n_rows": int(df["n_rows"]),
                "n_codes": int(df["n_codes"]),
                "first_date": str(df["first_date"]),
                "last_date": str(df["last_date"]),
            }
            _panel_info_cache.update({"ts": now, "value": value})
            return value
    except Exception:
        pass
    try:
        from core import sqldb as pg
        if pg.configured():
            df = pg.query_df(
                # n_codes 走 stock_basic，避免对 1160 万行 stock_daily
                # 做 count(DISTINCT ...)（实测约 27s，前端直接超时）
                "SELECT (SELECT count(*) FROM stock_daily) AS n_rows, "
                "(SELECT count(*) FROM stock_basic) AS n_codes, "
                "(SELECT min(trade_date) FROM stock_daily) AS first_date, "
                "(SELECT max(trade_date) FROM stock_daily) AS last_date "
            ).iloc[0]
            value = {
                "n_rows": int(df["n_rows"]),
                "n_codes": int(df["n_codes"]),
                "first_date": str(df["first_date"]),
                "last_date": str(df["last_date"]),
            }
            _panel_info_cache.update({"ts": now, "value": value})
            return value
    except Exception:
        pass
    # 回退：parquet 列投影，仍不加载整张面板
    import pandas as pd
    from core.data import PANEL_FILE, PANEL_PATH
    path = PANEL_FILE if PANEL_FILE.exists() else PANEL_PATH
    if path.exists():
        cols = pd.read_parquet(path, columns=["date", "code"])
        value = {
            "n_rows": int(len(cols)),
            "n_codes": int(cols["code"].nunique()),
            "first_date": str(cols["date"].min().date()),
            "last_date": str(cols["date"].max().date()),
        }
        _panel_info_cache.update({"ts": now, "value": value})
        return value
    return {"n_rows": 0, "n_codes": 0, "first_date": None, "last_date": None}


@router.post("/update")
def update(req: UpdateRequest):
    if services.UPDATE_STATE.get("running"):
        return {"status": "running"}
    services.run_update_background(req.mode, req.end)
    return {"status": "started"}


@router.get("/update/status")
def update_status():
    return services.UPDATE_STATE


@router.get("/indices")
def indices():
    """大盘指数基准列表（data/index.csv：上证/沪深300/科创50/中证500/中证1000/创业板指）。"""
    try:
        idx = load_index()
    except Exception:
        return {"items": []}
    seen: dict[str, str] = {}
    for _, row in idx.iterrows():
        seen.setdefault(str(row["code"]), str(row["name"]))
    return {"items": [{"code": c, "name": n} for c, n in sorted(seen.items(), key=lambda kv: kv[1])]}


@router.get("/indices/series")
def index_series(code: str, start: str, end: str):
    """单个大盘指数归一化收盘序列（窗口起点=1），用于叠加资金曲线基准。"""
    try:
        idx = load_index()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    sub = idx[idx["code"].astype(str) == str(code)]
    if sub.empty:
        return {"ok": False, "error": f"指数 {code} 无数据"}
    sub = sub.sort_values("date")
    sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
    if sub.empty:
        return {"ok": False, "error": f"指数 {code} 区间内无数据"}
    s = sub.set_index("date")["close"].astype(float)
    if s.iloc[0] == 0 or len(s) < 2:
        return {"ok": False, "error": f"指数 {code} 序列无效"}
    s = s / s.iloc[0]
    return {
        "ok": True,
        "code": str(code),
        "name": str(sub.iloc[0]["name"]),
        "items": [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 6)}
                  for d, v in s.items()],
    }
