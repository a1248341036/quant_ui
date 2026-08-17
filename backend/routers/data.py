from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from backend import services
from core.data import data_status


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
    """面板统计走 PG 轻量聚合（不加载整张面板，避免触发 800MB 全量加载）。"""
    now = time.time()
    if now - _panel_info_cache["ts"] < 60 and _panel_info_cache["value"] is not None:
        return _panel_info_cache["value"]
    try:
        from core import pg
        if pg.configured():
            df = pg.query_df(
                "SELECT count(*) AS n_rows, "
                "count(DISTINCT substr(ts_code, 1, 6)) AS n_codes, "
                "min(trade_date) AS first_date, max(trade_date) AS last_date "
                "FROM stock_daily WHERE close IS NOT NULL"
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
