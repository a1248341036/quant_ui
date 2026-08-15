from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend import services
from core.data import data_status


router = APIRouter()


class UpdateRequest(BaseModel):
    mode: str = "incremental"
    end: str | None = None


@router.get("/status")
def status():
    return data_status()


@router.get("/panel-info")
def panel_info():
    data = services.load_data()
    panel = data["panel"]
    return {
        "n_rows": int(len(panel)),
        "n_codes": int(panel["code"].nunique()),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
    }


@router.post("/update")
def update(req: UpdateRequest):
    if services.UPDATE_STATE.get("running"):
        return {"status": "running"}
    services.run_update_background(req.mode, req.end)
    return {"status": "started"}


@router.get("/update/status")
def update_status():
    return services.UPDATE_STATE
