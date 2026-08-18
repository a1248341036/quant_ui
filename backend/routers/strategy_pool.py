"""策略池 API：全量池 / 配置池 / 回收站。"""
from __future__ import annotations

import math

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from core import strategy_pool as sp
from core.sqldb import configured


router = APIRouter(prefix="/api/strategy-pool", tags=["strategy-pool"])


class NameReq(BaseModel):
    name: str


class ReorderReq(BaseModel):
    names: list[str]


def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: _clean_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_value(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat()
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _clean_df(df) -> list[dict]:
    if df is None or df.empty:
        return []
    return [{k: _clean_value(v) for k, v in r.items()} for _, r in df.iterrows()]


def _clean_pool_row(r: dict) -> dict:
    return {k: _clean_value(v) for k, v in r.items()}


@router.get("/full")
def full():
    if not configured():
        return {"items": [], "ok": False, "error": "PG 未配置"}
    return {"items": _clean_df(sp.full_pool()), "ok": True}


@router.get("")
def pooled():
    if not configured():
        return {"items": [], "ok": False, "error": "PG 未配置"}
    return {"items": _clean_df(sp.pool_rows()), "names": sp.pool_names(), "ok": True}


@router.get("/trash")
def trash():
    if not configured():
        return {"items": [], "ok": False, "error": "PG 未配置"}
    return {"items": _clean_df(sp.trash_rows()), "ok": True}


@router.post("/add")
def add(req: NameReq):
    if not configured():
        return {"ok": False, "error": "PG 未配置"}
    try:
        ok = sp.add_from_full(req.name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not ok:
        return {"ok": False, "error": f"策略不存在或已在配置池: {req.name}"}
    return {"ok": True}


@router.post("/restore")
def restore(req: NameReq):
    if not configured():
        return {"ok": False, "error": "PG 未配置"}
    return {"ok": sp.restore_from_trash(req.name)}


@router.post("/remove")
def remove(req: NameReq):
    """从配置池移除（只影响页面策略下拉，全量池保留）。"""
    if not configured():
        return {"ok": False, "error": "PG 未配置"}
    return {"ok": sp.remove_from_pool(req.name), "moved_to_trash": False}


@router.post("/full-delete")
def full_delete(req: NameReq):
    """从全量池删除进回收站；若在配置池则同步移除。"""
    if not configured():
        return {"ok": False, "error": "PG 未配置"}
    try:
        ok = sp.delete_from_full(req.name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not ok:
        return {"ok": False, "error": f"全量池中不存在: {req.name}"}
    return {"ok": True, "moved_to_trash": True}


@router.post("/trash/purge")
def purge(req: NameReq):
    if not configured():
        return {"ok": False, "error": "PG 未配置"}
    return {"ok": sp.purge_from_trash(req.name)}


@router.post("/trash/empty")
def empty():
    if not configured():
        return {"ok": False, "error": "PG 未配置"}
    return {"ok": True, "deleted": sp.empty_trash()}


@router.post("/reorder")
def reorder(req: ReorderReq):
    if not configured():
        return {"ok": False, "error": "PG 未配置"}
    sp.reorder_pool(req.names)
    return {"ok": True}
