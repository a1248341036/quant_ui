from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, Query

from backend import services
from core.data import load_stock_detail


router = APIRouter()


def _f(x) -> float | None:
    """numpy 标量 -> python float/None，保证 JSON 可序列化。"""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@router.get("/search")
def search(q: str = Query(..., min_length=1, max_length=32), limit: int = Query(20, ge=1, le=50)):
    """按代码前缀或名称子串搜索，返回带补全的候选列表。"""
    name_map = services.get_name_map()
    industry_map = services.get_industry_map()
    q = q.strip().lower()
    if not q:
        return {"items": []}
    code_matches: list[dict] = []
    name_matches: list[dict] = []
    for code, name in name_map.items():
        c = str(code).zfill(6)
        n = str(name or "")
        if c.startswith(q):
            code_matches.append({"code": c, "name": n,
                                 "industry": industry_map.get(c, "")})
        elif q in n.lower():
            name_matches.append({"code": c, "name": n,
                                 "industry": industry_map.get(c, "")})
    items = (code_matches + name_matches)[:limit]
    return {"items": items}


@router.get("/{code}")
def detail(code: str, days: int = Query(250, ge=10, le=2000)):
    """单只股票历史行情 + 最新报价。code 为 6 位代码。"""
    code = str(code).zfill(6)
    name_map = services.get_name_map()
    industry_map = services.get_industry_map()

    sub = load_stock_detail(code, days)
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"股票 {code} 不在面板数据中")

    history = []
    for _, r in sub.iterrows():
        history.append({
            "date": str(r["date"].date()),
            "open": _f(r["open"]),
            "high": _f(r["high"]),
            "low": _f(r["low"]),
            "close": _f(r["close"]),
            "volume": _f(r["volume"]),
            "amount": _f(r["amount"]),
            "turnover": _f(r["turnover"]),
            "turn20": _f(r["turn20"]),
            "am20": _f(r["am20"]),
        })

    latest = history[-1] if history else None
    change_pct = None
    if len(history) >= 2 and history[-1].get("close") and history[-2].get("close"):
        prev = history[-2]["close"]
        if prev:
            change_pct = round((history[-1]["close"] / prev - 1.0) * 100.0, 2)
    if latest:
        latest["change_pct"] = change_pct

    return {
        "code": code,
        "name": name_map.get(code, ""),
        "industry": industry_map.get(code, ""),
        "start": history[0]["date"] if history else None,
        "end": history[-1]["date"] if history else None,
        "n_days": len(history),
        "latest": latest,
        "history": history,
    }
