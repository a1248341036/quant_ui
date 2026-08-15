from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend import services
from core.ledger import (add_deposit, add_transaction, compute_equity,
                         current_positions, load_deposits, load_transactions)


router = APIRouter()


class TransactionRequest(BaseModel):
    date: str
    code: str
    name: str = ""
    action: str
    shares: float
    price: float
    fee: float = 0.0
    note: str = ""


class DepositRequest(BaseModel):
    date: str
    amount: float
    note: str = ""


@router.get("/transactions")
def transactions():
    nm = services.get_name_map()
    rows = load_transactions().to_dict(orient="records")
    for r in rows:
        if not r.get("name"):
            r["name"] = nm.get(str(r.get("code", "")), "")
    return rows


@router.post("/transactions")
def add_tx(req: TransactionRequest):
    add_transaction(req.date, req.code, req.name, req.action,
                    req.shares, req.price, req.fee, req.note)
    return {"status": "ok"}


@router.get("/deposits")
def deposits():
    return load_deposits().to_dict(orient="records")


@router.post("/deposits")
def add_dep(req: DepositRequest):
    add_deposit(req.date, req.amount, req.note)
    return {"status": "ok"}


@router.get("/equity")
def equity():
    panel = services.load_data()["panel"]
    eq = compute_equity(panel)
    if eq.empty:
        return {"items": [], "summary": None}
    return {
        "items": eq[["date", "cash", "market_value", "equity", "pnl", "pnl_pct"]]
                  .to_dict(orient="records"),
        "summary": {
            "latest_equity": float(eq["equity"].iloc[-1]),
            "cash": float(eq["cash"].iloc[-1]),
            "market_value": float(eq["market_value"].iloc[-1]),
            "pnl": float(eq["pnl"].iloc[-1]),
            "pnl_pct": float(eq["pnl_pct"].iloc[-1]),
        },
    }


@router.get("/positions")
def positions():
    panel = services.load_data()["panel"]
    pos = current_positions(panel)
    rows = services.clean_records(pos.to_dict(orient="records"))
    nm = services.get_name_map()
    for r in rows:
        if not r.get("name"):
            r["name"] = nm.get(str(r.get("code", "")), "")
    return rows
