"""明细读取与写入。"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .storage import (
    _q, _ex, _ex_id, _json_state, _json_save, _json_next_id,
    _ensure_columns,
)
from .. import sqldb as pg


# ---------- 明细读取 ----------

def account_orders(account_id: int) -> list[dict]:
    if pg.configured():
        try:
            df = _q("SELECT * FROM paper_orders WHERE account_id=%s"
                    " ORDER BY exec_date DESC, id", (int(account_id),))
            return df.to_dict(orient="records")
        except Exception as exc:
            print(f"[paper] SQLite 读取失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    return [o for o in state.get("orders", []) if o.get("account_id") == account_id]


def account_trades(account_id: int) -> list[dict]:
    if pg.configured():
        try:
            df = _q("SELECT * FROM paper_trades WHERE account_id=%s"
                    " ORDER BY exec_date DESC, id", (int(account_id),))
            return df.to_dict(orient="records")
        except Exception as exc:
            print(f"[paper] SQLite 读取失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    return [t for t in state.get("trades", []) if t.get("account_id") == account_id]


def account_positions(account_id: int) -> list[dict]:
    if pg.configured():
        try:
            df = _q("SELECT * FROM paper_positions WHERE account_id=%s"
                    " ORDER BY shares DESC", (int(account_id),))
            return df.to_dict(orient="records")
        except Exception as exc:
            print(f"[paper] SQLite 读取失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    return sorted(
        [p for p in state.get("positions", {}).get(str(account_id), {}).values()
         if p.get("shares", 0) > 0], key=lambda p: -p["shares"])


def account_equity(account_id: int) -> list[dict]:
    if pg.configured():
        try:
            df = _q("SELECT * FROM paper_equity_snapshots WHERE account_id=%s"
                    " ORDER BY date", (int(account_id),))
            return df.to_dict(orient="records")
        except Exception as exc:
            print(f"[paper] SQLite 读取失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    snaps = state.get("snapshots", {}).get(str(account_id), {})
    return [snaps[k] for k in sorted(snaps)]


def account_events(account_id: int) -> list[dict]:
    if pg.configured():
        try:
            df = _q("SELECT * FROM paper_events WHERE account_id=%s"
                    " ORDER BY date DESC, id DESC", (int(account_id),))
            return df.to_dict(orient="records")
        except Exception as exc:
            print(f"[paper] SQLite 读取失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    return [e for e in state.get("events", []) if e.get("account_id") == account_id]


def account_orders_with_names(account_id: int) -> list[dict]:
    rows = account_orders(account_id)
    # 历史订单在 shares 列加入前生成，用成交表按 order_id 回填股数/金额
    fills_by_order: dict[int, dict] = {}
    try:
        for tr in account_trades(account_id):
            oid = tr.get("order_id")
            if oid is not None and tr.get("shares") is not None:
                fills_by_order[int(oid)] = {
                    "shares": float(tr["shares"]),
                    "price": float(tr["price"]),
                }
    except Exception:
        fills_by_order = {}
    nm = _name_map()
    for r in rows:
        r["name"] = nm.get(str(r.get("code", "")), "")
        sh = r.get("shares")
        px = r.get("fill_price")
        if (sh is None or px is None) and r.get("status") == "filled":
            fb = fills_by_order.get(int(r["id"]))
            if fb:
                sh, px = fb["shares"], fb["price"]
                r["shares"] = sh
                r["fill_price"] = px
        if sh is not None and px is not None and float(sh) > 0:
            r["amount"] = float(sh) * float(px)
        else:
            r["amount"] = None
    return rows


def account_trades_with_names(account_id: int) -> list[dict]:
    rows = account_trades(account_id)
    nm = _name_map()
    for r in rows:
        r["name"] = nm.get(str(r.get("code", "")), "")
        r["amount"] = float(r["shares"]) * float(r["price"])
    return rows


def _name_map() -> dict[str, str]:
    try:
        from ..data import load_etf, load_fund, load_tech, load_universe
        m = {}
        for df in (load_universe(), load_tech()):
            if "code" in df and "name" in df:
                for c, n in zip(df["code"], df["name"]):
                    c = str(c).zfill(6)
                    if n and not pd.isna(n):
                        m.setdefault(c, str(n))
        # ETF/场外基金代码与股票池不重叠，后写覆盖保证模拟盘显示正确的产品名
        for loader in (load_etf, load_fund):
            try:
                df = loader()
            except Exception:
                continue
            for c, n in zip(df["code"], df["name"]):
                c = str(c).zfill(6)
                if n and not pd.isna(n):
                    m[c] = str(n)
        return m
    except Exception:
        return {}


# ---------- 写入明细 ----------

def _add_order(account_id: int, order: dict) -> int:
    if pg.configured():
        try:
            return _ex_id(
                "INSERT INTO paper_orders"
                "(account_id, code, side, target_pct, signal_date, exec_date,"
                " status, shares, fill_price, fee, reject_reason)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (account_id, order["code"], order["side"],
                 order.get("target_pct"), order["signal_date"], order["exec_date"],
                 order["status"], order.get("shares"), order.get("fill_price"),
                 order.get("fee", 0),
                 order.get("reject_reason")),
            )
        except Exception as exc:
            if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                return 0  # 幂等：重复订单直接忽略
            print(f"[paper] SQLite 订单写入失败: {exc}", flush=True)
    state = _json_state()
    oid = _json_next_id(state, "orders")
    order = dict(order)
    order["id"] = oid
    order["account_id"] = account_id
    order["created_at"] = datetime.now().isoformat()
    state.setdefault("orders", []).append(order)
    _json_save(state)
    return oid


def _add_trade(account_id: int, trade: dict) -> None:
    if pg.configured():
        try:
            _ex("INSERT INTO paper_trades"
                "(account_id, order_id, exec_date, code, side, shares, price, fee, note)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (account_id, trade.get("order_id"), trade["exec_date"],
                 trade["code"], trade["side"], float(trade["shares"]),
                 float(trade["price"]), float(trade.get("fee", 0)),
                 trade.get("note")))
            return
        except Exception as exc:
            print(f"[paper] SQLite 成交写入失败: {exc}", flush=True)
    state = _json_state()
    trade = dict(trade)
    trade["id"] = _json_next_id(state, "trades")
    trade["account_id"] = account_id
    state.setdefault("trades", []).append(trade)
    _json_save(state)


def _set_position(account_id: int, code: str, shares: float,
                  avg_cost: float, date: str) -> None:
    if pg.configured():
        try:
            if shares <= 1e-6:
                _ex("DELETE FROM paper_positions WHERE account_id=%s AND code=%s",
                    (account_id, code))
            else:
                _ex("INSERT INTO paper_positions"
                    "(account_id, code, shares, avg_cost, updated_date)"
                    " VALUES (%s,%s,%s,%s,%s)"
                    " ON CONFLICT (account_id, code)"
                    " DO UPDATE SET shares=EXCLUDED.shares,"
                    " avg_cost=EXCLUDED.avg_cost, updated_date=EXCLUDED.updated_date",
                    (account_id, code, float(shares), float(avg_cost), date))
            return
        except Exception as exc:
            print(f"[paper] SQLite 持仓写入失败: {exc}", flush=True)
    state = _json_state()
    pos = state.setdefault("positions", {}).setdefault(str(account_id), {})
    if shares <= 1e-6:
        pos.pop(code, None)
    else:
        pos[code] = {"account_id": account_id, "code": code, "shares": float(shares),
                     "avg_cost": float(avg_cost), "updated_date": date}
    _json_save(state)


def _load_positions(account_id: int) -> dict[str, dict]:
    rows = account_positions(account_id)
    return {str(r["code"]).zfill(6): r for r in rows}


def _add_snapshot(account_id: int, snap: dict) -> None:
    if pg.configured():
        try:
            _ex("INSERT INTO paper_equity_snapshots"
                "(account_id, date, cash, market_value, equity, pnl, pnl_pct)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (account_id, date)"
                " DO UPDATE SET cash=EXCLUDED.cash, market_value=EXCLUDED.market_value,"
                " equity=EXCLUDED.equity, pnl=EXCLUDED.pnl, pnl_pct=EXCLUDED.pnl_pct",
                (account_id, snap["date"], float(snap["cash"]),
                 float(snap["market_value"]), float(snap["equity"]),
                 float(snap["pnl"]), float(snap["pnl_pct"])))
            return
        except Exception as exc:
            print(f"[paper] SQLite 估值写入失败: {exc}", flush=True)
    state = _json_state()
    snaps = state.setdefault("snapshots", {}).setdefault(str(account_id), {})
    snaps[str(snap["date"])] = dict(snap)
    _json_save(state)


def _add_event(account_id: int, date: str, level: str, msg: str) -> None:
    if pg.configured():
        try:
            _ex("INSERT INTO paper_events (account_id, date, level, msg)"
                " VALUES (%s,%s,%s,%s)",
                (account_id, date, level, msg[:2000]))
            return
        except Exception as exc:
            print(f"[paper] SQLite 事件写入失败: {exc}", flush=True)
    state = _json_state()
    ev = {"id": _json_next_id(state, "events"), "account_id": account_id,
          "date": date, "level": level, "msg": msg[:2000],
          "created_at": datetime.now().isoformat()}
    state.setdefault("events", []).append(ev)
    _json_save(state)


def _update_account_dates(account_id: int, processed: str | None,
                          rebalance: str | None) -> None:
    if pg.configured():
        try:
            _ex("UPDATE paper_accounts SET last_processed_date=%s,"
                " last_rebalance_date=%s WHERE id=%s",
                (processed, rebalance, account_id))
            return
        except Exception as exc:
            print(f"[paper] SQLite 账户日期更新失败: {exc}", flush=True)
    state = _json_state()
    acc = state["accounts"].get(str(account_id))
    if acc is not None:
        acc["last_processed_date"] = processed
        acc["last_rebalance_date"] = rebalance
        _json_save(state)
