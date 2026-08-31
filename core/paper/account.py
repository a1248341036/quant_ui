"""账户管理 CRUD。"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .storage import (
    _q, _ex, _ex_id, _json_state, _json_save, _acc_row,
    _ensure_columns, DEFAULT_RISK, PAPER_DIR,
)
from ..store import normalize_universe
from .. import sqldb as pg


def create_account(
    name: str,
    strategy_name: str,
    factor: str,
    ascending: bool,
    universe: str = "科技TMT",
    capital: float = 100000.0,
    top_n: int = 3,
    freq: str = "monthly",
    risk_config: dict | None = None,
    strategy_type: str = "factor",
    module: str | None = None,
    event_strategy: str | None = None,
    start_date: str | None = None,
) -> dict:
    if not name.strip():
        raise ValueError("账户名称不能为空")
    universe = normalize_universe(universe)
    if freq not in ("daily", "weekly", "monthly", "semiannual"):
        raise ValueError("freq 仅支持 daily/weekly/monthly/semiannual")
    if strategy_type not in ("factor", "event"):
        raise ValueError("strategy_type 仅支持 factor/event")
    if strategy_type == "event" and (not module or not event_strategy):
        raise ValueError("事件策略账户需要 module 与 event_strategy")
    if start_date:
        pd.Timestamp(start_date)  # 校验格式
    _ensure_columns()
    risk = {**DEFAULT_RISK, **(risk_config or {})}
    if universe == "ETF":
        from ..assets import ETF_PROFILE
        supplied = risk_config or {}
        if "spread_bps" not in supplied:
            risk["spread_bps"] = ETF_PROFILE.spread_bps
        if "min_commission" not in supplied:
            risk["min_commission"] = ETF_PROFILE.min_commission
    if pg.configured():
        try:
            aid = _ex_id(
                "INSERT INTO paper_accounts"
                "(name, status, strategy_type, strategy_name, factor, ascending,"
                " module, event_strategy, start_date, universe, capital, top_n,"
                " freq, risk_config)"
                " VALUES (%s,'active',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (name.strip(), strategy_type, strategy_name, factor,
                 bool(ascending), module, event_strategy, start_date, universe,
                 float(capital), int(top_n), freq,
                 json.dumps(risk, ensure_ascii=False)),
            )
            return get_account(aid)
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                raise ValueError(f"账户名称已存在: {name}") from exc
            print(f"[paper] SQLite 写入失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    if any(str(a.get("name")) == name.strip()
           for a in state["accounts"].values()):
        raise ValueError(f"账户名称已存在: {name}")
    aid = 1
    while str(aid) in state["accounts"]:
        aid += 1
    acc = {
        "id": aid, "name": name.strip(), "status": "active",
        "strategy_type": strategy_type,
        "strategy_name": strategy_name, "factor": factor,
        "ascending": bool(ascending), "universe": universe,
        "module": module, "event_strategy": event_strategy,
        "start_date": start_date,
        "capital": float(capital), "top_n": int(top_n), "freq": freq,
        "risk_config": risk,
        "created_at": datetime.now().isoformat(),
        "last_processed_date": None, "last_rebalance_date": None,
    }
    state["accounts"][str(aid)] = acc
    _json_save(state)
    return acc


def list_accounts() -> list[dict]:
    _ensure_columns()
    if pg.configured():
        try:
            df = _q("SELECT * FROM paper_accounts ORDER BY id")
            return [_acc_row(r) for r in df.to_dict(orient="records")]
        except Exception as exc:
            print(f"[paper] SQLite 读取失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    return [_acc_row(a) for a in state["accounts"].values()]


def get_account(account_id: int) -> dict | None:
    if pg.configured():
        try:
            df = _q("SELECT * FROM paper_accounts WHERE id=%s", (int(account_id),))
            if df.empty:
                return None
            return _acc_row(df.iloc[0])
        except Exception as exc:
            print(f"[paper] SQLite 读取失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    return _acc_row(state["accounts"].get(str(account_id)))


def set_account_status(account_id: int, status: str) -> dict | None:
    if status not in ("active", "paused"):
        raise ValueError("status 仅支持 active/paused")
    if pg.configured():
        try:
            _ex("UPDATE paper_accounts SET status=%s WHERE id=%s",
                (status, int(account_id)))
            return get_account(account_id)
        except Exception as exc:
            print(f"[paper] SQLite 写入失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    acc = state["accounts"].get(str(account_id))
    if acc is None:
        return None
    acc["status"] = status
    _json_save(state)
    return _acc_row(acc)


def update_account_strategy(
    account_id: int,
    strategy_name: str | None = None,
    factor: str | None = None,
    ascending: bool | None = None,
    universe: str | None = None,
    top_n: int | None = None,
    freq: str | None = None,
    risk_config: dict | None = None,
) -> dict | None:
    """在线切换账户策略/参数（factor 账户）。事件账户不支持在线切换。

    只更新调用方显式传入的字段；返回更新后的账户，账户不存在返回 None。
    调用方负责在切换后 reset_account() 清空旧策略产生的历史。
    """
    if freq is not None and freq not in ("daily", "weekly", "monthly", "semiannual"):
        raise ValueError("freq 仅支持 daily/weekly/monthly/semiannual")
    if universe is not None:
        universe = normalize_universe(universe)
    acc = get_account(account_id)
    if acc is None:
        return None
    if acc.get("strategy_type") == "event":
        raise ValueError("事件策略账户不支持在线切换，请新建账户")
    fields = (
        ("strategy_name", strategy_name),
        ("factor", factor),
        ("ascending", ascending),
        ("universe", universe),
        ("top_n", top_n),
        ("freq", freq),
        ("risk_config", risk_config),
    )
    if pg.configured():
        try:
            sets = []
            params = []
            for col, val in fields:
                if val is None:
                    continue
                sets.append(f"{col}=%s")
                params.append(
                    bool(val) if col == "ascending"
                    else json.dumps(val, ensure_ascii=False)
                    if col == "risk_config"
                    else val)
            if sets:
                params.append(int(account_id))
                _ex(f"UPDATE paper_accounts SET {', '.join(sets)} WHERE id=%s",
                    tuple(params))
            return get_account(account_id)
        except Exception as exc:
            print(f"[paper] SQLite 更新失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    a = state["accounts"].get(str(account_id))
    if a is None:
        return None
    for col, val in fields:
        if val is not None:
            a[col] = val
    _json_save(state)
    return _acc_row(a)


def delete_account(account_id: int) -> bool:
    if pg.configured():
        try:
            _ex("DELETE FROM paper_events WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_equity_snapshots WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_positions WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_trades WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_orders WHERE account_id=%s", (int(account_id),))
            with pg.get_conn() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM paper_accounts WHERE id=%s", (int(account_id),))
                return cur.rowcount > 0
        except Exception as exc:
            print(f"[paper] SQLite 删除失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    key = str(account_id)
    if key not in state["accounts"]:
        return False
    del state["accounts"][key]
    state["orders"] = [o for o in state.get("orders", [])
                       if o.get("account_id") != account_id]
    state["trades"] = [t for t in state.get("trades", [])
                       if t.get("account_id") != account_id]
    state["events"] = [e for e in state.get("events", [])
                       if e.get("account_id") != account_id]
    _json_save(state)
    return True


def reset_account(account_id: int) -> bool:
    """清空订单/成交/持仓/估值/事件，保留账户并恢复初始资金。"""
    if pg.configured():
        try:
            _ex("DELETE FROM paper_events WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_equity_snapshots WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_positions WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_trades WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_orders WHERE account_id=%s", (int(account_id),))
            _ex("UPDATE paper_accounts SET last_processed_date=NULL,"
                " last_rebalance_date=NULL WHERE id=%s", (int(account_id),))
            return True
        except Exception as exc:
            print(f"[paper] SQLite 重置失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    key = str(account_id)
    if key not in state["accounts"]:
        return False
    state["orders"] = [o for o in state.get("orders", [])
                       if o.get("account_id") != account_id]
    state["trades"] = [t for t in state.get("trades", [])
                       if t.get("account_id") != account_id]
    state["events"] = [e for e in state.get("events", [])
                       if e.get("account_id") != account_id]
    state["accounts"][key]["last_processed_date"] = None
    state["accounts"][key]["last_rebalance_date"] = None
    _json_save(state)
    return True


def _clear_account_state(account_id: int) -> None:
    """清空订单/成交/持仓/估值（保留账户与事件日志），供事件策略重放。"""
    if pg.configured():
        try:
            _ex("DELETE FROM paper_equity_snapshots WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_positions WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_trades WHERE account_id=%s", (int(account_id),))
            _ex("DELETE FROM paper_orders WHERE account_id=%s", (int(account_id),))
            return
        except Exception as exc:
            print(f"[paper] SQLite 清空失败，回退 JSON: {exc}", flush=True)
    state = _json_state()
    state["orders"] = [o for o in state.get("orders", [])
                       if o.get("account_id") != account_id]
    state["trades"] = [t for t in state.get("trades", [])
                       if t.get("account_id") != account_id]
    state.setdefault("positions", {}).pop(str(account_id), None)
    state.setdefault("snapshots", {}).pop(str(account_id), None)
    _json_save(state)


def _update_account_start(account_id: int, start_date: str) -> None:
    if pg.configured():
        try:
            _ex("UPDATE paper_accounts SET start_date=%s WHERE id=%s",
                (start_date, int(account_id)))
            return
        except Exception as exc:
            print(f"[paper] SQLite 账户起始日更新失败: {exc}", flush=True)
    state = _json_state()
    acc = state["accounts"].get(str(account_id))
    if acc is not None:
        acc["start_date"] = start_date
        _json_save(state)
