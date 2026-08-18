from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import sqldb as pg
from .store import DATA_DIR


ACCOUNT_DIR = DATA_DIR / "account"
TRANSACTIONS_FILE = ACCOUNT_DIR / "transactions.csv"
DEPOSITS_FILE = ACCOUNT_DIR / "deposits.csv"

_migrated = False


def _atomic_write_csv(target: Path, df: pd.DataFrame) -> None:
    ACCOUNT_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(ACCOUNT_DIR), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_transactions_csv() -> pd.DataFrame:
    if not TRANSACTIONS_FILE.exists():
        return pd.DataFrame(columns=["date", "code", "name", "action",
                                     "shares", "price", "fee", "note"])
    df = pd.read_csv(TRANSACTIONS_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df.sort_values("date").reset_index(drop=True)


def _load_deposits_csv() -> pd.DataFrame:
    if not DEPOSITS_FILE.exists():
        return pd.DataFrame(columns=["date", "amount", "note"])
    df = pd.read_csv(DEPOSITS_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def migrate_legacy_csv() -> None:
    """一次性把旧 CSV 账本迁入 PG（PG 表为空且 CSV 有数据时执行）。"""
    global _migrated
    if _migrated or not pg.configured():
        return
    try:
        with pg.get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM ledger_transactions")
            tx_n = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM ledger_deposits")
            dep_n = int(cur.fetchone()[0])
            if tx_n == 0 and TRANSACTIONS_FILE.exists():
                df = _load_transactions_csv()
                if len(df):
                    for _, r in df.iterrows():
                        note = r.get("note", "")
                        if isinstance(note, float) and np.isnan(note):
                            note = ""
                        cur.execute(
                            "INSERT INTO ledger_transactions"
                            "(date, code, name, action, shares, price, fee, note)"
                            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                            (r["date"].date(), str(r["code"]).zfill(6), r.get("name", ""),
                             r["action"], float(r["shares"]), float(r["price"]),
                             float(r.get("fee", 0) or 0), note),
                        )
            if dep_n == 0 and DEPOSITS_FILE.exists():
                df = _load_deposits_csv()
                if len(df):
                    for _, r in df.iterrows():
                        note = r.get("note", "")
                        if isinstance(note, float) and np.isnan(note):
                            note = ""
                        cur.execute(
                            "INSERT INTO ledger_deposits (date, amount, note)"
                            " VALUES (%s,%s,%s)",
                            (r["date"].date(), float(r["amount"]), note),
                        )
        _migrated = True
    except Exception as exc:  # 迁移失败不阻断后续
        print(f"[ledger] CSV->PG 迁移失败: {exc}", flush=True)


def load_transactions() -> pd.DataFrame:
    migrate_legacy_csv()
    if pg.configured():
        try:
            df = pg.query_df(
                "SELECT date, code, name, action, shares, price, fee, note"
                " FROM ledger_transactions ORDER BY date, id")
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df["code"] = df["code"].astype(str).str.zfill(6)
                return df.reset_index(drop=True)
        except Exception as exc:
            print(f"[ledger] PG 读取失败，回退 CSV: {exc}", flush=True)
    return _load_transactions_csv()


def add_transaction(date, code, name, action, shares, price, fee=0.0, note="") -> None:
    migrate_legacy_csv()
    if pg.configured():
        try:
            with pg.get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ledger_transactions"
                    "(date, code, name, action, shares, price, fee, note)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (pd.Timestamp(date).date(), str(code).zfill(6), name, action,
                     float(shares), float(price), float(fee), note),
                )
            return
        except Exception as exc:
            print(f"[ledger] PG 写入失败，回退 CSV: {exc}", flush=True)
    df = _load_transactions_csv()
    row = pd.DataFrame([{
        "date": pd.Timestamp(date), "code": str(code).zfill(6), "name": name,
        "action": action, "shares": float(shares), "price": float(price),
        "fee": float(fee), "note": note,
    }])
    _atomic_write_csv(TRANSACTIONS_FILE, pd.concat([df, row], ignore_index=True))


def load_deposits() -> pd.DataFrame:
    migrate_legacy_csv()
    if pg.configured():
        try:
            df = pg.query_df(
                "SELECT date, amount, note FROM ledger_deposits ORDER BY date, id")
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df.reset_index(drop=True)
        except Exception as exc:
            print(f"[ledger] PG 读取失败，回退 CSV: {exc}", flush=True)
    return _load_deposits_csv()


def add_deposit(date, amount, note="") -> None:
    migrate_legacy_csv()
    if pg.configured():
        try:
            with pg.get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ledger_deposits (date, amount, note)"
                    " VALUES (%s,%s,%s)",
                    (pd.Timestamp(date).date(), float(amount), note),
                )
            return
        except Exception as exc:
            print(f"[ledger] PG 写入失败，回退 CSV: {exc}", flush=True)
    df = _load_deposits_csv()
    row = pd.DataFrame([{
        "date": pd.Timestamp(date), "amount": float(amount), "note": note,
    }])
    _atomic_write_csv(DEPOSITS_FILE, pd.concat([df, row], ignore_index=True))


def compute_equity(panel: pd.DataFrame,
                   transactions: pd.DataFrame | None = None,
                   deposits: pd.DataFrame | None = None) -> pd.DataFrame:
    """按交易流水逐日估值，返回 equity 明细。

    约定：交易按当日收盘价成交，用于估值和盈亏展示。
    """
    transactions = load_transactions() if transactions is None else transactions
    deposits = load_deposits() if deposits is None else deposits
    if transactions.empty and deposits.empty:
        return pd.DataFrame(columns=["date", "cash", "market_value", "equity", "pnl", "pnl_pct"])

    close = panel.pivot_table(index="date", columns="code", values="close",
                              aggfunc="last").sort_index()
    all_dates = close.index
    if transactions.empty:
        first = deposits["date"].min()
    else:
        first = min(deposits["date"].min() if not deposits.empty else all_dates[0],
                    transactions["date"].min())
    dates = all_dates[all_dates >= first]
    if len(dates) == 0:
        # 出入金日期晚于面板数据范围：返回完整空表，避免下游取列报错
        return pd.DataFrame(columns=["date", "cash", "market_value",
                                     "equity", "pnl", "pnl_pct"])
    close = close.reindex(dates).ffill()

    tx_by_date = {d: g for d, g in transactions.groupby("date")}
    dep_by_date = {d: g for d, g in deposits.groupby("date")}

    holdings: dict[str, float] = {}
    cash = 0.0
    rows = []
    for d in dates:
        dep = dep_by_date.get(d)
        if dep is not None:
            cash += float(dep["amount"].sum())
        tx = tx_by_date.get(d)
        if tx is not None:
            for _, t in tx.iterrows():
                code = str(t["code"]).zfill(6)
                price = float(t["price"])
                shares = float(t["shares"])
                fee = float(t["fee"])
                if t["action"] == "buy":
                    cash -= shares * price + fee
                    holdings[code] = holdings.get(code, 0.0) + shares
                else:
                    cash += shares * price - fee
                    holdings[code] = holdings.get(code, 0.0) - shares
                    if holdings[code] <= 1e-6:
                        holdings.pop(code, None)

        mv = 0.0
        for code, shares in holdings.items():
            px = close.loc[d, code] if code in close.columns else np.nan
            if not np.isnan(px):
                mv += shares * float(px)
        equity = cash + mv
        rows.append({"date": d, "cash": cash, "market_value": mv,
                     "equity": equity})

    out = pd.DataFrame(rows)
    out["pnl"] = out["equity"] - out["equity"].iloc[0]
    out["pnl_pct"] = out["pnl"] / out["equity"].iloc[0]
    return out


def current_positions(panel: pd.DataFrame,
                      transactions: pd.DataFrame | None = None) -> pd.DataFrame:
    transactions = load_transactions() if transactions is None else transactions
    if transactions.empty:
        return pd.DataFrame(columns=["code", "name", "shares", "avg_cost",
                                     "price", "market_value", "cost", "pnl", "pnl_pct"])
    close = panel.pivot_table(index="date", columns="code", values="close",
                              aggfunc="last").sort_index()
    last_date = close.index[-1]
    last_px = close.iloc[-1]

    holdings: dict[str, dict] = {}
    for _, t in transactions.iterrows():
        code = str(t["code"]).zfill(6)
        price = float(t["price"])
        shares = float(t["shares"])
        if t["action"] == "buy":
            h = holdings.setdefault(code, {"shares": 0.0, "cost": 0.0, "fee": 0.0})
            h["shares"] += shares
            h["cost"] += shares * price + float(t["fee"])
            h["fee"] += float(t["fee"])
        else:
            h = holdings.get(code)
            if h is None:
                continue
            # 按当日价格卖出，成本按比例减少
            ratio = shares / h["shares"] if h["shares"] else 0.0
            h["cost"] -= h["cost"] * ratio
            h["fee"] -= h["fee"] * ratio
            h["shares"] -= shares

    rows = []
    for code, h in holdings.items():
        if h["shares"] <= 1e-6:
            continue
        px = last_px.get(code, np.nan) if code in last_px.index else np.nan
        cost_total = h["cost"] - h["fee"]
        avg_cost = cost_total / h["shares"] if h["shares"] else np.nan
        mv = h["shares"] * px if not np.isnan(px) else np.nan
        pnl = mv - cost_total if not np.isnan(mv) else np.nan
        rows.append({
            "code": code, "name": "",
            "shares": h["shares"], "avg_cost": avg_cost, "price": px,
            "market_value": mv, "cost": cost_total, "pnl": pnl,
            "pnl_pct": pnl / cost_total if cost_total else np.nan,
        })

    names = {}
    if transactions is not None and len(transactions):
        names = dict(zip(transactions["code"], transactions["name"]))
    for r in rows:
        r["name"] = names.get(r["code"], "")
    return pd.DataFrame(rows)
