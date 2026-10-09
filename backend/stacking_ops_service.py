"""ML 组合运营总览：当前生效分数 / 最新名单 / 模拟盘对照 / 训练历史摘要。

只读聚合，无副作用；各分块独立降级（某块失败不影响其余返回）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .stacking_service import STACKING_ROOT, list_trainings

ROOT = Path(__file__).resolve().parents[1]


def _norm6(c: Any) -> str:
    import re

    m = re.search(r"(\d{6})", str(c))
    return (m.group(1) if m else str(c)).zfill(6)


def _score_source_run() -> dict[str, Any]:
    """判定 pred_demo 的来源训练：最新含 scores.parquet 的 stacking 目录。"""
    best: dict[str, Any] | None = None
    if STACKING_ROOT.is_dir():
        for d in STACKING_ROOT.iterdir():
            if not d.is_dir():
                continue
            sp = d / "scores.parquet"
            if sp.is_file():
                mt = sp.stat().st_mtime
                if best is None or mt > best["mtime"]:
                    best = {"train_id": d.name, "mtime": mt,
                            "source_rows": int(pd.read_parquet(sp, columns=["score"]).shape[0])}
    return best or {}


def _score_overview() -> dict[str, Any]:
    """生效分数概况：日期覆盖 / 股票数 / 低价过滤检测。"""
    from core.store import PRED_FILE

    out: dict[str, Any] = {"exists": False}
    if not PRED_FILE.is_file():
        return out
    df = pd.read_parquet(PRED_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].map(_norm6)
    out.update({
        "exists": True,
        "rows": int(len(df)),
        "date_min": df["date"].min().date().isoformat(),
        "date_max": df["date"].max().date().isoformat(),
        "n_codes": int(df["code"].nunique()),
        "days": int(df["date"].nunique()),
    })
    src = _score_source_run()
    if src:
        out["source_train_id"] = src["train_id"]
        out["source_rows"] = src["source_rows"]
        out["rows_ratio"] = round(len(df) / src["source_rows"], 3) if src["source_rows"] else None
    # 低价过滤检测：抽样最新日 join panel close，看是否存在 >10 元
    try:
        from alphaagent.data.adapters.cnequity import load_panel_from_cne

        last = df["date"].max()
        p = load_panel_from_cne(start=str(last.date()), end=str(last.date()),
                                include_fundamentals=False)
        pday = p.xs(last, level="datetime")
        prices = pd.Series(pday["close"].to_numpy(float),
                           index=[_norm6(c) for c in pday.index])
        last_day = df[df["date"] == last].drop_duplicates("code")
        merged = last_day.merge(prices.rename("close"), left_on="code", right_index=True,
                                how="left")
        out["filter_low_price"] = bool((merged["close"].dropna() <= 10.0).all())
        out["latest_day_codes"] = int(merged["code"].nunique())
    except Exception as exc:  # noqa: BLE001
        out["filter_low_price"] = None
        out["filter_check_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _latest_watchlist(top_n: int = 10, max_price: float = 10.0) -> dict[str, Any]:
    """最新分数日的主板 top N 名单（可执行视角：价格 / 一手金额）。"""
    from core.store import PRED_FILE

    out: dict[str, Any] = {"exists": False}
    if not PRED_FILE.is_file():
        return out
    df = pd.read_parquet(PRED_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].map(_norm6)
    last = df["date"].max()
    sub = df[df["date"] == last].copy()
    sub = sub[sub["code"].str.startswith("6")
              | sub["code"].str[:3].isin(("000", "001", "002", "003"))]
    sub = sub.sort_values("score", ascending=False)
    top = sub.head(top_n).copy()
    try:
        from alphaagent.data.adapters.cnequity import load_panel_from_cne

        p = load_panel_from_cne(start=str(last.date()), end=str(last.date()),
                                include_fundamentals=False)
        pday = p.xs(last, level="datetime")
        prices = pd.Series(pday["close"].to_numpy(float),
                           index=[_norm6(c) for c in pday.index])
        prices = prices[~prices.index.duplicated()]
        top = top.merge(prices.rename("close"), left_on="code", right_index=True, how="left")
    except Exception:
        top["close"] = None
    top["lot_amount"] = (top["close"] * 100).round(0)
    top["affordable_700"] = top["close"].apply(
        lambda c: bool(pd.notna(c) and c * 100 <= 700)
    )
    rows = [{
        "rank": i + 1,
        "code": str(r["code"]),
        "score": round(float(r["score"]), 5),
        "close": None if pd.isna(r.get("close")) else round(float(r["close"]), 3),
        "lot_amount": None if pd.isna(r.get("lot_amount")) else float(r["lot_amount"]),
        "affordable_700": bool(r["affordable_700"]),
    } for i, (_, r) in enumerate(top.iterrows())]
    out.update({
        "exists": True,
        "score_date": last.date().isoformat(),
        "pool_size": int(len(sub)),
        "items": rows,
        "total_lot_amount": float(top["lot_amount"].fillna(0).sum()),
    })
    return out


def _paper_accounts() -> list[dict[str, Any]]:
    """pred 因子驱动的模拟盘账户对照（轻量：不 enrich 持仓现价）。"""
    try:
        from core.paper import account_equity, list_accounts
        from core.paper.details import _load_positions
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    out: list[dict[str, Any]] = []
    for a in list_accounts():
        if a.get("factor") != "pred" or a.get("universe") == "ETF":
            continue
        item: dict[str, Any] = {
            "id": a.get("id"),
            "name": a.get("name"),
            "universe": a.get("universe"),
            "freq": a.get("freq"),
            "top_n": a.get("top_n"),
            "capital": a.get("capital"),
            "status": a.get("status"),
            "start_date": a.get("start_date"),
        }
        try:
            eq = account_equity(a["id"])
            if eq:
                last_eq = eq[-1]
                item["equity"] = round(float(last_eq.get("equity") or 0), 2)
                item["pnl"] = round(float(last_eq.get("pnl") or 0), 2)
                item["pnl_pct"] = round(float(last_eq.get("pnl_pct") or 0) * 100, 2)
                item["last_date"] = last_eq.get("date")
        except Exception as exc:  # noqa: BLE001
            item["equity_error"] = f"{type(exc).__name__}: {exc}"
        try:
            pos = _load_positions(a["id"])
            item["n_positions"] = len(pos)
        except Exception:
            item["n_positions"] = None
        out.append(item)
    return out


def _weekly_rhythm() -> dict[str, Any]:
    today = pd.Timestamp.today().normalize()
    # 0 = 今天（若今天是周五/周一，即今天就是更新/调仓日）
    days_to_fri = (4 - today.weekday()) % 7
    days_to_mon = (0 - today.weekday()) % 7
    return {
        "today": today.date().isoformat(),
        "next_score_update": (today + pd.Timedelta(days=days_to_fri)).date().isoformat(),
        "next_rebalance": (today + pd.Timedelta(days=days_to_mon)).date().isoformat(),
        "note": "每周五收盘后更新分数并出名单；下周一开盘按名单每只一手执行",
    }


def get_ops_overview() -> dict[str, Any]:
    """聚合运营总览（各块独立降级）。"""
    blocks: dict[str, Any] = {}
    for key, fn in (
        ("score", _score_overview),
        ("watchlist", _latest_watchlist),
        ("paper_accounts", _paper_accounts),
        ("weekly_rhythm", _weekly_rhythm),
    ):
        try:
            blocks[key] = fn()
        except Exception as exc:  # noqa: BLE001
            blocks[key] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        blocks["trainings"] = list_trainings(limit=10)
    except Exception as exc:  # noqa: BLE001
        blocks["trainings"] = [{"error": f"{type(exc).__name__}: {exc}"}]
    return blocks
