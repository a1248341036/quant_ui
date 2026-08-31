"""底层读写：SQLite 优先 / JSON 回退。"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import sqldb as pg
from ..store import DATA_DIR, normalize_universe
from .. import trading_config


PAPER_DIR = DATA_DIR / "paper"
PAPER_FILE = PAPER_DIR / "paper_store.json"

DEFAULT_RISK = {
    "buy_cost": trading_config.BUY_COST,
    "sell_cost": trading_config.SELL_COST,
    "spread_bps": 0.0,
    "min_commission": 0.0,
    "lot_size": trading_config.LOT_SIZE,
    "amount_q": trading_config.AMOUNT_Q,
    "max_weight": 0.5,
}


def _q(sql: str, params: tuple | None = None) -> pd.DataFrame:
    return pg.query_df(sql, params)


def _ex(sql: str, params: tuple | None = None) -> None:
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)


def _ex_id(sql: str, params: tuple | None = None) -> int:
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0])


def _json_state() -> dict:
    if not PAPER_FILE.exists():
        return {"accounts": {}, "orders": [], "trades": [], "events": []}
    try:
        return json.loads(PAPER_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"accounts": {}, "orders": [], "trades": [], "events": []}


def _json_save(state: dict) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(PAPER_DIR), prefix=".paper.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, PAPER_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _json_next_id(state: dict, key: str) -> int:
    items = state.get(key) or []
    ids = [int(x.get("id", 0)) for x in items]
    return (max(ids, default=0) + 1)


def _acc_row(r: Any) -> dict:
    d = dict(r)
    for k, v in list(d.items()):
        if isinstance(v, np.integer):
            d[k] = int(v)
        elif isinstance(v, np.floating):
            d[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            d[k] = bool(v)
        elif isinstance(v, (pd.Timestamp, datetime)):
            d[k] = v.isoformat()
        elif hasattr(v, "isoformat") and not isinstance(v, str):
            d[k] = v.isoformat()
    if isinstance(d.get("risk_config"), str):
        try:
            d["risk_config"] = json.loads(d["risk_config"])
        except (json.JSONDecodeError, TypeError):
            d["risk_config"] = {}
    if "universe" in d:
        d["universe"] = normalize_universe(d["universe"])
    return d


def _jsonable(obj: Any) -> Any:
    """把 numpy/日期等类型转成可 JSON 序列化的原生类型。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "isoformat") and not isinstance(obj, str):
        return obj.isoformat()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _load_module(path: str):
    """加载用户代码模块（与代码实验室一致，支持 EVENT_STRATEGIES）。"""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("paper_strategy", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载代码模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_cols_checked = False


def _ensure_columns() -> None:
    """SQLite 迁移：老表补策略/订单字段列。"""
    global _cols_checked
    if _cols_checked or not pg.configured():
        return
    try:
        for col, ddl in (
            ("strategy_type", "VARCHAR(16) NOT NULL DEFAULT 'factor'"),
            ("module", "TEXT"),
            ("event_strategy", "VARCHAR(64)"),
            ("start_date", "DATE"),
        ):
            _ex(f"ALTER TABLE paper_accounts ADD COLUMN IF NOT EXISTS {col} {ddl}")
        _ex("ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS shares DOUBLE PRECISION")
        _cols_checked = True
    except Exception as exc:
        print(f"[paper] 列迁移失败: {exc}", flush=True)
