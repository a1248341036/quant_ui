from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from core.data import data_status, load_index, load_panel, load_tech, load_universe
from core.fetcher import update_data
from core.store import INDEX_FILE, PANEL_FILE, TECH_FILE, UNIVERSE_FILE


DATA_CACHE: dict[str, Any] = {}
UPDATE_STATE: dict[str, Any] = {"running": False, "progress": 0.0,
                                "text": "", "result": None, "error": None}
_lock = threading.Lock()


def _mtimes() -> tuple:
    return tuple(p.stat().st_mtime if p.exists() else None
                 for p in (PANEL_FILE, UNIVERSE_FILE, TECH_FILE, INDEX_FILE))


def load_data(force: bool = False) -> dict:
    m = _mtimes()
    if not force and DATA_CACHE.get("loaded") and DATA_CACHE.get("mtimes") == m:
        return DATA_CACHE
    panel = load_panel()
    uni = load_universe()
    tech = load_tech()
    index = load_index()
    DATA_CACHE.update({
        "loaded": True,
        "panel": panel,
        "universe": uni,
        "tech": tech,
        "index": index,
        "mtimes": m,
    })
    return DATA_CACHE


def invalidate_data() -> None:
    DATA_CACHE.clear()


def get_name_map() -> dict[str, str]:
    """股票代码 -> 名称（universe + tech 合并，前 6 位代码去重）。"""
    data = load_data()
    m = {}
    for df in (data["universe"], data["tech"]):
        if "code" in df and "name" in df:
            for code, name in zip(df["code"], df["name"]):
                code = str(code).zfill(6)
                if name and not pd.isna(name):
                    m.setdefault(code, str(name))
    return m


def get_industry_map() -> dict[str, str]:
    """股票代码 -> 申万行业（来自科技行业缓存表，科技池全覆盖）。"""
    data = load_data()
    tech = data["tech"]
    return {str(c).zfill(6): str(ind)
            for c, ind in zip(tech["code"], tech["industry"])}


def build_codes(universe: str, exclude_kechuang: bool) -> list[str]:
    data = load_data()
    panel, uni, tech = data["panel"], data["universe"], data["tech"]
    if universe == "科技行业":
        codes = set(tech["code"])
    else:
        codes = set(uni["code"])
    codes &= set(panel["code"].unique())
    if exclude_kechuang:
        codes = {c for c in codes if not c.startswith(("300", "301", "688", "689"))}
    return sorted(codes)


def _to_float(x) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def series_to_points(s: pd.Series) -> list[dict]:
    out = []
    for idx, v in s.items():
        v = _to_float(v)
        if v is None:
            continue
        out.append({"date": str(pd.Timestamp(idx).date()), "value": v})
    return out


def clean_records(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        item = {}
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                item[k] = None
            else:
                item[k] = v
        out.append(item)
    return out


def run_update_background(mode: str, end: str) -> None:
    def worker():
        with _lock:
            UPDATE_STATE.update({"running": True, "progress": 0.0, "text": "启动",
                                 "result": None, "error": None})
        try:
            result = update_data(
                mode=mode, end=end,
                progress=lambda p, t, label: _set_progress(p, t, label),
            )
            invalidate_data()
            UPDATE_STATE.update({"running": False, "progress": 1.0,
                                 "result": result, "error": None})
        except Exception as exc:
            UPDATE_STATE.update({"running": False, "error": str(exc)})

    threading.Thread(target=worker, daemon=True).start()


def _set_progress(p: float, t: float, label: str) -> None:
    UPDATE_STATE.update({
        "progress": round(float(p) / max(float(t), 1.0), 4),
        "text": label,
    })
