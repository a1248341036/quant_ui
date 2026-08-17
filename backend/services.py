from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from core.data import (data_status, load_etf, load_etf_panel, load_fund,
                       load_fund_nav, load_fund_panel, load_index, load_panel,
                       load_tech, load_universe)
from core.fetcher import update_data
from core.store import (ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE,
                        FUND_PANEL_FILE, INDEX_FILE, PANEL_FILE, TECH_FILE,
                        UNIVERSE_FILE, normalize_universe)


DATA_CACHE: dict[tuple, dict] = {}
UPDATE_STATE: dict[str, Any] = {"running": False, "progress": 0.0,
                                "text": "", "result": None, "error": None}
_lock = threading.Lock()


def _mtimes() -> tuple:
    return tuple(p.stat().st_mtime if p.exists() else None
                 for p in (PANEL_FILE, UNIVERSE_FILE, TECH_FILE, INDEX_FILE,
                           ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE, FUND_PANEL_FILE))


def load_data(force: bool = False, start: str | None = None,
              end: str | None = None, need_panel: bool = True,
              codes: list[str] | None = None,
              need_heavy: bool = True) -> dict:
    """加载数据集。start/end 限定股票面板区间（回测只拉所需区间，避免全量内存）。

    need_panel=False 时不加载股票面板，供股票池/名称映射等轻量接口使用。
    codes 限定面板股票池，与 start/end 一起参与缓存键。
    need_heavy=False 时不加载 ETF/基金大 parquet（etf_panel/fund_nav/fund_panel）。
    """
    key = (start, end, need_panel, tuple(sorted(codes)) if codes else None, need_heavy)
    m = _mtimes()
    cached = DATA_CACHE.get(key)
    if not force and cached is not None and cached.get("loaded") and cached.get("mtimes") == m:
        return cached
    panel = load_panel(start=start, end=end, codes=codes) if need_panel else None
    uni = load_universe()
    tech = load_tech()
    index = load_index()
    etf = load_etf()
    fund = load_fund()
    etf_panel = load_etf_panel() if need_heavy else pd.DataFrame(
        columns=["date", "open", "close", "turnover", "amount", "code",
                 "turn20", "am20", "volume"])
    fund_nav = load_fund_nav() if need_heavy else pd.DataFrame(
        columns=["date", "code", "nav"])
    fund_panel = load_fund_panel() if need_heavy else pd.DataFrame(
        columns=["date", "open", "close", "turnover", "amount", "code",
                 "turn20", "am20", "volume"])
    out = {
        "loaded": True,
        "panel": panel,
        "universe": uni,
        "tech": tech,
        "index": index,
        "etf": etf,
        "etf_panel": etf_panel,
        "fund": fund,
        "fund_nav": fund_nav,
        "fund_panel": fund_panel,
        "mtimes": m,
    }
    DATA_CACHE[key] = out
    return out


def invalidate_data() -> None:
    DATA_CACHE.clear()
    try:
        from core.data import reset_caches
        reset_caches()
    except Exception:
        pass


def get_name_map() -> dict[str, str]:
    """股票代码 -> 名称（universe + tech 合并，前 6 位代码去重）。"""
    data = load_data(need_panel=False, need_heavy=False)
    m = {}
    for df in (data["universe"], data["tech"], data["etf"], data["fund"]):
        if "code" in df and "name" in df:
            for code, name in zip(df["code"], df["name"]):
                code = str(code).zfill(6)
                if name and not pd.isna(name):
                    m.setdefault(code, str(name))
    return m


def get_fund_name_map() -> dict[str, str]:
    """基金代码 -> 基金简称（场外基金池）。"""
    data = load_data(need_panel=False, need_heavy=False)
    fund = data.get("fund")
    if fund is None or len(fund) == 0 or "code" not in fund:
        return {}
    return {str(c).zfill(6): str(n)
            for c, n in zip(fund["code"], fund["name"])
            if n and not pd.isna(n)}


def get_industry_map() -> dict[str, str]:
    """股票代码 -> 申万行业（来自科技TMT缓存表，科技池全覆盖）。"""
    data = load_data(need_panel=False, need_heavy=False)
    tech = data["tech"]
    return {str(c).zfill(6): str(ind)
            for c, ind in zip(tech["code"], tech["industry"])}


def build_codes(universe: str, exclude_kechuang: bool,
                panel: pd.DataFrame | None = None) -> list[str]:
    universe = normalize_universe(universe)
    if universe == "科技TMT":
        codes = set(load_tech()["code"])
    elif universe == "ETF":
        etf = load_etf()
        etf_panel = load_etf_panel()
        if etf is None or len(etf) == 0 or etf_panel is None or len(etf_panel) == 0:
            return []
        codes = set(etf["code"]) & set(etf_panel["code"].unique())
        return sorted(codes)  # ETF 无科创/主板之分，跳过剔除
    elif universe == "场外科技基金":
        fund = load_fund()
        fund_nav = load_fund_nav()
        if fund is None or len(fund) == 0 or fund_nav is None or len(fund_nav) == 0:
            return []
        codes = set(fund["code"]) & set(fund_nav["code"].unique())
        return sorted(codes)
    else:
        codes = set(load_universe()["code"])
    if panel is not None:
        panel_codes = set(panel["code"].unique())
    else:
        from core.data import load_panel_codes
        panel_codes = load_panel_codes()
    codes &= panel_codes
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
