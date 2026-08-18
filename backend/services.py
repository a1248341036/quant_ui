from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from core.data import (data_status, load_etf, load_etf_panel, load_fund,
                       load_fund_nav, load_fund_panel, load_index, load_panel,
                       load_tech, load_universe)
from core.updater import refresh_all
from core.store import (ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE,
                        FUND_PANEL_FILE, INDEX_FILE, PANEL_FILE, TECH_FILE,
                        UNIVERSE_FILE, normalize_universe)


DATA_CACHE: dict[tuple, dict] = {}
UPDATE_STATE: dict[str, Any] = {"running": False, "progress": 0.0,
                                "text": "", "result": None, "error": None}
_lock = threading.Lock()

# 数据集缓存空闲 TTL：30 分钟没人访问就释放整份缓存，
# 避免“什么都没跑”时仍常驻几百 MB ~ 1G 的全量行情数据。
CACHE_TTL = float(os.getenv("QUANT_CACHE_TTL", "1800"))

# Vue「开始更新」路径的内存保护阈值（单位 MB）。
# 启动前要求有足够空闲内存；运行中每个进度回调都检查，低于下限立即中止。
MIN_START_MEMORY_MB = 700
MIN_RUN_MEMORY_MB = 500
UPDATE_MAX_WORKERS = 3


def _available_memory_mb() -> float:
    """返回当前可用内存（MB），取系统可用与 cgroup 可用两者中的较小值。"""
    sys_avail = None
    try:
        import psutil
        sys_avail = float(psutil.virtual_memory().available) / 1024 / 1024
    except Exception:
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        sys_avail = float(line.split()[1]) / 1024
                        break
        except Exception:
            pass
    cg_avail = None
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as f:
            max_mem = f.read().strip()
        if max_mem != "max":
            with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as f:
                cur = float(f.read().strip())
            cg_avail = (float(max_mem) - cur) / 1024 / 1024
    except Exception:
        pass
    candidates = [x for x in (sys_avail, cg_avail) if x is not None]
    return min(candidates) if candidates else float("inf")


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
    if (not force and cached is not None and cached.get("loaded")
            and cached.get("mtimes") == m
            and time.time() - cached.get("ts", 0) < CACHE_TTL):
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
        "ts": time.time(),
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
    # 只保留最近一份数据集：不同回测区间/股票池会各自缓存一份全量数据，
    # 无界缓存会在请求结束后仍占几份几百 MB 的内存（小内存机器直接卡死）。
    if key not in DATA_CACHE:
        DATA_CACHE.clear()
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
        from core.data import load_etf_panel_codes
        etf_panel_codes = load_etf_panel_codes()
        if etf is None or len(etf) == 0 or not etf_panel_codes:
            return []
        codes = set(etf["code"]) & etf_panel_codes
        return sorted(codes)  # ETF 无科创/主板之分，跳过剔除
    elif universe == "场外科技基金":
        fund = load_fund()
        from core.data import load_fund_nav_codes
        fund_nav_codes = load_fund_nav_codes()
        if fund is None or len(fund) == 0 or not fund_nav_codes:
            return []
        codes = set(fund["code"]) & fund_nav_codes
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
    if x is None:
        return None
    try:
        f = float(x)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
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
        try:
            avail = _available_memory_mb()
            if avail < MIN_START_MEMORY_MB:
                with _lock:
                    UPDATE_STATE.update({
                        "running": False, "progress": 0.0, "text": "",
                        "result": None,
                        "error": f"可用内存不足（{avail:.0f}MB < {MIN_START_MEMORY_MB}MB），已拒绝启动更新",
                    })
                return
            with _lock:
                UPDATE_STATE.update({"running": True, "progress": 0.0, "text": "启动",
                                     "result": None, "error": None})
            def guarded_progress(p: float, t: float, label: str) -> None:
                avail = _available_memory_mb()
                if avail < MIN_RUN_MEMORY_MB:
                    raise MemoryError(
                        f"运行中可用内存不足（{avail:.0f}MB < {MIN_RUN_MEMORY_MB}MB），已中止更新")
                _set_progress(p, t, label)
            result = refresh_all(
                mode=mode, end=end, max_workers=UPDATE_MAX_WORKERS,
                progress=guarded_progress,
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
