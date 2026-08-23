from __future__ import annotations

import os
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data import (data_status, load_etf, load_etf_panel, load_fund,
                       load_fund_nav, load_fund_panel, load_index, load_panel,
                       load_tech, load_universe)
from core.run_log import finish_run, start_run
from core.updater import refresh_all
from core.store import (ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE,
                        FUND_PANEL_FILE, INDEX_FILE, PANEL_FILE, TECH_FILE,
                        UNIVERSE_FILE, normalize_universe)


DATA_CACHE: dict[tuple, dict] = {}
UPDATE_STATE: dict[str, Any] = {"running": False, "progress": 0.0,
                                "text": "", "result": None, "error": None}
CONFIG_UPDATE_STATES: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_config_update_lock = threading.Lock()
ROOT = Path(__file__).resolve().parents[1]
DATA_TASK_CATALOG = ROOT / "data_status" / "catalog.json"
DATA_UPDATE_LOG_DIR = ROOT / "logs" / "data_updates"
_PARAM_TYPES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}

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
    etf_panel = load_etf_panel(start=start, end=end) if need_heavy else pd.DataFrame(
        columns=["date", "open", "close", "turnover", "amount", "code",
                 "turn20", "am20", "volume"])
    fund_nav = load_fund_nav(start=start, end=end) if need_heavy else pd.DataFrame(
        columns=["date", "code", "nav"])
    fund_panel = load_fund_panel(start=start, end=end) if need_heavy else pd.DataFrame(
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
    elif universe == "场外基金":
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
    run_id = start_run("quant_ui:update", metadata={"mode": mode, "end": end})

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
                finish_run(run_id, status="failed",
                           error_message=f"insufficient memory: {avail:.0f}MB")
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
            finish_run(run_id, status="success",
                       rows_written=result.get("rows_written", 0) if isinstance(result, dict) else 0)
        except Exception as exc:
            UPDATE_STATE.update({"running": False, "error": str(exc)})
            finish_run(run_id, status="failed", error_message=str(exc))

    threading.Thread(target=worker, daemon=True).start()


def _set_progress(p: float, t: float, label: str) -> None:
    UPDATE_STATE.update({
        "progress": round(float(p) / max(float(t), 1.0), 4),
        "text": label,
    })


def configured_update_tasks() -> list[dict[str, Any]]:
    """Read the declarative data-update catalog shared with data_status."""
    data = json.loads(DATA_TASK_CATALOG.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("data task catalog is invalid")
    out: list[dict[str, Any]] = []
    for raw in tasks:
        if not isinstance(raw, dict) or not all(key in raw for key in ("id", "name", "script")):
            continue
        item = {
            "id": str(raw["id"]),
            "name": str(raw["name"]),
            "group": str(raw.get("group", "其他")),
            "script": str(raw["script"]),
            "base": list(raw.get("base", [])),
            "required": list(raw.get("required", [])),
            "params": dict(raw.get("params", {})),
        }
        item["state"] = dict(CONFIG_UPDATE_STATES.get(item["id"], {}))
        out.append(item)
    return out


def _configured_task_command(task_id: str, supplied: dict[str, Any]) -> list[str]:
    task = next((item for item in configured_update_tasks() if item["id"] == task_id), None)
    if task is None:
        raise ValueError(f"unknown update task: {task_id}")
    script = ROOT / "scripts" / task["script"]
    if not script.is_file():
        raise ValueError(f"configured script does not exist: {task['script']}")
    argv = [sys.executable, str(script), *task["base"]]
    for name in task["required"]:
        if supplied.get(name) in (None, ""):
            raise ValueError(f"missing required parameter: {name}")
    for name, type_name in task["params"].items():
        if name not in supplied or supplied[name] in (None, "", False):
            continue
        expected = _PARAM_TYPES.get(str(type_name))
        if expected is None:
            raise ValueError(f"unsupported parameter type: {type_name}")
        value = supplied[name]
        if expected is bool:
            if value is True:
                argv.append("--" + name.replace("_", "-"))
            continue
        if expected is int:
            value = int(value)
        elif expected is float:
            value = float(value)
        elif not isinstance(value, str):
            raise ValueError(f"parameter {name} must be a string")
        argv.extend(["--" + name.replace("_", "-"), str(value)])
    return argv


def run_configured_update_background(task_id: str, supplied: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one catalog task at a time and retain state for the Vue data page."""
    supplied = supplied or {}
    command = _configured_task_command(task_id, supplied)
    with _config_update_lock:
        if any(state.get("running") for state in CONFIG_UPDATE_STATES.values()):
            raise RuntimeError("another configured data update is already running")
        state = {
            "running": True,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "error": None,
            "log": None,
            "command": command,
        }
        CONFIG_UPDATE_STATES[task_id] = state

    run_id = start_run(f"quant_ui:{task_id}", metadata={"task_id": task_id, "supplied": supplied})

    def worker() -> None:
        DATA_UPDATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = DATA_UPDATE_LOG_DIR / f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + subprocess.list2cmdline(command) + "\n\n")
                log.flush()
                completed = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(f"exit code {completed.returncode}")
            invalidate_data()
            result = {"status": "success", "error": None}
            finish_run(run_id, status="success")
        except Exception as exc:  # noqa: BLE001
            result = {"status": "failed", "error": str(exc)}
            finish_run(run_id, status="failed", error_message=str(exc))
        with _config_update_lock:
            CONFIG_UPDATE_STATES[task_id].update({
                "running": False,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "log": str(log_path),
                **result,
            })

    threading.Thread(target=worker, daemon=True, name=f"data-update-{task_id}").start()
    return dict(state)
