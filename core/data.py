from __future__ import annotations

from pathlib import Path

import pandas as pd

from .store import (INDEX_FILE, PANEL_FILE, TECH_FILE, UNIVERSE_FILE,
                    load_meta)


PANEL_PATH = Path("/tmp/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet")
UNIVERSE_PATH = Path("/tmp/universe_cs800.csv")
TECH_PATH = Path("/tmp/tech_universe_sw.csv")
INDEX_PATH = Path("/tmp/csi300_index.csv")


def load_panel() -> pd.DataFrame:
    if PANEL_FILE.exists():
        panel = pd.read_parquet(PANEL_FILE)
    elif PANEL_PATH.exists():
        panel = pd.read_parquet(PANEL_PATH)
    else:
        raise FileNotFoundError(f"面板数据不存在: {PANEL_PATH} 或 {PANEL_FILE}")
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def load_universe() -> pd.DataFrame:
    if UNIVERSE_FILE.exists():
        uni = pd.read_csv(UNIVERSE_FILE, dtype={"code": str})
    elif UNIVERSE_PATH.exists():
        uni = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
    else:
        raise FileNotFoundError(f"股票池数据不存在: {UNIVERSE_PATH}")
    uni["code"] = uni["code"].astype(str).str.zfill(6)
    return uni


def load_tech() -> pd.DataFrame:
    if TECH_FILE.exists():
        tech = pd.read_csv(TECH_FILE, dtype={"code": str})
    elif TECH_PATH.exists():
        tech = pd.read_csv(TECH_PATH, dtype={"code": str})
    else:
        raise FileNotFoundError(f"行业数据不存在: {TECH_PATH}")
    tech["code"] = tech["code"].astype(str).str.zfill(6)
    return tech


def load_index() -> pd.DataFrame:
    if INDEX_FILE.exists():
        idx = pd.read_csv(INDEX_FILE)
    elif INDEX_PATH.exists():
        idx = pd.read_csv(INDEX_PATH)
    else:
        raise FileNotFoundError(f"基准指数数据不存在: {INDEX_PATH}")
    idx["date"] = pd.to_datetime(idx["date"])
    return idx


def data_status() -> dict:
    files = {
        "panel": {"store": PANEL_FILE, "legacy": PANEL_PATH},
        "universe": {"store": UNIVERSE_FILE, "legacy": UNIVERSE_PATH},
        "tech": {"store": TECH_FILE, "legacy": TECH_PATH},
        "index": {"store": INDEX_FILE, "legacy": INDEX_PATH},
    }
    out = {}
    for key, paths in files.items():
        entry = {}
        for label, path in paths.items():
            entry[label] = {
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else None,
            }
        out[key] = entry
    out["meta"] = load_meta()
    return out
