from __future__ import annotations

from pathlib import Path

import pandas as pd


PANEL_PATH = Path("/tmp/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet")
UNIVERSE_PATH = Path("/tmp/universe_cs800.csv")
TECH_PATH = Path("/tmp/tech_universe_sw.csv")
INDEX_PATH = Path("/tmp/csi300_index.csv")


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"面板数据不存在: {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"股票池数据不存在: {UNIVERSE_PATH}")
    uni = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
    uni["code"] = uni["code"].astype(str).str.zfill(6)
    return uni


def load_tech() -> pd.DataFrame:
    if not TECH_PATH.exists():
        raise FileNotFoundError(f"行业数据不存在: {TECH_PATH}")
    tech = pd.read_csv(TECH_PATH, dtype={"code": str})
    tech["code"] = tech["code"].astype(str).str.zfill(6)
    return tech


def load_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"基准指数数据不存在: {INDEX_PATH}")
    idx = pd.read_csv(INDEX_PATH)
    idx["date"] = pd.to_datetime(idx["date"])
    return idx


def data_status() -> dict:
    files = {
        "panel": PANEL_PATH,
        "universe": UNIVERSE_PATH,
        "tech": TECH_PATH,
        "index": INDEX_PATH,
    }
    out = {}
    for key, path in files.items():
        if path.exists():
            out[key] = {"exists": True, "size_mb": round(path.stat().st_size / 1e6, 1)}
        else:
            out[key] = {"exists": False, "size_mb": None}
    return out
