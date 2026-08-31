"""资产池加载：股票池、行业分类、ETF、场外基金、指数。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from ..store import (
    ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_FEE_FILE,
    FUND_NAV_FILE, FUND_PANEL_FILE, INDEX_FILE, LEGACY_DATA_DIR,
    TECH_FILE, UNIVERSE_FILE,
)

from .panel import _USE_CNE

UNIVERSE_PATH = LEGACY_DATA_DIR / "panel/universe_cs800.csv"
TECH_PATH = LEGACY_DATA_DIR / "panel/tech_universe_sw.csv"
INDEX_PATH = LEGACY_DATA_DIR / "panel/csi300_index.csv"


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


def load_etf() -> pd.DataFrame:
    if not ETF_FILE.exists():
        return pd.DataFrame(columns=["code", "name"])
    etf = pd.read_csv(ETF_FILE, dtype={"code": str})
    etf["code"] = etf["code"].astype(str).str.zfill(6)
    return etf


def load_etf_panel(start: str | None = None,
                   end: str | None = None) -> pd.DataFrame:
    """ETF 日线面板；start/end 可选，pyarrow 下推区间过滤避免整面板进内存。"""
    if not ETF_PANEL_FILE.exists():
        return pd.DataFrame(columns=["date", "open", "close", "turnover",
                                     "amount", "code", "turn20", "am20",
                                     "volume"])
    filters: list = []
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))
    try:
        panel = pd.read_parquet(ETF_PANEL_FILE, filters=filters or None)
    except Exception:
        panel = pd.read_parquet(ETF_PANEL_FILE)
        if start:
            panel = panel[panel["date"] >= pd.Timestamp(start)]
        if end:
            panel = panel[panel["date"] <= pd.Timestamp(end)]
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    # ETF 日线固定为腾讯 qfq 前复权口径；不允许把 raw/hfq 数据静默混入回测。
    if "price_basis" not in panel.columns:
        panel["price_basis"] = "qfq"
    if not panel["price_basis"].fillna("qfq").eq("qfq").all():
        raise ValueError("ETF 面板必须使用 qfq 前复权口径")
    for c in ("turn20", "am20", "volume"):
        if c in panel.columns and panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel["code"] = panel["code"].astype("category")
    from ..assets import ETF_PROFILE, validate_ohlcv_panel
    validate_ohlcv_panel(panel, ETF_PROFILE)
    return panel


def load_etf_panel_codes() -> set[str]:
    """轻量返回 ETF 面板全部代码（只投影 code 列）。"""
    if not ETF_PANEL_FILE.exists():
        return set()
    codes = pd.read_parquet(ETF_PANEL_FILE, columns=["code"])["code"]
    return {str(c).zfill(6) for c in codes}


def load_fund() -> pd.DataFrame:
    if not FUND_FILE.exists():
        return pd.DataFrame(columns=["code", "name", "type"])
    fund = pd.read_csv(FUND_FILE, dtype={"code": str})
    fund["code"] = fund["code"].astype(str).str.zfill(6)
    return fund


def load_fund_nav(start: str | None = None,
                  end: str | None = None) -> pd.DataFrame:
    """基金净值；start/end 可选，pyarrow 下推区间过滤。"""
    if not FUND_NAV_FILE.exists():
        return pd.DataFrame(columns=["date", "code", "nav"])
    filters: list = []
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))
    try:
        panel = pd.read_parquet(FUND_NAV_FILE, filters=filters or None)
    except Exception:
        panel = pd.read_parquet(FUND_NAV_FILE)
        if start:
            panel = panel[panel["date"] >= pd.Timestamp(start)]
        if end:
            panel = panel[panel["date"] <= pd.Timestamp(end)]
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype("category")
    return panel


def load_fund_nav_codes() -> set[str]:
    """轻量返回基金净值面板全部代码（只投影 code 列）。"""
    if not FUND_NAV_FILE.exists():
        return set()
    codes = pd.read_parquet(FUND_NAV_FILE, columns=["code"])["code"]
    return {str(c).zfill(6) for c in codes}


def load_fund_panel(start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
    """基金净值标准面板（列结构与股票/ETF 面板一致，供统一回测）。"""
    if FUND_PANEL_FILE.exists():
        filters: list = []
        if start:
            filters.append(("date", ">=", pd.Timestamp(start)))
        if end:
            filters.append(("date", "<=", pd.Timestamp(end)))
        try:
            panel = pd.read_parquet(FUND_PANEL_FILE, filters=filters or None)
        except Exception:
            panel = pd.read_parquet(FUND_PANEL_FILE)
            if start:
                panel = panel[panel["date"] >= pd.Timestamp(start)]
            if end:
                panel = panel[panel["date"] <= pd.Timestamp(end)]
    else:
        try:
            from ..fund_engine import build_fund_panel
            panel = build_fund_panel(load_fund_nav())
        except Exception:
            return pd.DataFrame(columns=["date", "open", "close", "turnover",
                                         "amount", "code", "turn20", "am20",
                                         "volume"])
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    for c in ("turn20", "am20", "volume"):
        if c in panel.columns and panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel["code"] = panel["code"].astype("category")
    return panel


def load_index() -> pd.DataFrame:
    if _USE_CNE:
        try:
            from ..cne_reader import load_index as _cne_load_index
            return _cne_load_index()
        except Exception as exc:
            print(f"[cne] CNE 指数加载失败，回退 CSV: {exc}", file=sys.stderr)
    if INDEX_FILE.exists():
        idx = pd.read_parquet(INDEX_FILE)
    elif INDEX_PATH.exists():
        idx = pd.read_csv(INDEX_PATH)
    else:
        raise FileNotFoundError(f"基准指数数据不存在: {INDEX_PATH}")
    if "code" not in idx.columns:
        # 旧版单指数文件（沪深300）
        idx = idx.copy()
        idx["code"] = "sh000300"
        idx["name"] = "沪深300"
    idx["code"] = idx["code"].astype(str)
    idx["name"] = idx["name"].astype(str)
    idx["date"] = pd.to_datetime(idx["date"])
    cols = ["date", "code", "name"]
    for c in ("open", "close"):
        if c in idx.columns:
            cols.append(c)
    return idx[cols].sort_values(["code", "date"]).reset_index(drop=True)
