from __future__ import annotations

"""财务因子：Tushare parquet 财务宽表 -> point-in-time 因子矩阵（date x code）。

财务数据按公告日（ann_date）对齐，避免未来函数：每个交易日只使用
ann_date <= 当日的最近一期财报。

表缺失时返回空 dict，引擎自动降级为纯日线因子；
全市场补全用 `python scripts/sync_tushare_to_parquet.py --fina`。
"""


import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .store import PG_PARQUET_DIR

FINANCIAL_FACTORS = {
    "pb": "市净率(收盘/BPS，低估值)",
    "ep": "盈利收益率(EPS/收盘，高=便宜)",
    "roe": "ROE 加权",
    "gross_margin": "毛利率",
    "rev_yoy": "营收同比",
    "np_yoy": "净利同比",
}

_lock = threading.Lock()
_CACHE: dict = {}
_pg_parquet = __import__("os").getenv("QUANT_DATA_SOURCE", "cne").strip().lower() in (
    "cne", "pg", "pg_parquet"
)


def _map_code_to_ts() -> dict[str, str]:
    with _lock:
        if "code_map" in _CACHE:
            return _CACHE["code_map"]
        if _pg_parquet:
            try:
                path = PG_PARQUET_DIR / "stock_basic.parquet"
                if path.exists():
                    df = pd.read_parquet(path, columns=["ts_code", "symbol"])
                    _CACHE["code_map"] = {
                        str(r["symbol"]).zfill(6): str(r["ts_code"])
                        for _, r in df.iterrows()
                    }
                    return _CACHE["code_map"]
            except Exception:
                pass
        _CACHE["code_map"] = {}
        return _CACHE["code_map"]


def _load_raw() -> dict[str, pd.DataFrame]:
    with _lock:
        if "raw" in _CACHE:
            return _CACHE["raw"]
        if _pg_parquet:
            out: dict[str, pd.DataFrame] = {}
            try:
                fina_path = PG_PARQUET_DIR / "fina_indicator.parquet"
                inc_path = PG_PARQUET_DIR / "income.parquet"
                if fina_path.exists() and inc_path.exists():
                    fina = pd.read_parquet(fina_path, columns=[
                        "ts_code", "ann_date", "end_date", "bps", "eps",
                        "roe_waa", "grossprofit_margin"])
                    inc = pd.read_parquet(inc_path, columns=[
                        "ts_code", "ann_date", "end_date", "revenue",
                        "n_income_attr_p"])
                    fina["ann_date"] = pd.to_datetime(fina["ann_date"])
                    fina["end_date"] = pd.to_datetime(fina["end_date"])
                    inc["ann_date"] = pd.to_datetime(inc["ann_date"])
                    inc["end_date"] = pd.to_datetime(inc["end_date"])
                    out["fina"] = fina
                    out["inc"] = inc
                    _CACHE["raw"] = out
                    return out
            except Exception:
                out = {}
        _CACHE["raw"] = {}
        return _CACHE["raw"]


def _pt_frame(events: pd.DataFrame, cal: pd.DatetimeIndex,
              codes: list[str], code_map: dict[str, str]) -> pd.DataFrame:
    """events: {ann_date, code, value} -> date x code 矩阵（公告日对齐+后向填充）。"""
    if events.empty:
        return pd.DataFrame(index=cal, columns=codes, dtype=float)
    wide = events.pivot_table(index="ann_date", columns="code",
                              values="value", aggfunc="last")
    wide = wide.reindex(index=cal.union(wide.index).sort_values())
    wide = wide.ffill().reindex(cal)
    return wide.reindex(columns=codes)


def _growth_frame(inc: pd.DataFrame, cal: pd.DatetimeIndex,
                  codes: list[str], code_map: dict[str, str],
                  col: str) -> pd.DataFrame:
    """同比增速：当期 vs 去年同季度（按公告日对齐）。"""
    if inc.empty or col not in inc.columns:
        return pd.DataFrame(index=cal, columns=codes, dtype=float)
    d = inc[["ts_code", "ann_date", "end_date", col]].dropna(subset=[col]).copy()
    d["code"] = d["ts_code"].map({v: k for k, v in code_map.items()})
    d = d.dropna(subset=["code"])
    d["year"] = d["end_date"].dt.year
    d["qtr"] = d["end_date"].dt.quarter
    prev = d.rename(columns={"year": "prev_year", col: "prev"})
    m = d.merge(prev, on=["ts_code", "qtr"], suffixes=("", "_prev"))
    m = m[m["year"] == m["prev_year"] + 1].copy()
    m["value"] = m[col] / m["prev"] - 1.0
    m = m[m["value"].abs() < 10]  # 去异常增速
    return _pt_frame(m[["ann_date", "code", "value"]], cal, codes, code_map)


def financial_factor_frames(
    codes: list[str],
    cal: pd.DatetimeIndex,
    close: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """返回 {因子名: date x code 矩阵}，与 cal 对齐。表缺失时返回空 dict。"""
    code_map = _map_code_to_ts()
    raw = _load_raw()
    if not raw:
        return {}
    fina = raw.get("fina")
    inc = raw.get("inc")
    if fina is None or inc is None:
        return {}

    inv = {v: k for k, v in code_map.items()}
    fina = fina.copy()
    fina["code"] = fina["ts_code"].map(inv)
    fina = fina.dropna(subset=["code"])
    fina = fina[fina["code"].isin(codes)]

    frames: dict[str, pd.DataFrame] = {}
    for col, name in (("roe_waa", "roe"), ("grossprofit_margin", "gross_margin"),
                      ("bps", "bps"), ("eps", "eps")):
        ev = fina[["ann_date", "code", col]].rename(columns={col: "value"})
        frames[name] = _pt_frame(ev, cal, codes, code_map)

    bps = frames.pop("bps")
    eps = frames.pop("eps")
    frames["pb"] = close.reindex(index=cal).reindex(columns=codes) / bps
    frames["pb"] = frames["pb"].where(bps > 0)
    frames["ep"] = eps / close.reindex(index=cal).reindex(columns=codes)
    frames["rev_yoy"] = _growth_frame(inc, cal, codes, code_map, "revenue")
    frames["np_yoy"] = _growth_frame(inc, cal, codes, code_map, "n_income_attr_p")
    return frames
