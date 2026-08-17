from __future__ import annotations

"""财务因子：PG 财务宽表 -> point-in-time 因子矩阵（date x code）。

财务数据按公告日（ann_date）对齐，避免未来函数：每个交易日只使用
ann_date <= 当日的最近一期财报。

当前 PG 财务表为样例覆盖（约 30 只）；补全 `python scripts/sync_postgres.py --fina`
后全市场可用。未配置 PG 或表缺失时返回空 dict，引擎自动降级为纯日线因子。
"""


import threading

import numpy as np
import pandas as pd

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


def _map_code_to_ts() -> dict[str, str]:
    with _lock:
        if "code_map" in _CACHE:
            return _CACHE["code_map"]
        from .pg import query_df
        try:
            df = query_df("SELECT ts_code, symbol FROM stock_basic")
            code_map = {str(r["symbol"]).zfill(6): str(r["ts_code"])
                        for _, r in df.iterrows()}
        except Exception:
            code_map = {}
        _CACHE["code_map"] = code_map
        return code_map


def _load_raw() -> dict[str, pd.DataFrame]:
    with _lock:
        if "raw" in _CACHE:
            return _CACHE["raw"]
        from .pg import query_df
        out: dict[str, pd.DataFrame] = {}
        try:
            fina = query_df(
                "SELECT ts_code, ann_date, end_date, bps, eps, roe_waa, "
                "grossprofit_margin FROM fina_indicator")
            inc = query_df(
                "SELECT ts_code, ann_date, end_date, revenue, n_income_attr_p "
                "FROM income")
            fina["ann_date"] = pd.to_datetime(fina["ann_date"])
            fina["end_date"] = pd.to_datetime(fina["end_date"])
            inc["ann_date"] = pd.to_datetime(inc["ann_date"])
            inc["end_date"] = pd.to_datetime(inc["end_date"])
            out["fina"] = fina
            out["inc"] = inc
        except Exception:
            out = {}
        _CACHE["raw"] = out
        return out


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
    """返回 {因子名: date x code 矩阵}，与 cal 对齐。

    因子：pb/ep/roe/gross_margin/rev_yoy/np_yoy。PG 未配置/表缺失时返回空 dict。
    """
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
