from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from .store import (ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE, FUND_PANEL_FILE,
                    INDEX_FILE, PANEL_FILE, TECH_FILE, UNIVERSE_FILE,
                    load_meta)


def duck_query(sql: str, params=None) -> pd.DataFrame:
    """DuckDB SQL 查询（基于 data/ 下 parquet/csv 的视图）。"""
    from .db import query
    return query(sql, params)


PANEL_PATH = Path("/home/ubuntu/quant_data/panel/turn20/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet")
UNIVERSE_PATH = Path("/home/ubuntu/quant_data/panel/universe_cs800.csv")
TECH_PATH = Path("/home/ubuntu/quant_data/panel/tech_universe_sw.csv")
INDEX_PATH = Path("/home/ubuntu/quant_data/panel/csi300_index.csv")

# PG 面板只拉该日期以来的行情，避免全表 1100 万行进入 pandas（与 parquet 面板口径一致）。
PANEL_START = os.getenv("QUANT_PANEL_START", "2020-01-01").strip()
# turn20/am20 滚动窗口在区间起点需要约 20 个交易日历史，查询起点统一前移自然日缓冲，
# 保证区间化加载与全量加载的因子口径一致。
FACTOR_BUFFER_DAYS = 40

# pg=优先 PostgreSQL stock_daily（未补全/失败时自动回退 parquet），panel=只用本地文件
DATA_SOURCE = os.getenv("QUANT_DATA_SOURCE", "pg").strip().lower()
_pg_panel_cache: dict = {}
_panel_codes_cache: set | None = None


def _finalize_stock_df(df: pd.DataFrame, last_adj: pd.DataFrame | None = None) -> pd.DataFrame:
    """前复权 + turn20/am20 因子 + 类型压缩（PG stock_daily -> 面板口径）。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["ts_code"].str[:6]
    # PG 统一存 %（0.4289），面板历史口径是比例（0.0043），转回比例保持一致
    df["turnover"] = df["turnover"] / 100.0
    if last_adj is None:
        # 每只股票一行最新复权因子；不能用全量 df（会同代码笛卡尔合并爆炸）
        last_adj = (df.groupby("ts_code", observed=True)["adj_factor"]
                    .last().reset_index()
                    .rename(columns={"adj_factor": "last_adj"}))
    df = df.merge(last_adj, on="ts_code", how="left")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * df["adj_factor"] / df["last_adj"]
    df = df.sort_values(["ts_code", "date"])
    g = df.groupby("ts_code", observed=True)
    df["turn20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    df["am20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    for c in ("turn20", "am20", "volume"):
        df[c] = df[c].astype("float32")
    df["code"] = df["code"].astype("category")
    return df[["date", "open", "high", "low", "close", "turnover", "amount", "code",
               "turn20", "am20", "volume"]]


def _load_panel_pg(start: str | None = None, end: str | None = None,
                   codes: list[str] | None = None) -> pd.DataFrame:
    """从 PostgreSQL stock_daily 构建回测面板（前复权 + turn20/am20 因子）。

    start/end 可选，只拉回测所需区间；实际查询起点前移 FACTOR_BUFFER_DAYS，
    保证因子滚动窗口与全量加载口径一致。codes 限定股票池（如科技TMT 90 只），
    避免 5600 只全市场行情进入 pandas。
    """
    global _pg_panel_cache
    from .pg import configured as pg_configured, query_df
    if not pg_configured():
        raise RuntimeError("PG_DSN 未配置")
    meta_end = query_df(
        "SELECT max(trade_date) AS end FROM stock_daily"
    ).iloc[0]["end"]
    if meta_end is None:
        raise RuntimeError("stock_daily 为空")
    calc_start = start or PANEL_START
    if start:
        calc_start = (pd.Timestamp(start)
                      - pd.Timedelta(days=FACTOR_BUFFER_DAYS)).date().isoformat()
    codes_key = tuple(sorted(codes)) if codes else None
    key = (calc_start, end or str(meta_end), codes_key)
    if _pg_panel_cache.get("key") == key:
        return _pg_panel_cache["df"]
    where = ["adj_factor IS NOT NULL", "close IS NOT NULL"]
    params: list = []
    if calc_start:
        where.append("trade_date >= %s")
        params.append(calc_start)
    if end:
        where.append("trade_date <= %s")
        params.append(end)
    if codes:
        like_conds = " OR ".join(["ts_code LIKE %s"] * len(codes))
        where.append(f"({like_conds})")
        params.extend(f"{c}%" for c in codes)
    df = query_df(
        "SELECT ts_code, trade_date AS date, open, high, low, close, vol AS volume, amount, "
        "turnover_rate AS turnover, adj_factor FROM stock_daily "
        "WHERE " + " AND ".join(where) + " ORDER BY ts_code, trade_date",
        tuple(params),
    )
    if len(df) == 0:
        raise RuntimeError("所选区间/股票池在 stock_daily 中没有数据")
    if df["adj_factor"].isna().mean() > 0.05:
        raise RuntimeError("stock_daily 复权因子覆盖不足，历史补全后自动切换")
    # last_adj 直接取区间内各股票最新 adj_factor：回测终点通常是数据最新日，
    # 与全量口径一致；终点早于最新日时按区间内最新复权，避免全表 DISTINCT ON。
    out = _finalize_stock_df(df)
    _pg_panel_cache = {"key": key, "df": out}
    return out


def load_panel(start: str | None = None, end: str | None = None,
               codes: list[str] | None = None) -> pd.DataFrame:
    if DATA_SOURCE == "pg":
        try:
            return _load_panel_pg(start=start, end=end, codes=codes)
        except Exception as exc:
            print(f"PG 面板加载失败，回退 parquet: {exc}", file=sys.stderr)
    if PANEL_FILE.exists():
        path = PANEL_FILE
    elif PANEL_PATH.exists():
        path = PANEL_PATH
    else:
        raise FileNotFoundError(f"面板数据不存在: {PANEL_PATH} 或 {PANEL_FILE}")
    # parquet 读取时用 pyarrow 下推 code/date 过滤，只把回测所需行读进 pandas；
    # 版本或类型不兼容时回退全量读取，再在 pandas 里过滤（行为不变）。
    filters: list = []
    if codes:
        filters.append(("code", "in", [str(c).zfill(6) for c in codes]))
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))
    try:
        panel = pd.read_parquet(path, filters=filters or None)
    except Exception:
        panel = pd.read_parquet(path)
        if start:
            panel = panel[panel["date"] >= pd.Timestamp(start)]
        if end:
            panel = panel[panel["date"] <= pd.Timestamp(end)]
        if codes:
            panel = panel[panel["code"].isin([str(c).zfill(6) for c in codes])]
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    # 因子列/成交量降 float32（排序分位精度足够），价格与金额保持 float64；
    # code 用 category 省约 60MB 常驻内存
    for c in ("turn20", "am20", "volume"):
        if c in panel.columns and panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel["code"] = panel["code"].astype("category")
    return panel


def load_panel_codes() -> set[str]:
    """轻量返回股票面板全部代码（不加载整张面板），用于构建股票池。"""
    global _panel_codes_cache
    if _panel_codes_cache is not None:
        return _panel_codes_cache
    codes: set[str] = set()
    if DATA_SOURCE == "pg":
        try:
            from .pg import configured as pg_configured, query_df
            if pg_configured():
                # stock_basic 是静态股票注册表（5000+ 行），避免对 1174 万行日线做 DISTINCT
                df = query_df("SELECT ts_code FROM stock_basic WHERE ts_code IS NOT NULL")
                codes = {str(c)[:6].zfill(6) for c in df["ts_code"]}
        except Exception:
            codes = set()
    if not codes:
        path = PANEL_FILE if PANEL_FILE.exists() else PANEL_PATH
        if path.exists():
            codes = set(pd.read_parquet(path, columns=["code"])
                        ["code"].astype(str).str.zfill(6).unique())
    _panel_codes_cache = codes
    return codes


def load_stock_detail(code: str, days: int = 250) -> pd.DataFrame:
    """单只股票前复权行情 + turn20/am20，只查 PG 单股，避免加载整张面板。"""
    code = str(code).zfill(6)
    if DATA_SOURCE == "pg":
        try:
            from .pg import configured as pg_configured, query_df
            if pg_configured():
                df = query_df(
                    "SELECT ts_code, trade_date AS date, open, high, low, close, "
                    "vol AS volume, amount, turnover_rate AS turnover, adj_factor "
                    "FROM stock_daily "
                    "WHERE ts_code LIKE %s AND adj_factor IS NOT NULL AND close IS NOT NULL "
                    "ORDER BY ts_code, trade_date",
                    (f"{code}%",),
                )
                if not df.empty:
                    return _finalize_stock_df(df).tail(days).reset_index(drop=True)
        except Exception:
            pass
    panel = load_panel()
    sub = panel[panel["code"] == code].sort_values("date")
    return sub.tail(days).reset_index(drop=True)


def reset_caches() -> None:
    """数据更新后清空面板/代码缓存，避免 stale。"""
    global _pg_panel_cache, _panel_codes_cache
    _pg_panel_cache = {}
    _panel_codes_cache = None


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


def load_etf_panel() -> pd.DataFrame:
    if not ETF_PANEL_FILE.exists():
        return pd.DataFrame(columns=["date", "open", "close", "turnover",
                                     "amount", "code", "turn20", "am20",
                                     "volume"])
    panel = pd.read_parquet(ETF_PANEL_FILE)
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    for c in ("turn20", "am20", "volume"):
        if c in panel.columns and panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel["code"] = panel["code"].astype("category")
    return panel


def load_fund() -> pd.DataFrame:
    if not FUND_FILE.exists():
        return pd.DataFrame(columns=["code", "name", "type"])
    fund = pd.read_csv(FUND_FILE, dtype={"code": str})
    fund["code"] = fund["code"].astype(str).str.zfill(6)
    return fund


def load_fund_nav() -> pd.DataFrame:
    if not FUND_NAV_FILE.exists():
        return pd.DataFrame(columns=["date", "code", "nav"])
    panel = pd.read_parquet(FUND_NAV_FILE)
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype("category")
    return panel


def load_fund_panel() -> pd.DataFrame:
    """基金净值标准面板（列结构与股票/ETF 面板一致，供统一回测）。"""
    if FUND_PANEL_FILE.exists():
        panel = pd.read_parquet(FUND_PANEL_FILE)
    else:
        try:
            from .fund_engine import build_fund_panel
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
    if INDEX_FILE.exists():
        idx = pd.read_csv(INDEX_FILE)
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


def data_status() -> dict:
    files = {
        "panel": {"store": PANEL_FILE, "legacy": PANEL_PATH},
        "universe": {"store": UNIVERSE_FILE, "legacy": UNIVERSE_PATH},
        "tech": {"store": TECH_FILE, "legacy": TECH_PATH},
        "index": {"store": INDEX_FILE, "legacy": INDEX_PATH},
        "etf": {"store": ETF_FILE},
        "etf_panel": {"store": ETF_PANEL_FILE},
        "fund": {"store": FUND_FILE},
        "fund_nav": {"store": FUND_NAV_FILE},
        "fund_panel": {"store": FUND_PANEL_FILE},
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
