"""Tushare 数据客户端（支持自定义代理地址）。

配置项从项目根目录 .env 读取：
    TUSHARE_TOKEN   Tushare token
    TUSHARE_URL     代理地址，默认 https://t.xiaodefa.top/
    TUSHARE_ENABLED  1 时优先走 Tushare，失败由调用方回退腾讯行情

增量刷新默认走“按交易日批量拉取”：每天 3 次请求覆盖全市场
（daily / daily_basic / adj_factor），避免逐股上千次请求压垮代理。
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .panel_schema import AMOUNT_CNE_TO_ENGINE, TURNOVER_PERCENT_TO_RATIO

try:
    import tushare as ts
except ImportError:  # pragma: no cover
    ts = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()
TUSHARE_URL = os.getenv("TUSHARE_URL", "").strip() or "https://t.xiaodefa.top/"
TUSHARE_ENABLED = os.getenv("TUSHARE_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
TUSHARE_SLEEP = max(0.0, float(os.getenv("TUSHARE_SLEEP", "0.1")))

# 6 位代码 -> Tushare 市场后缀
_TS_CODE_RULES = (
    ("60", ".SH"), ("68", ".SH"), ("9", ".SH"),
    ("00", ".SZ"), ("30", ".SZ"), ("20", ".SZ"), ("12", ".SZ"),
    ("43", ".BJ"), ("83", ".BJ"), ("87", ".BJ"), ("92", ".BJ"),
)

INDEX_TS_MAP = {
    "sh000300": "000300.SH",
    "sh000905": "000905.SH",
    "sh000852": "000852.SH",
    "sz399006": "399006.SZ",
    "sh000688": "000688.SH",
    "sh000001": "000001.SH",
}

_pro: object | None = None
_pro_lock = threading.Lock()

_DAILY_COLS = ["date", "open", "close", "high", "low", "volume", "turnover", "amount", "code"]


def to_ts_code(code: str) -> str:
    """000001 -> 000001.SZ；688981 -> 688981.SH。"""
    code = code.strip().zfill(6)
    for prefix, suffix in _TS_CODE_RULES:
        if code.startswith(prefix):
            return code + suffix
    return code + ".SH"


def get_pro():
    """返回绑定代理地址的 Tushare pro 对象（进程内单例）。"""
    global _pro
    if ts is None:
        raise RuntimeError("tushare 未安装，请先 pip install tushare")
    if _pro is None:
        with _pro_lock:
            if _pro is None:
                if not TUSHARE_TOKEN:
                    raise RuntimeError("未配置 TUSHARE_TOKEN（quant_ui/.env）")
                pro = ts.pro_api(TUSHARE_TOKEN)
                pro._DataApi__http_url = TUSHARE_URL
                _pro = pro
    return _pro


def is_configured() -> bool:
    return TUSHARE_ENABLED and bool(TUSHARE_TOKEN)


def _dt(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _empty(code: str = "") -> pd.DataFrame:
    return pd.DataFrame(columns=_DAILY_COLS)


def _norm_num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def trade_dates(start: str, end: str) -> list[str]:
    """区间内 A 股交易日（YYYYMMDD），失败返回空列表。"""
    if not is_configured():
        return []
    try:
        pro = get_pro()
        df = pro.trade_cal(exchange="SSE", start_date=_dt(start), end_date=_dt(end))
        if df is None or df.empty:
            return []
        df = df[df["is_open"] == 1]
        return sorted(df["cal_date"].astype(str).tolist())
    except Exception:
        return []


def _fetch_date_batch(pro, trade_date: str, sleep: float) -> pd.DataFrame:
    """拉取单交易日全市场行情并计算前复权比例因子，返回统一列。"""
    daily = pro.daily(trade_date=trade_date)
    if daily is None or daily.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "ts_code": daily["ts_code"].astype(str),
        "date": pd.to_datetime(daily["trade_date"], format="%Y%m%d"),
        "open": _norm_num(daily["open"]),
        "close": _norm_num(daily["close"]),
        "high": _norm_num(daily["high"]),
        "low": _norm_num(daily["low"]),
        "volume": _norm_num(daily["vol"]),
        "amount": _norm_num(daily["amount"]) * AMOUNT_CNE_TO_ENGINE,
        "turnover": np.nan,
        "factor": 1.0,
    })
    try:
        basic = pro.daily_basic(trade_date=trade_date, fields="ts_code,trade_date,turnover_rate")
        if basic is not None and len(basic):
            basic = basic[["ts_code", "turnover_rate"]].copy()
            basic["turnover_rate"] = _norm_num(basic["turnover_rate"])
            out = out.merge(basic, on="ts_code", how="left")
            # 面板统一存换手率比例（0.43% -> 0.0043），与旧腾讯数据口径一致
            out["turnover"] = out.pop("turnover_rate") / TURNOVER_PERCENT_TO_RATIO
    except Exception:
        pass
    try:
        fac = pro.adj_factor(trade_date=trade_date, fields="ts_code,trade_date,adj_factor")
        if fac is not None and len(fac):
            fac = fac[["ts_code", "adj_factor"]].copy()
            fac["adj_factor"] = _norm_num(fac["adj_factor"]).fillna(1.0)
            out = out.merge(fac, on="ts_code", how="left")
            out["factor"] = out.pop("adj_factor").fillna(1.0)
    except Exception:
        # 复权因子失败时按 1.0 处理：最近增量日期 qfq 即原始价，可接受
        pass
    time.sleep(sleep)
    return out


def fetch_daily_batch(start: str, end: str) -> pd.DataFrame:
    """按交易日批量拉全市场日线（前复权），返回标准面板列。"""
    if not is_configured():
        return _empty()
    dates = trade_dates(start, end)
    frames = []
    for d in dates:
        try:
            sub = _fetch_date_batch(get_pro(), d, TUSHARE_SLEEP)
            if len(sub):
                frames.append(sub)
        except Exception:
            continue
    if not frames:
        return _empty()
    out = pd.concat(frames, ignore_index=True)
    # 前复权：价格 * 当日因子 / 区间末因子
    latest = out.groupby("ts_code")["factor"].max()
    out["latest_factor"] = out["ts_code"].map(latest).fillna(1.0)
    out["open"] = out["open"] * out["factor"] / out["latest_factor"]
    out["close"] = out["close"] * out["factor"] / out["latest_factor"]
    out["high"] = out["high"] * out["factor"] / out["latest_factor"]
    out["low"] = out["low"] * out["factor"] / out["latest_factor"]
    out["code"] = out["ts_code"].str.slice(0, 6)
    out = out.dropna(subset=["date", "close"])
    return out[["date", "open", "close", "high", "low", "volume", "turnover", "amount", "code"]] \
        .sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last") \
        .reset_index(drop=True)


def fetch_daily(code: str, start: str, end: str, retries: int = 2) -> pd.DataFrame:
    """拉取单只股票前复权日线，失败返回空表（由调用方回退腾讯）。"""
    if not is_configured():
        return _empty(code)
    try:
        pro = get_pro()
        ts_code = to_ts_code(code)
        daily = pro.daily(ts_code=ts_code, start_date=_dt(start), end_date=_dt(end))
        if daily is None or daily.empty:
            return _empty(code)
        out = pd.DataFrame({
            "date": pd.to_datetime(daily["trade_date"], format="%Y%m%d", errors="coerce"),
            "open": _norm_num(daily["open"]),
            "close": _norm_num(daily["close"]),
            "high": _norm_num(daily["high"]),
            "low": _norm_num(daily["low"]),
            "volume": _norm_num(daily["vol"]),
            "amount": _norm_num(daily["amount"]) * AMOUNT_CNE_TO_ENGINE,
            "turnover": np.nan,
        })
        fac = None
        try:
            fac = pro.adj_factor(ts_code=ts_code, start_date=_dt(start), end_date=_dt(end))
        except Exception:
            fac = None
        if fac is None or fac.empty:
            # 拿不到复权因子时宁可回退腾讯，避免把未复权价混进 qfq 面板
            return _empty(code)
        f = fac.copy()
        f["trade_date"] = pd.to_datetime(f["trade_date"], format="%Y%m%d")
        f = f.set_index("trade_date")["adj_factor"]
        f = pd.to_numeric(f, errors="coerce").fillna(1.0)
        latest = float(f.iloc[-1]) if len(f) else 1.0
        out["factor"] = out["date"].map(f).fillna(1.0)
        out["open"] = out["open"] * out["factor"] / latest
        out["close"] = out["close"] * out["factor"] / latest
        out["high"] = out["high"] * out["factor"] / latest
        out["low"] = out["low"] * out["factor"] / latest
        out = out.drop(columns=["factor"])
        try:
            db = pro.daily_basic(
                ts_code=ts_code,
                start_date=_dt(start),
                end_date=_dt(end),
                fields="ts_code,trade_date,turnover_rate",
            )
            if db is not None and len(db):
                tr = db.set_index("trade_date")["turnover_rate"]
                tr.index = pd.to_datetime(tr.index, format="%Y%m%d")
                out["turnover"] = out["date"].map(tr) / TURNOVER_PERCENT_TO_RATIO
        except Exception:
            pass
        out = out.dropna(subset=["date", "close"])
        out["code"] = code.strip().zfill(6)
        time.sleep(TUSHARE_SLEEP)
        return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    except Exception:
        if retries > 0:
            time.sleep(0.4 * (3 - retries))
            return fetch_daily(code, start, end, retries - 1)
        return _empty(code)


def fetch_index(symbol: str, name: str, start: str, end: str, retries: int = 2) -> pd.DataFrame:
    """拉取指数日线（open/close），失败返回空表。"""
    if not is_configured():
        return pd.DataFrame()
    ts_code = INDEX_TS_MAP.get(symbol)
    if ts_code is None:
        return pd.DataFrame()
    try:
        pro = get_pro()
        df = pro.index_daily(ts_code=ts_code, start_date=_dt(start), end_date=_dt(end))
        if df is None or df.empty:
            return pd.DataFrame()
        out = pd.DataFrame({
            "date": pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce"),
            "open": _norm_num(df["open"]),
            "close": _norm_num(df["close"]),
            "code": symbol,
            "name": name,
        })
        out = out.dropna(subset=["date", "close"])
        time.sleep(TUSHARE_SLEEP)
        return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    except Exception:
        if retries > 0:
            time.sleep(0.4 * (3 - retries))
            return fetch_index(symbol, name, start, end, retries - 1)
        return pd.DataFrame()
