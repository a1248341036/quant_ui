"""K线底层：腾讯行情接口、符号转换、窗口拆分、面板压缩。"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from .. import tushare_client

TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def sina_symbol(code: str) -> str:
    code = code.strip()
    # 沪市基金/ETF 5 开头，深市基金 1 开头（159/160/16/18）
    if code.startswith(("60", "68", "9", "5")):
        return "sh" + code
    if code.startswith(("00", "30", "20", "12", "15", "16", "18")):
        return "sz" + code
    return code


def split_windows(start: str, end: str, max_years: int = 2) -> list[tuple[str, str]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    cur = s
    while cur <= e:
        stop = min(e, cur + pd.DateOffset(years=max_years) - pd.DateOffset(days=1))
        out.append((cur.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        cur = stop + pd.DateOffset(days=1)
    return out


def _kline_count(start: str, end: str, max_count: int = 640) -> int:
    """按区间估算需要的K线根数：腾讯接口忽略 start，只返回最近 count 根。
    全量/长区间封顶 640，增量短区间只拉少量根数，避免每天白下 2 年数据。"""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    need = max(10, int(days * 5 / 7) + 10)  # 交易日粗估 + 缓冲
    return min(max_count, need)


def _fetch_kline(code: str, symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    rows_all: list[list] = []
    for ws, we in split_windows(start, end):
        params = {"_var": "kline_dayqfq",
                   "param": f"{symbol},day,{ws},{we},{_kline_count(ws, we)},qfq"}
        got = False
        for attempt in range(retries):
            try:
                resp = requests.get(TX_URL, params=params, headers=HEADERS, timeout=10)
                txt = resp.text
                body = txt[txt.find("=") + 1:]
                data = json.loads(body)
                d = data["data"].get(symbol, {})
                key = next((k for k in ("qfqday", "day", "hfqday") if k in d), None)
                if key is None:
                    break
                rows_all.extend(d[key])
                got = True
                break
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        if not got:
            return pd.DataFrame()
    if not rows_all:
        return pd.DataFrame()

    df = pd.DataFrame(rows_all)
    # 腾讯 kline 字段：0=date 1=open 2=close 3=high 4=low 5=volume 7=turnover 8=amount
    df = df.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8]].copy()
    df.columns = ["date", "open", "close", "high", "low", "volume", "turnover", "amount"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 单位换算常量见 core/panel_schema：腾讯 turnover 为 %、amount 为万元。
    from ..panel_schema import AMOUNT_TEN_THOUSAND_TO_ENGINE, TURNOVER_PERCENT_TO_RATIO
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / TURNOVER_PERCENT_TO_RATIO
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * AMOUNT_TEN_THOUSAND_TO_ENGINE
    df["volume"] = df["volume"].astype("float32")
    df["turnover"] = df["turnover"].astype("float32")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df["code"] = code
    return df


def _fetch_kline_any(code: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """优先 Tushare 拉日线，失败/为空时回退腾讯。"""
    if tushare_client.is_configured():
        df = tushare_client.fetch_daily(code, start, end)
        if df is not None and len(df):
            return df
    return _fetch_kline(code, symbol, start, end)


def _fetch_index_tencent(symbol: str, name: str, start: str, end: str) -> pd.DataFrame:
    """腾讯指数日线回退源。"""
    try:
        params = {"_var": "kline_dayqfq",
                   "param": f"{symbol},day,{start},{end},640,qfq"}
        resp = requests.get(TX_URL, params=params, headers=HEADERS, timeout=15)
        body = resp.text[resp.text.find("=") + 1:]
        data = json.loads(body)
        d = data["data"].get(symbol, {})
        key = next((k for k in ("day", "qfqday", "hfqday") if k in d), None)
        if key is None:
            return pd.DataFrame()
        df = pd.DataFrame(d[key])
        df = df.iloc[:, [0, 1, 4]].copy()
        df.columns = ["date", "open", "close"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        df["code"] = symbol
        df["name"] = name
        return df
    except Exception:
        return pd.DataFrame()


def _compact_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """压缩面板内存：code 转 category、成交量/因子降 float32（价格与金额保持 float64）。"""
    panel = panel.copy()
    for c in ("volume", "turnover", "turn20", "am20"):
        if c in panel.columns and panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel["code"] = panel["code"].astype("category")
    return panel


def _add_rolling_factors(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["code", "date"]).copy()
    g = panel.groupby("code", group_keys=False, observed=True)
    panel["turn20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    panel["am20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    return panel
