# -*- coding: utf-8 -*-
"""行情数据：sina 日线（qfq），落盘缓存。"""

import time
from pathlib import Path

import akshare as ak
import pandas as pd


def sina_symbol(code):
    if code.startswith(("6", "9", "5")):
        return "sh" + code
    if code.startswith(("0", "2", "3")):
        return "sz" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sh" + code


def _fetch(symbol, start_date, end_date, adjust="qfq"):
    last = None
    for i in range(4):
        try:
            return ak.stock_zh_a_daily(
                symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust
            )
        except Exception as e:
            last = e
            time.sleep(1 + i)
    raise last


def fetch_stock_price(code, start_date="20250101", end_date="20261231", cache_dir=None, refresh=False):
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{code}.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            out = df[["date", "open", "close"]]
            if "volume" in df.columns:
                out["volume"] = df["volume"]
            else:
                # 旧缓存没有成交量：重新抓取（成交量过滤需要）
                df = _fetch(sina_symbol(code), start_date, end_date)
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                df = df[["date", "open", "close", "volume"]].copy()
                df.to_csv(path, index=False)
                return df
            return out
    df = _fetch(sina_symbol(code), start_date, end_date)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[["date", "open", "close", "volume"]].copy()
    if cache_dir:
        df.to_csv(cache_dir / f"{code}.csv", index=False)
    return df


def fetch_index(cache_file=None, refresh=False):
    if cache_file:
        cache_file = Path(cache_file)
        if cache_file.exists() and not refresh:
            df = pd.read_csv(cache_file, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            return df[["date", "open", "close"]]
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[["date", "open", "close"]].copy()
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
    return df
