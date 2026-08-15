from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from .store import (INDEX_FILE, TECH_FILE, UNIVERSE_FILE, save_csv,
                    save_meta, save_panel)


TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def sina_symbol(code: str) -> str:
    code = code.strip()
    if code.startswith(("60", "68", "9")):
        return "sh" + code
    if code.startswith(("00", "30", "20", "12")):
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


def _fetch_kline(code: str, symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    rows_all: list[list] = []
    for ws, we in split_windows(start, end):
        params = {"_var": "kline_dayqfq", "param": f"{symbol},day,{ws},{we},640,qfq"}
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
    df = df.iloc[:, [0, 1, 2, 5, 7, 8]].copy()
    df.columns = ["date", "open", "close", "volume", "turnover", "amount"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ["open", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 10000.0
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df["code"] = code
    return df


def fetch_universe() -> pd.DataFrame:
    """沪深300 + 中证500 成分股（中证指数官网 OSS，带超时）。"""
    frames = []
    for idx in ("000300", "000905"):
        url = (f"https://oss-ch.csindex.com.cn/static/html/csindex/public/"
               f"uploads/file/autofile/cons/{idx}cons.xls")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content))
        sub = df[["成分券代码", "成分券名称"]].copy()
        sub.columns = ["code", "name"]
        frames.append(sub)
    uni = pd.concat(frames, ignore_index=True).drop_duplicates("code")
    uni["code"] = uni["code"].astype(str).str.zfill(6)
    return uni


def _load_cached_universe() -> pd.DataFrame | None:
    if UNIVERSE_FILE.exists():
        df = pd.read_csv(UNIVERSE_FILE, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    legacy = Path("/tmp/universe_cs800.csv")
    if legacy.exists():
        df = pd.read_csv(legacy, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    return None


def fetch_index(start: str, end: str) -> pd.DataFrame:
    params = {"_var": "kline_dayqfq", "param": f"sh000300,day,{start},{end},640,qfq"}
    resp = requests.get(TX_URL, params=params, headers=HEADERS, timeout=15)
    txt = resp.text
    body = txt[txt.find("=") + 1:]
    data = json.loads(body)
    d = data["data"].get("sh000300", {})
    key = next((k for k in ("day", "qfqday", "hfqday") if k in d), None)
    if key is None:
        raise RuntimeError("腾讯指数接口返回为空")
    df = pd.DataFrame(d[key])
    df = df.iloc[:, [0, 1, 4]].copy()
    df.columns = ["date", "open", "close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date")


def fetch_tech_universe(universe: pd.DataFrame) -> pd.DataFrame:
    """行业分类尽量抓取；东方财富接口被断连时回退本地缓存并和最新股票池取交集。"""
    industries = {"电子", "计算机", "通信", "传媒"}
    frames = []
    try:
        import akshare as ak

        for name in industries:
            df = ak.stock_board_industry_cons_em(symbol=name)
            code_col = next(c for c in df.columns if "代码" in c)
            name_col = next(c for c in df.columns if "名称" in c)
            sub = df[[code_col, name_col]].copy()
            sub.columns = ["code", "name"]
            sub["industry"] = name
            frames.append(sub)
        tech = pd.concat(frames, ignore_index=True).drop_duplicates("code")
        tech["code"] = tech["code"].astype(str).str.zfill(6)
    except Exception:
        tech = _load_cached_tech()
        if tech is not None:
            tech = tech[tech["code"].isin(set(universe["code"]))].copy()
    if tech is None or tech.empty:
        raise RuntimeError("无法获取行业分类且本地无行业缓存")
    return tech


def _load_cached_tech() -> pd.DataFrame | None:
    if TECH_FILE.exists():
        df = pd.read_csv(TECH_FILE, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    legacy = Path("/tmp/tech_universe_sw.csv")
    if legacy.exists():
        df = pd.read_csv(legacy, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    return None


def fetch_daily_bars(
    codes: list[str],
    start: str,
    end: str,
    existing: pd.DataFrame | None,
    max_workers: int = 12,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """增量抓取日线：缓存里已到最新日期的代码跳过，只抓新增区间。"""
    end_ts = pd.Timestamp(end)
    last_by_code: dict[str, pd.Timestamp] = {}
    if existing is not None and len(existing):
        last_by_code = existing.groupby("code")["date"].max().to_dict()

    tasks: list[tuple[str, str, str]] = []
    for code in codes:
        last = last_by_code.get(code)
        if last is not None and last >= end_ts:
            continue
        fetch_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last is not None else start
        if fetch_start > end:
            continue
        tasks.append((code, sina_symbol(code), fetch_start))

    frames: list[pd.DataFrame] = []
    total = len(tasks)
    done = 0

    def worker(task):
        code, symbol, fetch_start = task
        return _fetch_kline(code, symbol, fetch_start, end)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                df = fut.result()
                if len(df):
                    frames.append(df)
            except Exception:
                pass
            done += 1
            if progress:
                progress(done, total, code)

    if not frames and existing is not None:
        return existing
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "open", "close", "volume", "turnover", "amount", "code"])
    if existing is not None and len(existing):
        panel = pd.concat([existing, new], ignore_index=True)
        panel = panel.drop_duplicates(["code", "date"], keep="last")
    else:
        panel = new
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)]
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    panel = _add_rolling_factors(panel)
    return panel


def _add_rolling_factors(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["code", "date"]).copy()
    g = panel.groupby("code", group_keys=False)
    panel["turn20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    panel["am20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    return panel


def update_data(
    mode: str = "incremental",
    start: str = "2020-01-01",
    end: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """一键更新：股票池 + 指数 + 日线缓存。"""
    from .store import PANEL_FILE, save_panel as _save_panel

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    stage = {"text": "正在更新股票池..."}
    if progress:
        progress(0, 4, stage["text"])
    try:
        universe = fetch_universe()
        save_csv(universe, UNIVERSE_FILE)
    except Exception:
        universe = _load_cached_universe()
        if universe is None:
            raise
    if progress:
        progress(1, 4, "股票池完成，更新指数...")
    index = fetch_index(start, end)
    save_csv(index, INDEX_FILE)
    if progress:
        progress(2, 4, "指数完成，更新行业分类...")
    tech = fetch_tech_universe(universe)
    save_csv(tech, TECH_FILE)
    if progress:
        progress(3, 4, "行业完成，增量更新日线...")

    existing = None
    if not PANEL_FILE.exists():
        legacy = Path("/tmp/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet")
        if legacy.exists():
            _save_panel(_add_rolling_factors(pd.read_parquet(legacy)))
    if PANEL_FILE.exists():
        existing = pd.read_parquet(PANEL_FILE)
        existing["code"] = existing["code"].astype(str).str.zfill(6)

    panel = fetch_daily_bars(
        sorted(universe["code"]),
        start=start,
        end=end,
        existing=existing,
        progress=lambda d, t, c: progress(3 + d / max(t, 1) * 0.9, 4, f"日线 {c} ({d}/{t})")
        if progress else None,
    )
    save_panel(panel)
    save_meta({"mode": mode, "start": start, "end": end,
               "n_codes": int(panel["code"].nunique()),
               "n_rows": int(len(panel))})
    return {"n_codes": int(panel["code"].nunique()), "n_rows": int(len(panel))}
