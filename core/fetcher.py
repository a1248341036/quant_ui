from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd
import requests

from .store import (ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE,
                    INDEX_FILE, LEGACY_DATA_DIR, TECH_FILE, UNIVERSE_FILE, save_csv,
                    save_meta, save_panel)
from . import tushare_client


TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

INDEX_SYMBOLS = {
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sh000852": "中证1000",
    "sz399006": "创业板指",
    "sh000688": "科创50",
    "sh000001": "上证指数",
}


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
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 10000.0
    df["volume"] = df["volume"].astype("float32")
    df["turnover"] = df["turnover"].astype("float32")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df["code"] = code
    return df


def fetch_universe() -> pd.DataFrame:
    """沪深300 + 中证500 + 中证1000 成分股（中证指数官网 OSS，带超时）。"""
    frames = []
    for idx in ("000300", "000905", "000852"):
        url = (f"https://oss-ch.csindex.com.cn/static/html/csindex/public/"
               f"uploads/file/autofile/cons/{idx}cons.xls")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content))
        code_col = next(c for c in df.columns if "成份券代码" in c or "成分券代码" in c)
        name_col = next(c for c in df.columns if "成份券名称" in c or "成分券名称" in c)
        sub = df[[code_col, name_col]].copy()
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
    legacy = LEGACY_DATA_DIR / "panel/universe_cs800.csv"
    if legacy.exists():
        df = pd.read_csv(legacy, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    return None


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


def fetch_indices(start: str, end: str) -> pd.DataFrame:
    """抓取多个 A股指数日线，优先 Tushare，失败回退腾讯。"""
    frames = []
    for symbol, name in INDEX_SYMBOLS.items():
        df = tushare_client.fetch_index(symbol, name, start, end)
        if df is None or df.empty:
            df = _fetch_index_tencent(symbol, name, start, end)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        raise RuntimeError("指数数据源全部失败（Tushare + 腾讯）")
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "code", "name", "open", "close"]].sort_values(["code", "date"])


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
    legacy = LEGACY_DATA_DIR / "panel/tech_universe_sw.csv"
    if legacy.exists():
        df = pd.read_csv(legacy, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    return None


def fetch_etf_universe() -> pd.DataFrame:
    """全市场 ETF 列表（东财快照），失败时回退本地缓存。"""
    try:
        import akshare as ak
        df = ak.fund_etf_spot_em()
        code_col = next(c for c in df.columns if "代码" in c)
        name_col = next(c for c in df.columns if "名称" in c)
        out = df[[code_col, name_col]].copy()
        out.columns = ["code", "name"]
        out["code"] = out["code"].astype(str).str.zfill(6)
        out = out.drop_duplicates("code")
        return out
    except Exception:
        if ETF_FILE.exists():
            df = pd.read_csv(ETF_FILE, dtype={"code": str})
            df["code"] = df["code"].astype(str).str.zfill(6)
            return df
        raise RuntimeError("无法获取 ETF 列表且本地无缓存")


def fetch_etf_daily_bars(
    codes: list[str],
    start: str,
    end: str,
    existing: pd.DataFrame | None,
    max_workers: int = 6,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """ETF 日线（腾讯行情，不走 Tushare 股票日线接口）。

    字段与股票 panel 一致：date/open/close/volume/turnover/amount + turn20/am20。
    腾讯对 ETF 返回 qfqday 前复权序列，可直接复用 _fetch_kline。
    """
    end_ts = pd.Timestamp(end)
    last_by_code: dict[str, pd.Timestamp] = {}
    if existing is not None and len(existing):
        last_by_code = existing.groupby("code", observed=True)["date"].max().to_dict()

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
        return _compact_panel(existing)
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "open", "close", "high", "low", "volume", "turnover", "amount", "code"])
    if existing is not None and len(existing):
        panel = pd.concat([existing, new], ignore_index=True)
        panel = panel.drop_duplicates(["code", "date"], keep="last")
    else:
        panel = new
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)]
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    panel = _add_rolling_factors(panel)
    return _compact_panel(panel)


FUND_TECH_KEYWORDS = (
    "科技", "半导体", "芯片", "电子", "计算机", "通信", "传媒", "人工智能",
    "数字", "软件", "信息", "互联网", "5G", "智能", "科创", "创新",
)


def fetch_fund_universe(keywords: tuple[str, ...] = FUND_TECH_KEYWORDS) -> pd.DataFrame:
    """科技相关场外基金池：天天基金全量列表按简称/类型过滤。"""
    try:
        import akshare as ak
        df = ak.fund_name_em()
        code_col = next(c for c in df.columns if "基金代码" in c)
        name_col = next(c for c in df.columns if "基金简称" in c)
        type_col = next(c for c in df.columns if "基金类型" in c)
        out = df[[code_col, name_col, type_col]].copy()
        out.columns = ["code", "name", "type"]
        out["code"] = out["code"].astype(str).str.zfill(6)
        out["name"] = out["name"].astype(str)
        out["type"] = out["type"].astype(str)
        # 只保留权益类（股票/混合/指数/QDII），剔除债券/货币/理财
        keep_type = out["type"].str.contains(
            "股票|混合|指数|QDII", regex=True, na=False)
        keep_name = out["name"].str.contains(
            "|".join(keywords), regex=True, na=False)
        # 剔除场内 ETF（东财基金列表会混入 ETF，代码与场外重复/净值口径不同）
        keep_otc = ~out["name"].str.contains("ETF", na=False)
        out = out[keep_type & keep_name & keep_otc].drop_duplicates("code")
        return out.reset_index(drop=True)
    except Exception:
        if FUND_FILE.exists():
            df = pd.read_csv(FUND_FILE, dtype={"code": str})
            df["code"] = df["code"].astype(str).str.zfill(6)
            return df
        raise RuntimeError("无法获取场外基金列表且本地无缓存")


def fetch_fund_navs(
    codes: list[str],
    start: str,
    end: str,
    existing: pd.DataFrame | None,
    max_workers: int = 6,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """场外基金历史单位净值（天天基金，逐只抓取）。"""
    end_ts = pd.Timestamp(end)
    last_by_code: dict[str, pd.Timestamp] = {}
    if existing is not None and len(existing):
        last_by_code = existing.groupby("code", observed=True)["date"].max().to_dict()

    tasks: list[str] = []
    for code in codes:
        last = last_by_code.get(code)
        if last is not None and last >= end_ts:
            continue
        if last is not None and (end_ts - last).days <= 3:
            continue  # 净值 T+1 公布，3 天内可能还没更新，避免重复请求
        tasks.append(code)

    frames: list[pd.DataFrame] = []
    total = len(tasks)
    done = 0

    def worker(code: str):
        import akshare as ak
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or len(df) == 0:
            return pd.DataFrame()
        date_col = next(c for c in df.columns if "日期" in c)
        nav_col = next(c for c in df.columns if "单位净值" in c)
        out = df[[date_col, nav_col]].copy()
        out.columns = ["date", "nav"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
        out = out.dropna(subset=["date", "nav"])
        out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= end_ts)]
        out["code"] = code
        return out[["date", "code", "nav"]]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, code): code for code in tasks}
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
        columns=["date", "code", "nav"])
    if existing is not None and len(existing):
        panel = pd.concat([existing, new], ignore_index=True)
        panel = panel.drop_duplicates(["code", "date"], keep="last")
    else:
        panel = new
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    panel["code"] = panel["code"].astype("category")
    return panel

def _fetch_kline_any(code: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """优先 Tushare 拉日线，失败/为空时回退腾讯。"""
    if tushare_client.is_configured():
        df = tushare_client.fetch_daily(code, start, end)
        if df is not None and len(df):
            return df
    return _fetch_kline(code, symbol, start, end)


def fetch_daily_bars(
    codes: list[str],
    start: str,
    end: str,
    existing: pd.DataFrame | None,
    max_workers: int = 6,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """增量抓取日线：缓存里已到最新日期的代码跳过，只抓新增区间。"""
    end_ts = pd.Timestamp(end)
    last_by_code: dict[str, pd.Timestamp] = {}
    if existing is not None and len(existing):
        last_by_code = existing.groupby("code", observed=True)["date"].max().to_dict()

    per_code_tasks: list[tuple[str, str, str]] = []
    batch_codes: set[str] = set()
    batch_start: str | None = None
    for code in codes:
        last = last_by_code.get(code)
        if last is not None and last >= end_ts:
            continue
        fetch_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last is not None else start
        if fetch_start > end:
            continue
        # 尾部缺少量交易日的代码合并成“按交易日批量”请求，避免逐股上千次调用
        if last is not None and (end_ts - last).days <= 7:
            batch_codes.add(code)
            if batch_start is None or fetch_start < batch_start:
                batch_start = fetch_start
        else:
            per_code_tasks.append((code, sina_symbol(code), fetch_start))

    frames: list[pd.DataFrame] = []
    total = len(per_code_tasks) + (1 if batch_codes else 0)
    done = 0

    if batch_codes and batch_start and batch_start <= end:
        try:
            batch = tushare_client.fetch_daily_batch(batch_start, end)
            if batch is not None and len(batch):
                batch = batch[batch["code"].isin(batch_codes)]
                if len(batch):
                    frames.append(batch)
        except Exception:
            # 批量失败时这些代码并入逐股路径（Tushare 单股 -> 腾讯回退）
            for code in batch_codes:
                last = last_by_code.get(code)
                fetch_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last is not None else start
                if fetch_start <= end:
                    per_code_tasks.append((code, sina_symbol(code), fetch_start))
        done += 1
        if progress:
            progress(done, total, "批量(Tushare)")

    def worker(task):
        code, symbol, fetch_start = task
        return _fetch_kline_any(code, symbol, fetch_start, end)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t[0] for t in per_code_tasks}
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
        return _compact_panel(existing)
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "open", "close", "high", "low", "volume", "turnover", "amount", "code"])
    if existing is not None and len(existing):
        panel = pd.concat([existing, new], ignore_index=True)
        panel = panel.drop_duplicates(["code", "date"], keep="last")
    else:
        panel = new
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)]
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    panel = _add_rolling_factors(panel)
    return _compact_panel(panel)


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


def update_data(
    mode: str = "incremental",
    start: str = "2020-01-01",
    end: str | None = None,
    max_workers: int = 6,
    progress: Callable[[int, int, str], None] | None = None,
    include_stocks: bool = True,
) -> dict:
    """一键更新：股票池 + 指数 + 日线缓存。"""
    from .store import PANEL_FILE, save_panel as _save_panel

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    # include_stocks=False（股票由 Tushare PG 承担）时不会刷新股票面板，
    # 初始化空表避免后续 save_meta 引用未定义变量。
    panel = pd.DataFrame(columns=["date", "code"])
    stage = {"text": "正在更新股票池..."}
    if progress:
        progress(0, 6, stage["text"])
    try:
        universe = fetch_universe()
        save_csv(universe, UNIVERSE_FILE)
    except Exception:
        universe = _load_cached_universe()
        if universe is None:
            raise
    if progress:
        progress(1, 6, "股票池完成，更新指数...")
    index = fetch_indices(start, end)
    save_csv(index, INDEX_FILE)
    if progress:
        progress(2, 6, "指数完成，更新行业分类...")
    tech = fetch_tech_universe(universe)
    save_csv(tech, TECH_FILE)
    if progress:
        progress(3, 6, "行业完成，增量更新日线...")

    if include_stocks:
        existing = None
        if mode != "full":
            if not PANEL_FILE.exists():
                legacy = LEGACY_DATA_DIR / "panel/turn20/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet"
                if legacy.exists():
                    _save_panel(_add_rolling_factors(pd.read_parquet(legacy)))
            if PANEL_FILE.exists():
                existing = pd.read_parquet(PANEL_FILE)
                existing["code"] = existing["code"].astype(str).str.zfill(6)
                existing = _compact_panel(existing)

        panel = fetch_daily_bars(
            sorted(universe["code"]),
            start=start,
            end=end,
            existing=existing,
            max_workers=max_workers,
            progress=lambda d, t, c: progress(3 + d / max(t, 1) * 0.9, 6, f"日线 {c} ({d}/{t})")
            if progress else None,
        )
        save_panel(panel)
    else:
        print("跳过腾讯股票日线（股票行情由 Tushare PG 承担）", flush=True)

    # ---- ETF：池子 + 日线 ----
    if progress:
        progress(4.4, 6, "更新 ETF 列表...")
    try:
        etf = fetch_etf_universe()
        save_csv(etf, ETF_FILE)
        etf_existing = None
        if ETF_PANEL_FILE.exists():
            etf_existing = pd.read_parquet(ETF_PANEL_FILE)
            etf_existing["code"] = etf_existing["code"].astype(str).str.zfill(6)
            etf_existing = _compact_panel(etf_existing)
        etf_panel = fetch_etf_daily_bars(
            sorted(etf["code"]),
            start=start,
            end=end,
            existing=etf_existing,
            max_workers=max_workers,
            progress=lambda d, t, c: progress(4.5 + d / max(t, 1) * 0.5, 6,
                                               f"ETF {c} ({d}/{t})")
            if progress else None,
        )
        if len(etf_panel):
            ETF_PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            etf_panel.to_parquet(ETF_PANEL_FILE, index=False)
        etf_stats = {"n_codes": int(etf_panel["code"].nunique()) if len(etf_panel) else 0,
                     "n_rows": int(len(etf_panel))}
    except Exception as exc:
        print(f"[fetcher] ETF 更新失败（不影响股票面板）: {exc}", file=sys.stderr)
        etf_stats = {"n_codes": 0, "n_rows": 0}

    # ---- 场外基金：科技相关池 + 净值 ----
    if progress:
        progress(5.2, 6, "更新场外基金池...")
    try:
        fund = fetch_fund_universe()
        save_csv(fund, FUND_FILE)
        fund_existing = None
        if FUND_NAV_FILE.exists():
            fund_existing = pd.read_parquet(FUND_NAV_FILE)
            fund_existing["code"] = fund_existing["code"].astype(str).str.zfill(6)
        fund_panel = fetch_fund_navs(
            sorted(fund["code"]),
            start=start,
            end=end,
            existing=fund_existing,
            max_workers=max_workers,
            progress=lambda d, t, c: progress(5.3 + d / max(t, 1) * 0.6, 6,
                                               f"基金净值 {c} ({d}/{t})")
            if progress else None,
        )
        if len(fund_panel):
            FUND_NAV_FILE.parent.mkdir(parents=True, exist_ok=True)
            fund_panel.to_parquet(FUND_NAV_FILE, index=False)
        fund_stats = {"n_codes": int(fund_panel["code"].nunique()) if len(fund_panel) else 0,
                      "n_rows": int(len(fund_panel))}
    except Exception as exc:
        print(f"[fetcher] 场外基金更新失败（不影响股票面板）: {exc}", file=sys.stderr)
        fund_stats = {"n_codes": 0, "n_rows": 0}
    if progress:
        progress(6, 6, "完成")

    if len(panel):
        # meta 记录面板真实覆盖范围（实际最后交易日），而不是请求的日历日；
        # 周末/节假日请求日（如 2026-08-16）不会产生行情，写请求日会误导展示。
        save_meta({"mode": mode,
                   "start": str(panel["date"].min().date()),
                   "end": str(panel["date"].max().date()),
                   "n_codes": int(panel["code"].nunique()),
                   "n_rows": int(len(panel))})
    return {"n_codes": int(panel["code"].nunique()) if len(panel) else 0,
            "n_rows": int(len(panel)),
            "etf": etf_stats, "fund": fund_stats}
