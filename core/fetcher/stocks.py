"""股票池/行业/指数/日线抓取。"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd
import requests

from ..store import (LEGACY_DATA_DIR, TECH_FILE, UNIVERSE_FILE)
from .. import tushare_client

from .kline import (
    sina_symbol, _fetch_kline, _fetch_kline_any, _fetch_index_tencent,
    _compact_panel, _add_rolling_factors,
)

INDEX_SYMBOLS = {
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sh000852": "中证1000",
    "sz399006": "创业板指",
    "sh000688": "科创50",
    "sh000001": "上证指数",
}


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
        # 尾部缺少量交易日的代码合并成"按交易日批量"请求，避免逐股上千次调用
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
