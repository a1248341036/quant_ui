"""ETF / 场外基金抓取。"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd

from ..store import (ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE)

from .kline import (
    sina_symbol, _fetch_kline, _compact_panel,
)


FUND_TECH_KEYWORDS = (
    "科技", "半导体", "芯片", "电子", "计算机", "通信", "传媒", "人工智能",
    "数字", "软件", "信息", "互联网", "5G", "智能", "科创", "创新",
)


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
    panel["price_basis"] = "qfq"
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    from .kline import _add_rolling_factors
    panel = _add_rolling_factors(panel)
    return _compact_panel(panel)


def fetch_fund_universe(keywords: tuple[str, ...] | None = None) -> pd.DataFrame:
    """全市场场外基金池：天天基金全量列表，保留权益类，剔除债券/货币/理财/ETF。

    keywords 为 None 时拉全量权益类基金；传入关键词时按简称过滤（向后兼容）。
    """
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
        # 剔除非权益类：债券/货币/理财/FOF/固收/商品/债
        drop_type = out["type"].str.contains(
            "债券|货币|理财|FOF|固收|商品|债", regex=True, na=False)
        # 只保留权益类（股票型/混合偏股/混合灵活/混合平衡/指数型/QDII 股票）
        keep_type = out["type"].str.contains(
            "股票|混合|指数|QDII", regex=True, na=False)
        # 剔除场内 ETF（东财基金列表会混入 ETF，代码与场外重复/净值口径不同）
        keep_otc = ~out["name"].str.contains("ETF", na=False)
        mask = keep_type & ~drop_type & keep_otc
        if keywords:
            mask = mask & out["name"].str.contains(
                "|".join(keywords), regex=True, na=False)
        out = out[mask].drop_duplicates("code")
        return out.reset_index(drop=True)
    except Exception:
        if FUND_FILE.exists():
            df = pd.read_csv(FUND_FILE, dtype={"code": str})
            df["code"] = df["code"].astype(str).str.zfill(6)
            return df
        raise RuntimeError("无法获取场外基金列表且本地无缓存")


def _parse_open_fund_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """把 fund_open_fund_daily_em 宽表解析成长表 [date, code, nav]。

    净值列形如 '2026-08-25-单位净值'，接口通常携带最近两个交易日的全市场
    快照；日历列缺失或净值空/'--' 的行丢弃。
    """
    empty = pd.DataFrame(columns=["date", "code", "nav"])
    code_col = next((c for c in raw.columns if "代码" in str(c)), None)
    if code_col is None:
        return empty
    nav_cols = [c for c in raw.columns
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}-单位净值", str(c))]
    frames = []
    for col in nav_cols:
        day = pd.Timestamp(str(col)[:10])
        part = raw[[code_col, col]].copy()
        part.columns = ["code", "nav"]
        part["code"] = part["code"].astype(str).str.zfill(6)
        part["nav"] = pd.to_numeric(part["nav"], errors="coerce")
        part["date"] = day
        frames.append(part.dropna(subset=["nav"])[["date", "code", "nav"]])
    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True)


def fetch_fund_navs(
    codes: list[str],
    start: str,
    end: str,
    existing: pd.DataFrame | None,
    max_workers: int = 6,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """场外基金历史单位净值。

    优先走 fund_open_fund_daily_em 市场快照：一次请求携带最近两个交易日
    的全市场净值（约 9 秒），日常增量不再逐只拉全量历史。快照之后仍落后
    >3 天的少数基金（新发/暂停披露/快照未收录）才回落到逐只全量接口。
    """
    end_ts = pd.Timestamp(end)

    def _last_map(df: pd.DataFrame | None) -> dict:
        if df is None or not len(df):
            return {}
        return df.groupby("code", observed=True)["date"].max().to_dict()

    base = existing
    try:
        import akshare as ak

        snap = _parse_open_fund_daily(ak.fund_open_fund_daily_em())
        snap = snap[snap["code"].isin(set(codes))]
        snap = snap[(snap["date"] >= pd.Timestamp(start)) & (snap["date"] <= end_ts)]
        print(f"基金净值快照: {len(snap)} 行 "
              f"({snap['date'].min().date()} ~ {snap['date'].max().date()})"
              if len(snap) else "基金净值快照: 0 行", flush=True)
        if len(snap):
            if base is not None and len(base):
                base = pd.concat([base, snap], ignore_index=True).drop_duplicates(
                    ["code", "date"], keep="last")
            else:
                base = snap
    except Exception as exc:
        print(f"[fetcher] 基金净值快照失败，整批回退逐只全量: {exc}", file=sys.stderr)

    last_by_code = _last_map(base)
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

    if not frames and base is not None and len(base):
        return base
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "code", "nav"])
    panel = new
    if base is not None and len(base):
        panel = pd.concat([base, new], ignore_index=True)
        panel = panel.drop_duplicates(["code", "date"], keep="last")
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    panel["code"] = panel["code"].astype("category")
    return panel
