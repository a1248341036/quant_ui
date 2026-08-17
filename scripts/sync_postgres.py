#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Tushare / 本地面板数据同步到 PostgreSQL/TimescaleDB。

示例:
    python scripts/sync_postgres.py --init
    python scripts/sync_postgres.py --basic
    python scripts/sync_postgres.py --daily-from-panel
    python scripts/sync_postgres.py --daily-since 20260801
    python scripts/sync_postgres.py --report-rc
    python scripts/sync_postgres.py --events --limit 50
    python scripts/sync_postgres.py --fina --limit 50
    python scripts/sync_postgres.py --minutes --days 1 --codes 000001.SZ,600519.SH
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pg import get_conn, create_schema, query_df, df_to_pg
from core.store import PANEL_FILE
from core.tushare_client import get_pro, is_configured, to_ts_code, trade_dates, TUSHARE_SLEEP

# COPY 分批大小：避免几百万行一次性构造 CSV 字符串导致内存峰值过高
COPY_CHUNK_ROWS = 200_000

# 腾讯不复权日线只更新这些列；adj_factor / pe / pb 等字段一律不碰
DAILY_UPDATE_COLS = ["open", "high", "low", "close", "vol", "amount", "turnover_rate"]
TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
TX_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _log(msg: str) -> None:
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


def _norm_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def _norm_num(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    for c in df.columns:
        if c in exclude or c in ("ts_code", "index_code", "con_code", "ts") or "date" in c.lower():
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def upsert_df(df: pd.DataFrame, table: str, pk: list[str], conflict_columns: list[str] | None = None) -> int:
    """用临时表 + COPY + ON CONFLICT 批量 upsert，返回影响行数。"""
    if df is None or df.empty:
        return 0
    cols = list(df.columns)
    pk_sql = ", ".join(pk)
    upd_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in set(pk))
    if conflict_columns:
        conflict_sql = f"({', '.join(conflict_columns)})"
    else:
        conflict_sql = f"({pk_sql})"
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f'CREATE TEMP TABLE _tmp_{table} (LIKE "{table}" EXCLUDING CONSTRAINTS)')
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = ANY (current_schemas(true)) AND table_name = %s AND is_nullable = 'NO'",
            (f"_tmp_{table}",),
        )
        for (col,) in cur.fetchall():
            cur.execute(f'ALTER TABLE _tmp_{table} ALTER COLUMN "{col}" DROP NOT NULL')
        for i in range(0, len(df), COPY_CHUNK_ROWS):
            buf = io.StringIO()
            df.iloc[i:i + COPY_CHUNK_ROWS].to_csv(buf, index=False, header=False, na_rep="")
            buf.seek(0)
            with cur.copy(
                f"COPY _tmp_{table} ({cols_sql}) FROM STDIN WITH (FORMAT CSV, NULL '')"
            ) as copy:
                copy.write(buf.getvalue())
        cur.execute(
            f'INSERT INTO "{table}" ({cols_sql}) SELECT {cols_sql} FROM _tmp_{table} '
            f'ON CONFLICT {conflict_sql} DO UPDATE SET {upd_sql}'
        )
        n = cur.rowcount
    return n


def upsert_adj_factor(df: pd.DataFrame, table: str = "stock_daily") -> int:
    """只更新 adj_factor 的轻量 upsert，不触碰 OHLCV 等已有字段。"""
    df = df[["ts_code", "trade_date", "adj_factor"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce").dt.date
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE _tmp_adj_factor (ts_code VARCHAR(12), trade_date DATE, adj_factor DOUBLE PRECISION)"
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        with cur.copy("COPY _tmp_adj_factor FROM STDIN WITH (FORMAT CSV, NULL '')") as copy:
            copy.write(buf.getvalue())
        cur.execute(
            f'INSERT INTO "{table}" (ts_code, trade_date, adj_factor) '
            "SELECT ts_code, trade_date, adj_factor FROM _tmp_adj_factor "
            f"ON CONFLICT (ts_code, trade_date) DO UPDATE SET adj_factor = EXCLUDED.adj_factor"
        )
        n = cur.rowcount
    return n


def _fetch_all(pro, api: str, retries: int = 4, **kwargs) -> pd.DataFrame:
    """带 sleep 和指数退避重试的代理请求（覆盖 SSL 抖动 / 429 限流）。"""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        time.sleep(max(0.0, TUSHARE_SLEEP))
        try:
            return getattr(pro, api)(**kwargs)
        except Exception as exc:
            last_err = exc
            if attempt == retries:
                break
            delay = min(30.0, 1.5 * (2 ** (attempt - 1)) + random.uniform(0, 1.0))
            _log(f"{api} 失败（{exc.__class__.__name__}: {exc}），{delay:.1f}s 后第 {attempt + 1}/{retries} 次重试")
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def cmd_init() -> None:
    create_schema()
    _log("schema 就绪")


def cmd_basic() -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    df = _fetch_all(pro, "stock_basic", exchange="", list_status="L",
                    fields="ts_code,symbol,name,area,industry,market,list_date")
    df = _norm_date(df, ["list_date"])
    n = upsert_df(df, "stock_basic", ["ts_code"])
    _log(f"stock_basic: {n} 行")

    cal = _fetch_all(pro, "trade_cal", exchange="SSE", start_date="20150101", end_date="20301231")
    cal = _norm_date(cal, ["cal_date", "pretrade_date"])
    n = upsert_df(cal, "trade_cal", ["exchange", "cal_date"])
    _log(f"trade_cal: {n} 行")


def _panel_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"code": "ts_code", "date": "trade_date", "volume": "vol",
                             "turnover": "turnover_rate", "amount": "amount"})
    out["ts_code"] = out["ts_code"].map(to_ts_code)
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    # panel 换手率是比例（0.0043），统一为 Tushare daily_basic 的 % 口径（0.4289）
    out["turnover_rate"] = out["turnover_rate"] * 100.0
    keep = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "turnover_rate", "amount"]
    out = out[[c for c in keep if c in out.columns]].copy()
    return out


def cmd_daily_from_panel() -> None:
    """面板日线 -> stock_daily（只填空行，不覆盖已有原始价）。

    panel 是前复权价，PG stock_daily 的 OHLCV 必须是 Tushare 不复权原始价
    （与 adj_factor 配套）。这里仅用于 bootstrap：某日完全没有行情行时填上
    占位数据，后续 --daily-since / 全量回填会用原始价覆盖。
    """
    if not PANEL_FILE.exists():
        raise SystemExit(f"panel 不存在: {PANEL_FILE}")
    panel = pd.read_parquet(PANEL_FILE)
    daily = _panel_to_daily(panel)
    if daily.empty:
        _log("panel -> stock_daily: 空面板，跳过")
        return
    n = _upsert_daily_fill_missing(daily)
    _log(f"panel -> stock_daily: {n} 行（仅填空行，不覆盖已有原始价）")


def _upsert_daily_fill_missing(df: pd.DataFrame, table: str = "stock_daily") -> int:
    """COPY + ON CONFLICT，只填充 open/close 为空的行情行，不覆盖既有 OHLCV。"""
    cols = ["ts_code", "trade_date"] + DAILY_UPDATE_COLS
    df = df[cols].copy()
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    upd_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in DAILY_UPDATE_COLS)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f'CREATE TEMP TABLE _tmp_daily (LIKE "{table}" EXCLUDING CONSTRAINTS)')
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = ANY (current_schemas(true)) AND table_name = %s AND is_nullable = 'NO'",
            (f"_tmp_{table}",),
        )
        for (col,) in cur.fetchall():
            cur.execute(f'ALTER TABLE _tmp_{table} ALTER COLUMN "{col}" DROP NOT NULL')
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        with cur.copy(f"COPY _tmp_{table} ({cols_sql}) FROM STDIN WITH (FORMAT CSV, NULL '')") as copy:
            copy.write(buf.getvalue())
        cur.execute(
            f'INSERT INTO "{table}" ({cols_sql}) SELECT {cols_sql} FROM _tmp_{table} '
            f'ON CONFLICT (ts_code, trade_date) DO UPDATE SET {upd_sql} '
            "WHERE stock_daily.open IS NULL OR stock_daily.close IS NULL"
        )
        n = cur.rowcount
    return n


def _fetch_one_trade_date(pro, d: str) -> tuple[str, int]:
    """拉取单个交易日 daily/daily_basic/adj_factor 并写库，返回 (日期, 行数)。"""
    daily = _fetch_all(pro, "daily", trade_date=d)
    basic = _fetch_all(pro, "daily_basic", trade_date=d)
    adj = _fetch_all(pro, "adj_factor", trade_date=d)
    basic = basic.drop(columns=["close"], errors="ignore")
    merged = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
    merged = merged.merge(adj, on=["ts_code", "trade_date"], how="left")
    merged = _norm_date(merged, ["trade_date"])
    merged = _norm_num(merged)
    n = upsert_df(merged, "stock_daily", ["ts_code", "trade_date"])
    _log(f"{d}: {n} 行")
    return d, n


def cmd_daily_since(since: str, workers: int = 1) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    dates = trade_dates(since, pd.Timestamp.now().strftime("%Y-%m-%d"))
    if not dates:
        raise SystemExit("无交易日")
    _log(f"拉取 {len(dates)} 个交易日: {dates[0]} ~ {dates[-1]}（并发 {workers}）")
    total = 0
    failed: list[str] = []
    if workers <= 1:
        for d in dates:
            try:
                _, n = _fetch_one_trade_date(pro, d)
                total += n
            except Exception as exc:
                failed.append(d)
                _log(f"{d} 失败: {exc.__class__.__name__}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one_trade_date, pro, d): d for d in dates}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    _, n = fut.result()
                    total += n
                except Exception as exc:
                    failed.append(d)
                    _log(f"{d} 失败: {exc.__class__.__name__}: {exc}")
    if failed:
        _log(f"失败 {len(failed)} 个交易日: {failed[0]} ~ {failed[-1]}（后续可重跑补齐）")
    _log(f"daily 合计 {total} 行")


def cmd_adj_factor_since(since: str) -> None:
    """只回填复权因子：每个交易日 1 次 adj_factor 请求，内存占用极小。"""
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    dates = trade_dates(since, pd.Timestamp.now().strftime("%Y-%m-%d"))
    if not dates:
        raise SystemExit("无交易日")
    _log(f"回填复权因子 {len(dates)} 个交易日: {dates[0]} ~ {dates[-1]}")
    total = 0
    for d in dates:
        adj = _fetch_all(pro, "adj_factor", trade_date=d)
        if adj is None or adj.empty:
            _log(f"{d}: 无数据")
            continue
        n = upsert_adj_factor(adj)
        total += n
        _log(f"{d}: {n} 行")
    _log(f"复权因子合计 {total} 行")


def cmd_report_rc() -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    df = _fetch_all(pro, "report_rc")
    if df is None or df.empty:
        _log("report_rc 无数据")
        return
    df = _norm_date(df, ["report_date"])
    df = _norm_num(df)
    df = df.drop_duplicates(subset=["ts_code", "report_date", "org_name", "report_title"])
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM report_rc")
    df_to_pg(df, "report_rc", if_exists="append")
    n = len(df)
    _log(f"report_rc: {n} 行（{df['report_date'].min()} ~ {df['report_date'].max()}）")


def _sync_events_one(pro, code: str) -> dict[str, int]:
    counts = {"dividend": 0, "share_float": 0, "namechange": 0}
    for api, table in (
        ("dividend", "dividend"),
        ("share_float", "share_float"),
        ("namechange", "namechange"),
    ):
        try:
            df = _fetch_all(pro, api, ts_code=code)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        date_cols = [c for c in df.columns if "date" in c]
        df = _norm_date(df, date_cols)
        df = _norm_num(df)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(f'DELETE FROM "{table}" WHERE ts_code = %s', (code,))
        df_to_pg(df, table, if_exists="append")
        counts[table] = len(df)
    return counts


def cmd_events(limit: int, codes_file: str | None, workers: int = 3) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    codes = _pick_codes(pro, limit, codes_file)
    _log(f"事件表处理 {len(codes)} 只（workers={workers}）")
    counts = {"dividend": 0, "share_float": 0, "namechange": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sync_events_one, pro, code): code for code in codes}
        for fut in as_completed(futs):
            try:
                sub = fut.result()
                for k in counts:
                    counts[k] += sub.get(k, 0)
            except Exception:
                pass
            done += 1
            if done % 100 == 0 or done == len(codes):
                _log(f"进度 {done}/{len(codes)}")
    _log(f"完成: {counts}")


def _sync_fina_one(pro, code: str) -> dict[str, int]:
    counts = {api: 0 for api in ("fina_indicator", "income", "balancesheet", "cashflow")}
    for api in counts:
        try:
            df = _fetch_all(pro, api, ts_code=code)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        date_cols = [c for c in df.columns if c in ("ann_date", "end_date", "f_ann_date")]
        df = _norm_date(df, date_cols)
        df = _norm_num(df)
        pk = ["ts_code", "end_date", "report_type"] if api != "fina_indicator" else ["ts_code", "end_date"]
        df = df.drop_duplicates(subset=pk, keep="last")
        counts[api] = upsert_df(df, api, pk)
    return counts


def cmd_fina(limit: int, codes_file: str | None, workers: int = 3) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    codes = _pick_codes(pro, limit, codes_file)
    _log(f"财务表处理 {len(codes)} 只（workers={workers}）")
    for api in ("fina_indicator", "income", "balancesheet", "cashflow"):
        _ensure_fina_table(pro, api)
    counts = {api: 0 for api in ("fina_indicator", "income", "balancesheet", "cashflow")}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sync_fina_one, pro, code): code for code in codes}
        for fut in as_completed(futs):
            try:
                sub = fut.result()
                for k in counts:
                    counts[k] += sub.get(k, 0)
            except Exception:
                pass
            done += 1
            if done % 50 == 0 or done == len(codes):
                _log(f"进度 {done}/{len(codes)}")
    _log(f"完成: {counts}")


def _ensure_fina_table(pro, api: str) -> None:
    """按接口返回列动态建财务宽表（缺失才建）。"""
    existing = {r[0] for r in query_df("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")["table_name"]}
    if api in existing:
        return
    df = _fetch_all(pro, api, ts_code="000001.SZ")
    if df is None or df.empty:
        return
    _date_cols = {"ann_date", "end_date", "f_ann_date"}
    cols = []
    for c in df.columns:
        if c == "ts_code":
            continue
        if c in _date_cols:
            t = "DATE"
        else:
            t = "DOUBLE PRECISION"
        cols.append(f'"{c}" {t}')
    pk_sql = (
        "PRIMARY KEY (ts_code, end_date, report_type)"
        if api != "fina_indicator"
        else "PRIMARY KEY (ts_code, end_date)"
    )
    create = f'CREATE TABLE IF NOT EXISTS "{api}" (ts_code VARCHAR(12) NOT NULL, ' + ", ".join(cols) + f", {pk_sql})"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(create)
        cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{api}_code ON "{api}" (ts_code)')
    _log(f"建表 {api}")


def _pick_codes(pro, limit: int, codes_file: str | None) -> list[str]:
    if codes_file:
        return [line.strip() for line in Path(codes_file).read_text().splitlines() if line.strip()]
    df = query_df("SELECT ts_code FROM stock_basic ORDER BY ts_code")
    codes = df["ts_code"].tolist()
    if limit and limit > 0:
        codes = codes[:limit]
    return codes


def _sync_surv_one(pro, code: str) -> dict[str, int]:
    counts = {"forecast": 0, "express": 0, "stk_surv": 0}
    for api, table in (
        ("forecast", "forecast"),
        ("express", "express"),
        ("stk_surv", "stk_surv"),
    ):
        try:
            df = _fetch_all(pro, api, ts_code=code)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        date_cols = [c for c in df.columns if "date" in c]
        df = _norm_date(df, date_cols)
        df = _norm_num(df)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(f'DELETE FROM "{table}" WHERE ts_code = %s', (code,))
        df_to_pg(df, table, if_exists="append")
        counts[table] = len(df)
    return counts


def cmd_surv(limit: int, codes_file: str | None, workers: int = 3) -> None:
    """业绩预告 / 业绩快报 / 机构调研（Tushare 可用，非新闻舆情）。"""
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    codes = _pick_codes(pro, limit, codes_file)
    _log(f"事件表处理 {len(codes)} 只（forecast/express/stk_surv, workers={workers}）")
    counts = {"forecast": 0, "express": 0, "stk_surv": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sync_surv_one, pro, code): code for code in codes}
        for fut in as_completed(futs):
            try:
                sub = fut.result()
                for k in counts:
                    counts[k] += sub.get(k, 0)
            except Exception:
                pass
            done += 1
            if done % 100 == 0 or done == len(codes):
                _log(f"进度 {done}/{len(codes)}")
    _log(f"完成: {counts}")


def _tx_symbol(ts_code: str) -> str:
    """ts_code -> 腾讯 symbol（sh/sz/bj + 6 位）。"""
    code6 = ts_code[:6]
    if ts_code.endswith(".SH"):
        return "sh" + code6
    if ts_code.endswith(".SZ"):
        return "sz" + code6
    if ts_code.endswith(".BJ"):
        return "bj" + code6
    return code6


def _tx_windows(start: str, end: str, max_years: int = 2) -> list[tuple[str, str]]:
    """腾讯一次最多 640 根 K 线，按 2 年窗口分段。"""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    cur = s
    while cur <= e:
        stop = min(e, cur + pd.DateOffset(years=max_years) - pd.DateOffset(days=1))
        out.append((cur.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        cur = stop + pd.DateOffset(days=1)
    return out


def _fetch_kline_tencent(symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """腾讯单段不复权日线（day），返回 date/open/close/high/low/volume/turnover/amount。"""
    params = {"_var": "kline_dayqfq", "param": f"{symbol},day,{start},{end},640,"}
    for attempt in range(retries):
        try:
            resp = requests.get(TX_URL, params=params, headers=TX_HEADERS, timeout=10)
            body = resp.text[resp.text.find("=") + 1:]
            data = json.loads(body)
            d = data["data"].get(symbol, {})
            key = next((k for k in ("day", "qfqday", "hfqday") if k in d), None)
            if key is None:
                return pd.DataFrame()
            rows = d.get(key) or []
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            # 腾讯 day 字段：0=date 1=open 2=close 3=high 4=low 5=volume 7=turnover(%) 8=amount(万元)
            df = df.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8]].copy()
            df.columns = ["date", "open", "close", "high", "low", "volume", "turnover", "amount"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            for c in ["open", "close", "high", "low", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 10000.0
            return df.sort_values("date").drop_duplicates("date", keep="last")
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()


def _fetch_stock_history_tencent(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """拉单只股票全历史不复权日线，转成 stock_daily 列。"""
    symbol = _tx_symbol(ts_code)
    frames = []
    for ws, we in _tx_windows(start, end):
        df = _fetch_kline_tencent(symbol, ws, we)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last")
    out["ts_code"] = ts_code
    out["trade_date"] = out["date"].dt.date
    out = out.rename(columns={"volume": "vol", "turnover": "turnover_rate"})
    return out[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "turnover_rate"]]


def upsert_daily_tencent(df: pd.DataFrame, table: str = "stock_daily") -> int:
    """COPY + ON CONFLICT 只补 OHLCV；已有行情行不覆盖，空壳行（仅 adj_factor）补齐。"""
    if df is None or df.empty:
        return 0
    cols = ["ts_code", "trade_date"] + DAILY_UPDATE_COLS
    df = df[cols].copy()
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    upd_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in DAILY_UPDATE_COLS)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f'CREATE TEMP TABLE _tmp_daily (LIKE "{table}" EXCLUDING CONSTRAINTS)')
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = ANY (current_schemas(true)) AND table_name = %s AND is_nullable = 'NO'",
            ("_tmp_daily",),
        )
        for (col,) in cur.fetchall():
            cur.execute(f'ALTER TABLE _tmp_daily ALTER COLUMN "{col}" DROP NOT NULL')
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        with cur.copy(f"COPY _tmp_daily ({cols_sql}) FROM STDIN WITH (FORMAT CSV, NULL '')") as copy:
            copy.write(buf.getvalue())
        cur.execute(
            f'INSERT INTO "{table}" ({cols_sql}) SELECT {cols_sql} FROM _tmp_daily '
            f'ON CONFLICT (ts_code, trade_date) DO UPDATE SET {upd_sql} '
            "WHERE stock_daily.open IS NULL OR stock_daily.close IS NULL"
        )
        n = cur.rowcount
    return n


def cmd_daily_tencent(workers: int = 3, max_codes: int = 0, batch_size: int = 50) -> None:
    """腾讯不复权日线补全全市场历史（2015-01-01 起），不依赖 Tushare。"""
    df = query_df("SELECT ts_code FROM stock_basic ORDER BY ts_code")
    codes = df["ts_code"].tolist()
    if max_codes and max_codes > 0:
        codes = codes[:max_codes]
    start = "2015-01-01"
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    _log(f"腾讯不复权日线补全 {len(codes)} 只（workers={workers}, batch={batch_size}）: {start} ~ {end}")
    total = 0
    done = 0
    failed: list[str] = []
    buf: list[pd.DataFrame] = []

    def _one(code: str) -> tuple[str, pd.DataFrame]:
        try:
            return code, _fetch_stock_history_tencent(code, start, end)
        except Exception:
            return code, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, c): c for c in codes}
        for fut in as_completed(futs):
            code, sub = fut.result()
            done += 1
            if sub is None or sub.empty:
                failed.append(code)
            else:
                buf.append(sub)
                total += len(sub)
            if len(buf) >= batch_size:
                n = upsert_daily_tencent(pd.concat(buf, ignore_index=True))
                buf = []
                _log(f"写入 {n} 行（进度 {done}/{len(codes)}，累计 {total} 行）")
            elif done % 200 == 0:
                _log(f"进度 {done}/{len(codes)}，累计 {total} 行")
    if buf:
        n = upsert_daily_tencent(pd.concat(buf, ignore_index=True))
        _log(f"收尾写入 {n} 行")
    _log(f"腾讯日线完成: {total} 行，失败 {len(failed)} 只: {','.join(failed[:10])}")


def cmd_index_weight(trade_date: str | None = None) -> None:
    """拉宽基指数权重（默认最近交易日）。"""
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    if not trade_date:
        td = query_df("SELECT max(trade_date)::text d FROM stock_daily").iloc[0]["d"]
        trade_date = td.replace("-", "")
    frames = []
    for idx in ("000300.SH", "000905.SH", "000852.SH"):
        got = False
        for idx_arg in (idx, idx.replace(".SH", ".SZ")):
            try:
                sub = _fetch_all(pro, "index_weight", index_code=idx_arg, trade_date=trade_date)
            except Exception:
                continue
            if sub is None or sub.empty:
                continue
            sub = sub[["index_code", "con_code", "trade_date", "weight"]].copy()
            sub["trade_date"] = pd.to_datetime(sub["trade_date"], format="%Y%m%d", errors="coerce").dt.date
            sub["weight"] = pd.to_numeric(sub["weight"], errors="coerce")
            frames.append(sub)
            got = True
            break
        _log(f"{idx}: {'ok' if got else '无数据'}")
    if not frames:
        raise SystemExit("index_weight 无数据")
    out = pd.concat(frames, ignore_index=True)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM index_weight WHERE trade_date = %s", (out["trade_date"].iloc[0],))
    n = upsert_df(out, "index_weight", ["index_code", "con_code", "trade_date"])
    _log(f"index_weight {trade_date}: {n} 行")


def cmd_minutes(days: int, codes: list[str], freq: str) -> None:
    """akshare 分钟样例 -> stock_minute（Tushare stk_mins 未开通前验证链路）。"""
    import akshare as ak
    end = pd.Timestamp.now()
    start = (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    total = 0
    for code in codes:
        df = None
        src = ""
        try:
            df = ak.stock_zh_a_hist_min_em(symbol=code, period=freq, adjust="")
            src = "em"
            df = df.rename(columns={"时间": "ts", "开盘": "open", "最高": "high", "最低": "low",
                                    "收盘": "close", "成交量": "volume", "成交额": "amount"})
        except Exception:
            try:
                sym = ("sh" if to_ts_code(code).endswith(".SH") else "sz") + code
                df = ak.stock_zh_a_minute(symbol=sym, period=freq, adjust="")
                src = "tencent"
                df = df.rename(columns={"day": "ts"})
            except Exception as exc:
                _log(f"{code}: 分钟拉取失败 {exc}")
                continue
        if df is None or df.empty:
            continue
        df["ts"] = pd.to_datetime(df["ts"])
        df = df[df["ts"] >= start]
        df["ts_code"] = to_ts_code(code)
        df["freq"] = "1min" if freq == "1" else freq
        df = _norm_num(df, exclude={"ts_code", "freq"})
        n = upsert_df(df, "stock_minute", ["ts_code", "ts", "freq"])
        total += n
        _log(f"{code}: {n} 行（源={src}）")
    _log(f"分钟合计 {total} 行")


def main() -> int:
    parser = argparse.ArgumentParser(description="同步 Tushare 数据到 PostgreSQL/TimescaleDB")
    parser.add_argument("--init", action="store_true", help="建 schema")
    parser.add_argument("--basic", action="store_true", help="股票列表 + 交易日历")
    parser.add_argument("--daily-from-panel", action="store_true", help="本地 panel.parquet -> stock_daily")
    parser.add_argument("--daily-since", metavar="YYYYMMDD", help="按交易日增量拉 daily/daily_basic/adj_factor")
    parser.add_argument("--daily-workers", type=int, default=1, help="日线按交易日并发数（代理限流建议 3，别超过 4）")
    parser.add_argument("--daily-tencent", action="store_true", help="腾讯不复权日线补全全市场历史")
    parser.add_argument("--adj-factor-since", metavar="YYYYMMDD", help="只回填复权因子（每天 1 次请求）")
    parser.add_argument("--index-weight", action="store_true", help="宽基指数权重（默认最近交易日）")
    parser.add_argument("--sleep", type=float, default=None, help="Tushare 请求间隔秒数，覆盖 TUSHARE_SLEEP")
    parser.add_argument("--report-rc", action="store_true", help="一致预期")
    parser.add_argument("--events", action="store_true", help="dividend/share_float/namechange")
    parser.add_argument("--fina", action="store_true", help="财务宽表 fina_indicator/income/...")
    parser.add_argument("--surv", action="store_true", help="业绩预告/快报/机构调研")
    parser.add_argument("--minutes", action="store_true", help="akshare 分钟样例")
    parser.add_argument("--days", type=int, default=2, help="分钟样例回看天数")
    parser.add_argument("--codes", default="000001,600519", help="分钟样例股票（逗号分隔，6 位）")
    parser.add_argument("--freq", default="1", choices=["1", "5", "15", "30", "60"], help="分钟周期")
    parser.add_argument("--limit", type=int, default=0, help="事件/财务全市场时只处理前 N 只")
    parser.add_argument("--max-codes", type=int, default=0, help="腾讯日线只处理前 N 只")
    parser.add_argument("--batch", type=int, default=50, help="腾讯日线写入批次（只数）")
    parser.add_argument("--codes-file", help="事件/财务处理指定股票列表文件")
    parser.add_argument("--workers", type=int, default=3, help="事件/财务并发请求数（代理建议 3，别超过 4）")
    args = parser.parse_args()

    if args.sleep is not None:
        global TUSHARE_SLEEP
        TUSHARE_SLEEP = max(0.0, args.sleep)

    if not any([args.init, args.basic, args.daily_from_panel, args.daily_since, args.adj_factor_since,
                args.daily_tencent, args.index_weight, args.report_rc, args.events, args.fina, args.surv,
                args.minutes]):
        parser.print_help()
        return 1

    if args.init:
        cmd_init()
    if args.basic:
        cmd_basic()
    if args.daily_from_panel:
        cmd_daily_from_panel()
    if args.daily_since:
        cmd_daily_since(args.daily_since, args.daily_workers)
    if args.daily_tencent:
        cmd_daily_tencent(args.workers, args.max_codes, args.batch)
    if args.adj_factor_since:
        cmd_adj_factor_since(args.adj_factor_since)
    if args.index_weight:
        cmd_index_weight()
    if args.report_rc:
        cmd_report_rc()
    if args.events:
        cmd_events(args.limit, args.codes_file, args.workers)
    if args.fina:
        cmd_fina(args.limit, args.codes_file, args.workers)
    if args.surv:
        cmd_surv(args.limit, args.codes_file, args.workers)
    if args.minutes:
        cmd_minutes(args.days, [c.strip() for c in args.codes.split(",")], args.freq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
