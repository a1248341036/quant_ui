#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Tushare 数据直接增量写入 data/pg_parquet/*.parquet（不再经过 PostgreSQL）。

用法:
    python scripts/sync_tushare_to_parquet.py --basic
    python scripts/sync_tushare_to_parquet.py --daily-since 20260801 [--daily-workers 3] [--force-daily]
    python scripts/sync_tushare_to_parquet.py --last-adj
    python scripts/sync_tushare_to_parquet.py --report-rc
    python scripts/sync_tushare_to_parquet.py --events --limit 50
    python scripts/sync_tushare_to_parquet.py --fina --limit 50
    python scripts/sync_tushare_to_parquet.py --surv --limit 50

内存约束（3.6G 小机）:
    - 日线按交易日拉取，单日全市场约 5500 行，每天只合并新增尾部，
      DuckDB 流式 COPY，不整表进 pandas。
    - 财务/事件表按股票逐只拉取，每只最多几百行；合并时用 DuckDB
      ANTI JOIN 替换该股票旧行，避免逐行 upsert。
    - stock_daily.parquet（约 1200 万行）永远不整表载入 pandas。
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tushare_client import (TUSHARE_SLEEP, get_pro, is_configured,
                                 trade_dates)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "data" / "pg_parquet"

# 与导出脚本一致的显式 schema：避免全 NULL 列/对象类型导致 schema 漂移。
STOCK_DAILY_SCHEMA = pa.schema([
    pa.field("ts_code", pa.string()),
    pa.field("trade_date", pa.date32()),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("pre_close", pa.float64()),
    pa.field("change", pa.float64()),
    pa.field("pct_chg", pa.float64()),
    pa.field("vol", pa.float64()),
    pa.field("amount", pa.float64()),
    pa.field("turnover_rate", pa.float64()),
    pa.field("turnover_rate_f", pa.float64()),
    pa.field("volume_ratio", pa.float64()),
    pa.field("pe", pa.float64()),
    pa.field("pe_ttm", pa.float64()),
    pa.field("pb", pa.float64()),
    pa.field("ps", pa.float64()),
    pa.field("ps_ttm", pa.float64()),
    pa.field("dv_ratio", pa.float64()),
    pa.field("dv_ttm", pa.float64()),
    pa.field("total_share", pa.float64()),
    pa.field("float_share", pa.float64()),
    pa.field("free_share", pa.float64()),
    pa.field("total_mv", pa.float64()),
    pa.field("circ_mv", pa.float64()),
    pa.field("adj_factor", pa.float64()),
])

STOCK_BASIC_SCHEMA = pa.schema([
    pa.field("ts_code", pa.string()),
    pa.field("symbol", pa.string()),
    pa.field("name", pa.string()),
    pa.field("area", pa.string()),
    pa.field("industry", pa.string()),
    pa.field("market", pa.string()),
    pa.field("list_date", pa.date32()),
    pa.field("updated_at", pa.timestamp("us", tz="UTC")),
])

TRADE_CAL_SCHEMA = pa.schema([
    pa.field("exchange", pa.string()),
    pa.field("cal_date", pa.date32()),
    pa.field("is_open", pa.int64()),
    pa.field("pretrade_date", pa.date32()),
])

LAST_ADJ_SCHEMA = pa.schema([
    pa.field("ts_code", pa.string()),
    pa.field("ref_date", pa.date32()),
    pa.field("last_adj", pa.float64()),
])

# 日线合并时忽略这些来自 daily_basic 的重复列（basic 不带 close）
_DATE_COLS = {"ann_date", "end_date", "f_ann_date", "trade_date", "cal_date",
              "pretrade_date", "list_date", "record_date", "ex_date", "pay_date",
              "div_listdate", "imp_ann_date", "float_date", "start_date",
              "surv_date", "report_date"}


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
        if c in exclude or c in ("ts_code", "index_code", "con_code", "ts", "id"):
            continue
        if c in _DATE_COLS or "date" in c.lower():
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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
            _log(f"{api} 失败（{exc.__class__.__name__}: {exc}），"
                 f"{delay:.1f}s 后第 {attempt + 1}/{retries} 次重试")
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def _schema_names(path: Path) -> list[str]:
    """读 parquet 的列名（pyarrow，不加载数据）。"""
    try:
        return pq.ParquetFile(path).schema_arrow.names
    except Exception:
        return []


def _align_df(df: pd.DataFrame, schema_names: list[str]) -> pd.DataFrame:
    """按已有 parquet 列顺序/集合对齐 DataFrame，缺失列补 NaN。"""
    if not schema_names:
        return df
    out = pd.DataFrame(index=df.index)
    for c in schema_names:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = None
    return out


def _coerce_by_schema(df: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    """按 Arrow schema 类型转换 DataFrame（避免把 name/summary 等文本列转 NaN）。"""
    out = df.copy()
    for field in schema:
        c = field.name
        if c not in out.columns:
            continue
        if pa.types.is_date(field.type):
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date
        elif pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif pa.types.is_string(field.type):
            out[c] = out[c].astype("string", errors="ignore")
    return out


def _table_from_df(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """pandas DataFrame -> Arrow Table，对象日期列转 date32。"""
    for field in schema:
        c = field.name
        if c not in df.columns:
            df[c] = None
        if pa.types.is_date(field.type):
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df[c] = df[c].dt.date
    return pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)


def _write_full(path: Path, df: pd.DataFrame, schema: pa.Schema) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _table_from_df(df.copy(), schema)
    pq.write_table(table, path, compression="zstd")
    return len(df)


def _merge_tail(path: Path, tail_path: Path, tmp_path: Path,
                on: str = "ts_code, trade_date",
                row_group_size: int = 100_000) -> int:
    """DuckDB 流式合并：旧文件去掉 tail 中的 key，再 UNION ALL tail。"""
    import duckdb

    con = duckdb.connect()
    try:
        join_sql = " USING (" + on + ")"
        con.execute(
            "COPY ("
            f"SELECT old.* FROM read_parquet('{path}') old "
            f"ANTI JOIN read_parquet('{tail_path}') tail{join_sql} "
            f"UNION ALL SELECT * FROM read_parquet('{tail_path}')"
            ") TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE ?)",
            [str(tmp_path), row_group_size],
        )
        return con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(tmp_path)]
        ).fetchone()[0]
    finally:
        con.close()


def _merge_replace_code(path: Path, df: pd.DataFrame, code: str,
                        schema: pa.Schema | None = None) -> int:
    """把单只股票整段数据替换进 parquet（ANTI JOIN ts_code）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.parquet")
    tail = path.with_name(f".{path.stem}.tail.{os.getpid()}.parquet")
    try:
        existing_schema = None
        if path.exists():
            try:
                existing_schema = pq.ParquetFile(path).schema_arrow
            except Exception:
                existing_schema = None
        if existing_schema is not None:
            df = _align_df(df, existing_schema.names)
            df = _coerce_by_schema(df, existing_schema)
            table = pa.Table.from_pandas(df, schema=existing_schema,
                                         preserve_index=False)
        elif schema is not None:
            table = _table_from_df(df.copy(), schema)
        else:
            table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tail, compression="zstd")
        if not path.exists():
            os.replace(tail, path)
            return len(df)
        _merge_tail(path, tail, tmp, on="ts_code")
        os.replace(tmp, path)
        tail.unlink(missing_ok=True)
        return len(df)
    except Exception:
        for p in (tmp, tail):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _existing_daily_dates(path: Path, since: str) -> tuple[set, set]:
    """返回 parquet 中 since 以来已存在的交易日与缺 OHLCV 的日期。"""
    if not path.exists():
        return set(), set()
    import duckdb

    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT trade_date,
                   count(*) FILTER (
                       WHERE open IS NULL OR high IS NULL OR low IS NULL
                          OR close IS NULL OR vol IS NULL OR amount IS NULL
                   ) AS incomplete
            FROM read_parquet(?)
            WHERE trade_date >= CAST(? AS DATE)
            GROUP BY trade_date
            """,
            [str(path), pd.Timestamp(since).date()],
        ).fetchall()
    finally:
        con.close()
    present = {r[0] for r in rows}
    need_repair = {r[0] for r in rows if r[1] > 0}
    return present, need_repair


def _max_trade_date(path: Path) -> date | None:
    if not path.exists():
        return None
    import duckdb

    con = duckdb.connect()
    try:
        row = con.execute(
            "SELECT max(trade_date) FROM read_parquet(?)", [str(path)]
        ).fetchone()
        return row[0] if row and row[0] is not None else None
    finally:
        con.close()


def _codes_from_stock_basic(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"stock_basic 不存在: {path}（先跑 --basic）")
    import duckdb

    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT ts_code FROM read_parquet(?) ORDER BY ts_code", [str(path)]
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def cmd_basic(out_dir: Path) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    df = _fetch_all(pro, "stock_basic", exchange="", list_status="L",
                    fields="ts_code,symbol,name,area,industry,market,list_date")
    df = _norm_date(df, ["list_date"])
    df["updated_at"] = pd.Timestamp.now(tz="UTC")
    n = _write_full(out_dir / "stock_basic.parquet", df, STOCK_BASIC_SCHEMA)
    _log(f"stock_basic: {n} 行")

    cal = _fetch_all(pro, "trade_cal", exchange="SSE",
                     start_date="20150101", end_date="20301231")
    cal = _norm_date(cal, ["cal_date", "pretrade_date"])
    n = _write_full(out_dir / "trade_cal.parquet", cal, TRADE_CAL_SCHEMA)
    _log(f"trade_cal: {n} 行")


def _fetch_one_trade_date(pro, d: str) -> pd.DataFrame:
    """拉取单个交易日 daily/daily_basic/adj_factor，返回 27 列 stock_daily 行。"""
    daily = _fetch_all(pro, "daily", trade_date=d)
    basic = _fetch_all(pro, "daily_basic", trade_date=d)
    adj = _fetch_all(pro, "adj_factor", trade_date=d)
    basic = basic.drop(columns=["close"], errors="ignore")
    merged = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
    merged = merged.merge(adj, on=["ts_code", "trade_date"], how="left")
    merged = _norm_date(merged, ["trade_date"])
    merged = _norm_num(merged)
    return merged


def cmd_daily_since(since: str, out_dir: Path, workers: int = 1,
                    force: bool = False) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    dates = trade_dates(since, pd.Timestamp.now().strftime("%Y-%m-%d"))
    if not dates:
        _log("无交易日")
        return
    path = out_dir / "stock_daily.parquet"
    if not force:
        present, need_repair = _existing_daily_dates(path, since)
        all_dates = dates
        dates = [
            d for d in all_dates
            if pd.Timestamp(d).date() not in present
            or pd.Timestamp(d).date() in need_repair
        ]
        skipped = len(all_dates) - len(dates)
        if skipped:
            _log(f"跳过 {skipped} 个已完整入库的交易日（强制重拉用 --force-daily）")
    if not dates:
        _log(f"{since} 之后无待拉取交易日（parquet 已是最新）")
        return
    _log(f"拉取 {len(dates)} 个交易日: {dates[0]} ~ {dates[-1]}（并发 {workers}）")
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    total = 0
    if workers <= 1:
        for d in dates:
            try:
                sub = _fetch_one_trade_date(pro, d)
                total += len(sub)
                if not sub.empty:
                    frames.append(sub)
                _log(f"{d}: {len(sub)} 行")
            except Exception as exc:
                failed.append(d)
                _log(f"{d} 失败: {exc.__class__.__name__}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fetch_one_trade_date, pro, d): d for d in dates}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    sub = fut.result()
                    total += len(sub)
                    if not sub.empty:
                        frames.append(sub)
                    _log(f"{d}: {len(sub)} 行")
                except Exception as exc:
                    failed.append(d)
                    _log(f"{d} 失败: {exc.__class__.__name__}: {exc}")
    if failed:
        _log(f"失败 {len(failed)} 个交易日: {failed[0]} ~ {failed[-1]}（后续可重跑补齐）")
    if frames:
        tail = path.with_name(f".{path.stem}.tail.{os.getpid()}.parquet")
        tmp = path.with_name(f".{path.stem}.merge.{os.getpid()}.parquet")
        try:
            df = pd.concat(frames, ignore_index=True)
            df = _align_df(df, _schema_names(path) or STOCK_DAILY_SCHEMA.names)
            table = _table_from_df(df, STOCK_DAILY_SCHEMA)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, tail, compression="zstd")
            if not path.exists():
                os.replace(tail, path)
                merged_rows = len(df)
            else:
                merged_rows = _merge_tail(path, tail, tmp, on="ts_code, trade_date")
                os.replace(tmp, path)
                tail.unlink(missing_ok=True)
            _log(f"stock_daily: 合并后 {merged_rows:,} 行（新增 {len(df):,} 行）-> {path}")
        except Exception:
            for p in (tail, tmp):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
    if failed:
        raise SystemExit(f"daily 拉取失败 {len(failed)} 个交易日")
    _log(f"daily 合计 {total} 行")


def cmd_last_adj(out_dir: Path) -> None:
    path = out_dir / "stock_daily.parquet"
    if not path.exists():
        raise SystemExit(f"stock_daily 不存在: {path}")
    import duckdb

    con = duckdb.connect()
    try:
        row = con.execute(
            "SELECT max(trade_date) FROM read_parquet(?)", [str(path)]
        ).fetchone()
        max_date = row[0] if row and row[0] is not None else None
        if max_date is None:
            raise SystemExit("stock_daily 无数据，无法生成复权锚点")
        rows = con.execute(
            "SELECT ts_code, trade_date AS ref_date, adj_factor AS last_adj "
            "FROM read_parquet(?) WHERE trade_date = ? AND adj_factor IS NOT NULL",
            [str(path), max_date],
        ).fetchall()
    finally:
        con.close()
    df = pd.DataFrame(rows, columns=["ts_code", "ref_date", "last_adj"])
    out_path = out_dir / "stock_daily_last_adj.parquet"
    n = _write_full(out_path, df, LAST_ADJ_SCHEMA)
    _log(f"stock_daily_last_adj: {n} 行（ref_date={max_date}）")


def cmd_report_rc(out_dir: Path) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    df = _fetch_all(pro, "report_rc")
    if df is None or df.empty:
        _log("report_rc 无数据")
        return
    df = _norm_date(df, ["report_date"])
    df = df.drop_duplicates(subset=["ts_code", "report_date", "org_name", "report_title"])
    path = out_dir / "report_rc.parquet"
    if path.exists():
        schema = pq.ParquetFile(path).schema_arrow
        df = _align_df(df, schema.names)
        df = _coerce_by_schema(df, schema)
        out_path = path.with_name(f".{path.stem}.tmp.{os.getpid()}.parquet")
        df.to_parquet(out_path, index=False, compression="zstd")
        os.replace(out_path, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="zstd")
    _log(f"report_rc: {len(df)} 行（{df['report_date'].min()} ~ {df['report_date'].max()}）")


def _pick_codes(out_dir: Path, limit: int, codes_file: str | None) -> list[str]:
    if codes_file:
        return [line.strip() for line in Path(codes_file).read_text().splitlines() if line.strip()]
    codes = _codes_from_stock_basic(out_dir / "stock_basic.parquet")
    if limit and limit > 0:
        codes = codes[:limit]
    return codes


def _sync_events_one(out_dir: Path, pro, code: str) -> dict[str, int]:
    counts = {"dividend": 0, "share_float": 0, "namechange": 0}
    for api in ("dividend", "share_float", "namechange"):
        try:
            df = _fetch_all(pro, api, ts_code=code)
        except Exception as exc:
            _log(f"WARNING {api} {code} 失败: {exc.__class__.__name__}: {exc}")
            continue
        if df is None or df.empty:
            continue
        df = _norm_date(df, [c for c in df.columns if "date" in c.lower()])
        path = out_dir / f"{api}.parquet"
        counts[api] = _merge_replace_code(path, df, code)
    return counts


def cmd_events(out_dir: Path, limit: int, codes_file: str | None,
               workers: int = 3) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    codes = _pick_codes(out_dir, limit, codes_file)
    _log(f"事件表处理 {len(codes)} 只（dividend/share_float/namechange, workers={workers}）")
    counts = {"dividend": 0, "share_float": 0, "namechange": 0}
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sync_events_one, out_dir, pro, c): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                sub = fut.result()
                for k in counts:
                    counts[k] += sub.get(k, 0)
            except Exception as exc:
                failed += 1
                _log(f"WARNING events {code} 异常: {exc.__class__.__name__}: {exc}")
            done += 1
            if done % 100 == 0 or done == len(codes):
                _log(f"进度 {done}/{len(codes)} 失败 {failed}")
    _log(f"完成: {counts} 失败 {failed}/{len(codes)}")


def _sync_fina_one(out_dir: Path, pro, code: str) -> dict[str, int]:
    counts = {api: 0 for api in ("fina_indicator", "income", "balancesheet", "cashflow")}
    for api in counts:
        try:
            df = _fetch_all(pro, api, ts_code=code)
        except Exception as exc:
            _log(f"WARNING {api} {code} 失败: {exc.__class__.__name__}: {exc}")
            continue
        if df is None or df.empty:
            continue
        df = _norm_date(df, [c for c in df.columns if c in _DATE_COLS])
        pk = ["ts_code", "end_date", "report_type"] if api != "fina_indicator" else ["ts_code", "end_date"]
        df = df.drop_duplicates(subset=pk, keep="last")
        path = out_dir / f"{api}.parquet"
        counts[api] = _merge_replace_code(path, df, code)
    return counts


def cmd_fina(out_dir: Path, limit: int, codes_file: str | None,
             workers: int = 3) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    codes = _pick_codes(out_dir, limit, codes_file)
    _log(f"财务表处理 {len(codes)} 只（fina_indicator/income/balancesheet/cashflow, workers={workers}）")
    counts = {api: 0 for api in ("fina_indicator", "income", "balancesheet", "cashflow")}
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sync_fina_one, out_dir, pro, c): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                sub = fut.result()
                for k in counts:
                    counts[k] += sub.get(k, 0)
            except Exception as exc:
                failed += 1
                _log(f"WARNING fina {code} 异常: {exc.__class__.__name__}: {exc}")
            done += 1
            if done % 50 == 0 or done == len(codes):
                _log(f"进度 {done}/{len(codes)} 失败 {failed}")
    _log(f"完成: {counts} 失败 {failed}/{len(codes)}")


def _sync_surv_one(out_dir: Path, pro, code: str) -> dict[str, int]:
    counts = {"forecast": 0, "express": 0, "stk_surv": 0}
    for api in ("forecast", "express", "stk_surv"):
        try:
            df = _fetch_all(pro, api, ts_code=code)
        except Exception as exc:
            _log(f"WARNING {api} {code} 失败: {exc.__class__.__name__}: {exc}")
            continue
        if df is None or df.empty:
            continue
        df = _norm_date(df, [c for c in df.columns if "date" in c.lower()])
        path = out_dir / f"{api}.parquet"
        counts[api] = _merge_replace_code(path, df, code)
    return counts


def cmd_surv(out_dir: Path, limit: int, codes_file: str | None,
             workers: int = 3) -> None:
    if not is_configured():
        raise SystemExit("未配置 TUSHARE_TOKEN")
    pro = get_pro()
    codes = _pick_codes(out_dir, limit, codes_file)
    _log(f"事件表处理 {len(codes)} 只（forecast/express/stk_surv, workers={workers}）")
    counts = {"forecast": 0, "express": 0, "stk_surv": 0}
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sync_surv_one, out_dir, pro, c): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                sub = fut.result()
                for k in counts:
                    counts[k] += sub.get(k, 0)
            except Exception as exc:
                failed += 1
                _log(f"WARNING surv {code} 异常: {exc.__class__.__name__}: {exc}")
            done += 1
            if done % 100 == 0 or done == len(codes):
                _log(f"进度 {done}/{len(codes)} 失败 {failed}")
    _log(f"完成: {counts} 失败 {failed}/{len(codes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="同步 Tushare 数据直接写入 Parquet")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--basic", action="store_true", help="股票列表 + 交易日历（全量覆盖）")
    parser.add_argument("--daily-since", metavar="YYYYMMDD", help="按交易日增量拉 daily/daily_basic/adj_factor")
    parser.add_argument("--daily-workers", type=int, default=1, help="日线按交易日并发数（代理限流建议 3，别超过 4）")
    parser.add_argument("--force-daily", action="store_true", help="强制重拉 --daily-since 区间内全部日期")
    parser.add_argument("--last-adj", action="store_true", help="从 stock_daily.parquet 重建复权因子锚点快照")
    parser.add_argument("--report-rc", action="store_true", help="一致预期（全量覆盖）")
    parser.add_argument("--events", action="store_true", help="dividend/share_float/namechange")
    parser.add_argument("--fina", action="store_true", help="财务宽表 fina_indicator/income/balancesheet/cashflow")
    parser.add_argument("--surv", action="store_true", help="业绩预告/快报/机构调研")
    parser.add_argument("--limit", type=int, default=0, help="事件/财务全市场时只处理前 N 只")
    parser.add_argument("--codes-file", help="事件/财务处理指定股票列表文件")
    parser.add_argument("--workers", type=int, default=3, help="事件/财务并发请求数（代理建议 3，别超过 4）")
    parser.add_argument("--sleep", type=float, default=None, help="Tushare 请求间隔秒数，覆盖 TUSHARE_SLEEP")
    args = parser.parse_args()

    if args.sleep is not None:
        global TUSHARE_SLEEP
        TUSHARE_SLEEP = max(0.0, args.sleep)

    if args.daily_since or args.last_adj:
        raise SystemExit(
            "股票日线已迁移到 CNEquity 年度档案，请改用 "
            "python scripts/sync_daily_to_cne.py [--since YYYY-MM-DD] [--end YYYY-MM-DD]"
        )
    if not any([args.basic, args.report_rc, args.events, args.fina, args.surv]):
        parser.print_help()
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.basic:
        cmd_basic(out_dir)
    if args.daily_since:
        cmd_daily_since(args.daily_since, out_dir, args.daily_workers,
                        force=args.force_daily)
    if args.last_adj:
        cmd_last_adj(out_dir)
    if args.report_rc:
        cmd_report_rc(out_dir)
    if args.events:
        cmd_events(out_dir, args.limit, args.codes_file, args.workers)
    if args.fina:
        cmd_fina(out_dir, args.limit, args.codes_file, args.workers)
    if args.surv:
        cmd_surv(out_dir, args.limit, args.codes_file, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
