#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Tushare parquet 原始价 + 复权因子重建/增量更新股票 panel.parquet（稳定前复权）。

用法:
    python scripts/rebuild_stock_panel_from_pg.py [--start 2020-01-01] [--batch 200]
    python scripts/rebuild_stock_panel_from_pg.py --force-full   # 强制全量重建

数据来源:
    - 行情: data/pg_parquet/stock_daily.parquet（sync_tushare_to_parquet.py 直写）
    - 前复权锚点: data/pg_parquet/stock_daily_last_adj.parquet（同步时生成）
    - 股票池: data/universe.csv（沪深300+中证500+中证1000）

口径:
    与 core.data._finalize_stock_df 一致：qfq = 原始价 * adj_factor / last_adj，
    turn20/am20 为每只股票滚动 20 日均值（min_periods=15）。
    panel 文件保留原 schema（date/open/close/turnover/amount/code/turn20/am20/volume）。

内存:
    按股票分批处理，每批几百只股票的历史，3.6G 机器安全。

增量策略:
    panel 的 qfq 价格以“最新交易日的复权因子”为锚点（last_adj）。
    若新交易日没有复权因子变化，旧 panel 行仍有效，只需追加新日期；
    只有 last_adj 变化的股票才全量重算并替换，避免每天全量重扫 400 万行。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data import _finalize_stock_df  # noqa: E402
from core.store import PANEL_FILE, UNIVERSE_FILE  # noqa: E402

PANEL_SCHEMA = pa.schema([
    pa.field("date", pa.timestamp("ns")),
    pa.field("open", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("turnover", pa.float32()),
    pa.field("amount", pa.float64()),
    pa.field("code", pa.string()),
    pa.field("turn20", pa.float32()),
    pa.field("am20", pa.float32()),
    pa.field("volume", pa.float32()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
])

KEEP_COLS = ["date", "open", "close", "turnover", "amount", "code",
             "turn20", "am20", "volume", "high", "low"]
BUFFER_DAYS = 40
ANCHOR_FILE = ROOT / "data" / "panel_anchor.parquet"
LAST_ADJ_FILE = ROOT / "data" / "pg_parquet" / "stock_daily_last_adj.parquet"


def load_universe_codes() -> list[str]:
    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"股票池不存在: {UNIVERSE_FILE}")
    df = pd.read_csv(UNIVERSE_FILE, dtype={"code": str})
    codes = sorted({str(c).zfill(6) for c in df["code"].astype(str)})
    if not codes:
        raise SystemExit(f"股票池为空: {UNIVERSE_FILE}")
    return codes


def load_ts_code_map() -> dict[str, str]:
    """6 位代码 -> 完整 ts_code（读导出快照 stock_basic.parquet）。"""
    path = ROOT / "data" / "pg_parquet" / "stock_basic.parquet"
    if not path.exists():
        raise SystemExit(f"股票注册表不存在: {path}（先跑 sync_tushare_to_parquet.py --basic）")
    df = pd.read_parquet(path, columns=["ts_code", "symbol"])
    return {str(r["symbol"]).zfill(6): str(r["ts_code"]) for _, r in df.iterrows()}


def load_last_adj_snapshot() -> pd.DataFrame | None:
    """读取当前复权因子快照（ts_code, last_adj, ref_date）。"""
    if not LAST_ADJ_FILE.exists():
        return None
    df = pd.read_parquet(LAST_ADJ_FILE)
    if "last_adj" not in df.columns:
        df = df.rename(columns={"adj_factor": "last_adj"})
    return df[["ts_code", "last_adj", "ref_date"]]


def write_panel_parquet(df: pd.DataFrame, path: Path) -> None:
    df = df[KEEP_COLS].copy()
    df["code"] = df["code"].astype(str)
    df["turn20"] = df["turn20"].astype("float32")
    df["am20"] = df["am20"].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    table = pa.Table.from_pandas(df, schema=PANEL_SCHEMA, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def save_anchor(last_adj: pd.DataFrame, start: pd.Timestamp, stock_max) -> None:
    anchor = last_adj[["ts_code", "last_adj"]].copy()
    anchor["ref_date"] = str(stock_max)
    anchor["start"] = str(start.date())
    ANCHOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    anchor.to_parquet(ANCHOR_FILE, index=False)


def rebuild_full(con, ts_codes: list[str], start: pd.Timestamp,
                 calc_start: str, stock_path: Path, batch: int,
                 last_adj: pd.DataFrame, stock_max) -> None:
    """全量重建：所有股票全历史重算（复权锚点变化或首次构建时）。"""
    t0 = time.time()
    rows_total = 0
    writer = None
    tmp_path = PANEL_FILE.with_name(f".{PANEL_FILE.name}.rebuild.tmp")
    batches = [ts_codes[i:i + batch] for i in range(0, len(ts_codes), batch)]
    for bi, sub in enumerate(batches, 1):
        marks = ", ".join(["?"] * len(sub))
        sql = f"""
            SELECT ts_code, trade_date AS date, open, high, low, close,
                   vol AS volume, amount, turnover_rate AS turnover, adj_factor
            FROM read_parquet(?)
            WHERE ts_code IN ({marks}) AND trade_date >= CAST(? AS DATE)
            ORDER BY ts_code, trade_date
        """
        df = con.execute(sql, [str(stock_path)] + sub + [calc_start]).fetchdf()
        if df.empty:
            print(f"  batch {bi}/{len(batches)}: 空", flush=True)
            continue
        out = _finalize_stock_df(df, adj="qfq")
        out = out.dropna(subset=["open", "close"])
        out = out[out["date"] >= start]
        if out.empty:
            print(f"  batch {bi}/{len(batches)}: 空（过滤后）", flush=True)
            continue
        out = out[KEEP_COLS].copy()
        out["code"] = out["code"].astype(str)
        if writer is None:
            PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(tmp_path), PANEL_SCHEMA, compression="zstd")
        table = pa.Table.from_pandas(out, schema=PANEL_SCHEMA, preserve_index=False)
        writer.write_table(table)
        rows_total += len(out)
        print(f"  batch {bi}/{len(batches)}: +{len(out)} 行（累计 {rows_total}）", flush=True)
    if writer is None:
        raise SystemExit("无数据写入，panel 未重建")
    writer.close()
    os.replace(tmp_path, PANEL_FILE)
    save_anchor(last_adj, start, stock_max)
    print(f"完成（全量）: rows={rows_total} codes={len(ts_codes)} "
          f"time={time.time()-t0:.0f}s -> {PANEL_FILE}", flush=True)


def rebuild_incremental(con, ts_codes: list[str], affected: set[str],
                        start: pd.Timestamp, calc_start: str, stock_path: Path,
                        panel_max, last_adj: pd.DataFrame, stock_max, batch: int) -> None:
    """增量更新：未受复权锚点影响的股票只追加新日期，受影响股票全量替换。"""
    panel_max = pd.Timestamp(panel_max)
    affected_6 = {str(t)[:6] for t in affected}
    unaffected = [t for t in ts_codes if t not in affected]
    t0 = time.time()

    # 1) 受影响股票全历史重算（新锚点）
    affected_rows = pd.DataFrame()
    if affected:
        batches = [sorted(affected)[i:i + batch] for i in range(0, len(affected), batch)]
        for sub in batches:
            marks = ", ".join(["?"] * len(sub))
            sql = f"""
                SELECT ts_code, trade_date AS date, open, high, low, close,
                       vol AS volume, amount, turnover_rate AS turnover, adj_factor
                FROM read_parquet(?)
                WHERE ts_code IN ({marks}) AND trade_date >= CAST(? AS DATE)
                ORDER BY ts_code, trade_date
            """
            df = con.execute(sql, [str(stock_path)] + sub + [calc_start]).fetchdf()
            if df.empty:
                continue
            out = _finalize_stock_df(df, adj="qfq")
            out = out.dropna(subset=["open", "close"])
            out = out[out["date"] >= start]
            if not out.empty:
                affected_rows = pd.concat([affected_rows, out[KEEP_COLS]],
                                          ignore_index=True)
        print(f"  复权因子变化 {len(affected)} 只，全量重算 {len(affected_rows)} 行",
              flush=True)

    # 2) 未受影响股票：只取新日期，并用旧 panel 尾部保证滚动因子正确
    new_rows = pd.DataFrame()
    if unaffected:
        marks = ", ".join(["?"] * len(unaffected))
        sql = f"""
            SELECT ts_code, trade_date AS date, open, high, low, close,
                   vol AS volume, amount, turnover_rate AS turnover, adj_factor
            FROM read_parquet(?)
            WHERE ts_code IN ({marks}) AND trade_date > CAST(? AS DATE)
            ORDER BY ts_code, trade_date
        """
        df_new = con.execute(
            sql, [str(stock_path)] + unaffected + [str(panel_max)]
        ).fetchdf()
        if not df_new.empty:
            qfq_new = _finalize_stock_df(df_new, adj="qfq")
            qfq_new = qfq_new.dropna(subset=["open", "close"])
            qfq_new = qfq_new[qfq_new["date"] > panel_max]
            if not qfq_new.empty:
                un6 = [str(t)[:6] for t in unaffected]
                tail_marks = ", ".join(["?"] * len(un6))
                tail = con.execute(
                    f"""
                    SELECT * FROM read_parquet(?)
                    WHERE code IN ({tail_marks})
                      AND date > (SELECT max(date) - INTERVAL '45 days'
                                  FROM read_parquet(?))
                    """,
                    [str(PANEL_FILE)] + un6 + [str(PANEL_FILE)],
                ).fetchdf()
                combined = pd.concat([tail[KEEP_COLS], qfq_new[KEEP_COLS]],
                                     ignore_index=True)
                combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
                g = combined.groupby("code", observed=True)
                combined["turn20"] = g["turnover"].transform(
                    lambda s: s.rolling(20, min_periods=15).mean())
                combined["am20"] = g["amount"].transform(
                    lambda s: s.rolling(20, min_periods=15).mean())
                new_rows = combined[combined["date"] > panel_max].copy()
                new_rows = new_rows.drop_duplicates(["code", "date"], keep="last")
        print(f"  未受影响 {len(unaffected)} 只，新增 {len(new_rows)} 行", flush=True)

    if affected_rows.empty and new_rows.empty:
        print("无新增/变更行，panel 保持不变", flush=True)
        save_anchor(last_adj, start, stock_max)
        return

    # 3) DuckDB 流式合并：旧 panel（剔除受影响股票） + 重算行 + 新日期行
    old_sql = f"SELECT * FROM read_parquet('{PANEL_FILE}')"
    universe_6 = [str(c)[:6] for c in ts_codes]
    if universe_6:
        old_sql += f" WHERE code IN ({', '.join(['?'] * len(universe_6))})"
    if affected_6:
        old_sql += (f" AND code NOT IN ({', '.join(['?'] * len(affected_6))})"
                    if universe_6
                    else f" WHERE code NOT IN ({', '.join(['?'] * len(affected_6))})")
    parts = [old_sql]
    params = universe_6 + (sorted(affected_6) if affected_6 else [])
    tmp_files: list[Path] = []
    if not affected_rows.empty:
        affected_file = PANEL_FILE.with_name(".panel.affected.tmp.parquet")
        write_panel_parquet(affected_rows, affected_file)
        tmp_files.append(affected_file)
        parts.append(f"SELECT * FROM read_parquet('{affected_file}')")
    if not new_rows.empty:
        new_file = PANEL_FILE.with_name(".panel.new.tmp.parquet")
        write_panel_parquet(new_rows, new_file)
        tmp_files.append(new_file)
        parts.append(f"SELECT * FROM read_parquet('{new_file}')")

    tmp_path = PANEL_FILE.with_name(f".{PANEL_FILE.name}.merge.tmp")
    try:
        esc_tmp = str(tmp_path).replace("'", "''")
        con.execute(
            "COPY (" + " UNION ALL ".join(parts) + f") TO '{esc_tmp}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)",
            params,
        )
        os.replace(tmp_path, PANEL_FILE)
    finally:
        for p in tmp_files + [tmp_path]:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    save_anchor(last_adj, start, stock_max)
    total = con.execute(
        "SELECT count(*) FROM read_parquet(?)", [str(PANEL_FILE)]
    ).fetchone()[0]
    print(f"完成（增量）: rows={total:,} 重算={len(affected_rows):,} "
          f"新增={len(new_rows):,} time={time.time()-t0:.0f}s -> {PANEL_FILE}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="从 PG 原始价+复权因子重建/增量更新股票 panel.parquet")
    ap.add_argument("--start", default="2015-01-01", help="面板起始日期（默认 2015-01-01，覆盖旧 panel 全部区间）")
    ap.add_argument("--batch", type=int, default=200, help="每批股票数（内存安全）")
    ap.add_argument("--force-full", action="store_true", help="强制全量重建（默认增量追加）")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    calc_start = (start - pd.Timedelta(days=BUFFER_DAYS)).date().isoformat()
    codes = load_universe_codes()
    ts_map = load_ts_code_map()
    ts_by_code = {c: ts_map.get(c) for c in codes}
    missing = [c for c, t in ts_by_code.items() if t is None]
    if missing:
        print(f"警告: {len(missing)} 只股票不在 stock_basic 中（跳过）: {missing[:10]}", flush=True)
    ts_codes = sorted(t for t in ts_by_code.values() if t)
    if not ts_codes:
        raise SystemExit("股票池无可匹配 ts_code，中止")
    stock_path = ROOT / "data" / "pg_parquet" / "stock_daily.parquet"
    if not stock_path.exists():
        raise SystemExit(f"行情快照不存在: {stock_path}（先跑 sync_tushare_to_parquet.py --daily-since）")

    con = duckdb.connect()
    stock_max = con.execute(
        "SELECT max(trade_date) FROM read_parquet(?)", [str(stock_path)]
    ).fetchone()[0]
    print(f"股票池 {len(codes)} 只（可匹配 {len(ts_codes)} 只），行情 {start.date()} ~ {stock_max}（起点前移 {BUFFER_DAYS} 天做滚动因子）", flush=True)

    last_adj = load_last_adj_snapshot()
    if last_adj is None:
        raise SystemExit(
            f"复权因子快照不存在: {LAST_ADJ_FILE}（先跑 sync_tushare_to_parquet.py --last-adj）")

    # panel 已是最新则跳过
    panel_max = None
    if PANEL_FILE.exists():
        try:
            panel_max = con.execute(
                "SELECT max(date) FROM read_parquet(?)", [str(PANEL_FILE)]
            ).fetchone()[0]
        except Exception:
            panel_max = None
    if panel_max is not None and pd.Timestamp(panel_max) >= pd.Timestamp(stock_max):
        print(f"panel 已是最新（{panel_max}），跳过重建", flush=True)
        return

    # 判定全量 or 增量：复权锚点（last_adj）变化或首次构建时全量
    full = args.force_full or not PANEL_FILE.exists() or panel_max is None
    affected: set[str] = set()
    if not full:
        prev_anchor = pd.read_parquet(ANCHOR_FILE) if ANCHOR_FILE.exists() else None
        if prev_anchor is None or str(prev_anchor["start"].iloc[0]) != str(start.date()):
            full = True
        else:
            merged = prev_anchor[["ts_code", "last_adj"]].merge(
                last_adj[["ts_code", "last_adj"]],
                on="ts_code", how="outer", suffixes=("_old", "_new"),
            )
            changed = merged[
                merged["last_adj_old"].fillna(-1) != merged["last_adj_new"].fillna(-1)
            ]["ts_code"].tolist()
            affected = {str(t) for t in changed if str(t) in set(ts_codes)}
            affected |= {t for t in ts_codes if t not in set(prev_anchor["ts_code"])}
        if full:
            affected = set()

    try:
        if full:
            rebuild_full(con, ts_codes, start, calc_start, stock_path,
                         args.batch, last_adj, stock_max)
        else:
            rebuild_incremental(con, ts_codes, affected, start, calc_start,
                                stock_path, panel_max, last_adj, stock_max,
                                args.batch)
    finally:
        con.close()


if __name__ == "__main__":
    main()
