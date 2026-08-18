#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 Qlib 数据源：从 data/panel.parquet 生成本地 qlib_data/。

用法示例:
    python scripts/export_qlib.py --qlib-dir qlib_data/cn_stock
    python scripts/export_qlib.py --codes 000001,600519 --keep-csv   # 小样本调试
    python scripts/export_qlib.py --use-tushare-calendar             # 用 Tushare 校准日历

流程:
    1. 读 panel.parquet（长表 date/code/open/high/low/close/volume/amount/turnover）
    2. 计算 vwap = amount / (volume * 100)   # amount:元, volume:手 -> 元/股
    3. 按股票导出 CSV（symbol = ts_code，如 000001.SZ.csv）
    4. 调用 scripts/dump_bin.py dump_all 生成:
         - calendars/day.txt（交易日历）
         - instruments/all.txt（股票代码 + 上市/退市区间）
         - features/{symbol}/*.bin（行情字段，Qlib 用 $字段名 引用）

依赖: pyqlib（已装），scripts/dump_bin.py（对应 pyqlib 0.9.7 的官方脚本）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(PROJECT_ROOT))

from core.data import load_panel
from core.store import UNIVERSE_FILE
from core.tushare_client import to_ts_code

DUMP_BIN_SCRIPT = PROJECT_ROOT / "scripts" / "dump_bin.py"
DEFAULT_QLIB_DIR = PROJECT_ROOT / "qlib_data" / "cn_stock"
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "qlib_csv"

# Qlib 因子/模型用到的基础行情字段（对应 features/ 下的字段名）
INCLUDE_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "amount", "turnover"]


def resolve_symbols(codes: list[str] | None = None) -> list[str]:
    """6 位代码 -> Qlib symbol（带交易所后缀），默认读 universe.csv。"""
    if codes is None:
        uni = pd.read_csv(UNIVERSE_FILE, dtype={"code": str})
        codes = sorted(uni["code"].astype(str).str.zfill(6).tolist())
    return [to_ts_code(c) for c in codes]


def add_vwap(panel: pd.DataFrame) -> pd.DataFrame:
    """vwap = 成交额(元) / (成交量(手) * 100) -> 元/股。"""
    out = panel.copy()
    vol = out["volume"].astype(float)
    out["vwap"] = np.where(vol > 0, out["amount"].astype(float) / (vol * 100.0), np.nan)
    return out


def export_csvs(panel: pd.DataFrame, csv_dir: Path,
                symbols: list[str] | None = None) -> list[str]:
    """按股票写出 CSV（csv_dir/{symbol}.csv），返回 symbol 列表。

    dump_bin.py 会从文件名推断 symbol，因此文件名必须是 ts_code.csv。
    date 列固定为 YYYY-MM-DD，其余列统一 float64（Qlib 要求数值列）。
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    sym_set = set(symbols) if symbols else None
    written: list[str] = []
    for code, sub in panel.groupby("code", observed=True):
        sym = to_ts_code(str(code).zfill(6))
        if sym_set is not None and sym not in sym_set:
            continue
        sub = sub[["date"] + INCLUDE_FIELDS].copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")
        for col in INCLUDE_FIELDS:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").astype("float64")
        sub.sort_values("date").drop_duplicates("date", keep="last").to_csv(
            csv_dir / f"{sym}.csv", index=False)
        written.append(sym)
    print(f"[export_qlib] CSV: {len(written)} 只 -> {csv_dir}")
    return written


def run_dump(csv_dir: Path, qlib_dir: Path, max_workers: int) -> None:
    """调用 scripts/dump_bin.py dump_all 生成 Qlib features/calendars/instruments。

    pyqlib 0.9.7 的包内没有 dump 模块，官方脚本在 GitHub scripts/dump_bin.py，
    已放到项目 scripts/ 下。只支持 .bin 数据格式（Qlib 默认）。
    """
    if not DUMP_BIN_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 dump 脚本: {DUMP_BIN_SCRIPT}")
    cmd = [
        sys.executable, str(DUMP_BIN_SCRIPT), "dump_all",
        "--data_path", str(csv_dir),
        "--qlib_dir", str(qlib_dir),
        "--include_fields", ",".join(INCLUDE_FIELDS),
        "--date_field_name", "date",
        "--freq", "day",
        "--max_workers", str(max_workers),
    ]
    print(f"[export_qlib] 运行 dump_bin ...\n  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def apply_tushare_calendar(panel: pd.DataFrame, qlib_dir: Path) -> None:
    """用 Tushare trade_cal 覆盖 calendars/day.txt（可选校准）。

    dump_bin 生成的日历只来自 CSV 里的日期；如果 panel 有缺失交易日，
    用官方日历可保证 Qlib 的日历与真实市场一致。
    """
    try:
        from core.tushare_client import trade_dates
        days = pd.to_datetime(trade_dates(
            str(panel["date"].min().date()), str(panel["date"].max().date())))
    except Exception as exc:  # noqa: BLE001
        print(f"[export_qlib] Tushare 日历失败，保留 dump_bin 生成结果: {exc}",
              file=sys.stderr)
        return
    if not len(days):
        print("[export_qlib] Tushare 日历为空，保留 dump_bin 生成结果", file=sys.stderr)
        return
    cal_dir = qlib_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "day.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in sorted(days)) + "\n",
        encoding="utf-8")
    print(f"[export_qlib] calendars/day.txt 已用 Tushare 校准: {len(days)} 个交易日")


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 Qlib 数据源")
    parser.add_argument("--qlib-dir", type=Path, default=DEFAULT_QLIB_DIR)
    parser.add_argument("--csv-tmp", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--start", default=None, help="面板起始日 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="面板截止日 YYYY-MM-DD")
    parser.add_argument("--codes", default=None,
                        help="逗号分隔 6 位代码，仅导出这些（调试用）")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--use-tushare-calendar", action="store_true",
                        help="dump 后用 Tushare trade_cal 覆盖日历")
    parser.add_argument("--keep-csv", action="store_true", help="保留中间 CSV")
    parser.add_argument("--skip-dump", action="store_true",
                        help="只生成 CSV，不跑 dump（调试转换逻辑用）")
    args = parser.parse_args()

    print("[export_qlib] 读取 panel.parquet ...", flush=True)
    panel = load_panel(start=args.start, end=args.end)
    panel = panel.copy()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    print(f"[export_qlib] panel: rows={len(panel)} codes={panel['code'].nunique()}")

    panel = add_vwap(panel)
    symbols = resolve_symbols(
        [c.strip() for c in args.codes.split(",")] if args.codes else None)
    export_csvs(panel, args.csv_tmp, symbols)

    if not args.skip_dump:
        run_dump(args.csv_tmp, args.qlib_dir, args.max_workers)
        if args.use_tushare_calendar:
            apply_tushare_calendar(panel, args.qlib_dir)

    if not args.keep_csv:
        shutil.rmtree(args.csv_tmp, ignore_errors=True)

    print(f"[export_qlib] 完成: {args.qlib_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())