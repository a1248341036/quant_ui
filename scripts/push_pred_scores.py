# -*- coding: utf-8 -*-
"""pred 分数下发：raw scores → 主板/价格池过滤 → ``date, code, score`` 长表。

背景（2026-11-05 修复）
----------------------
``artifacts/alphaagent/stacking/_ops_b_20261030/b_ops_filter.py`` 用

    m.loc[m["close"] <= 10, "score"] = np.nan

实现注释里写的"低价池(close>10)"，**方向与名字相反**——它剔除的是 10 元以下，
实际产出的是高价池。后果：ML 组合 pred 池 1549 只里 10 元以下只有 15 只，
38 只银行只剩 10 只（工行/农行/中行从未进入过），模拟盘与回测从未覆盖低价股。

同时仓库里存在两套方向相反的下发脚本（``push_icir_tail.py`` 保留 <=10，
``b_ops_filter.py`` 保留 >10），二者都能覆盖 ``pred_demo.parquet``。本模块把下发
口径收口成**带名字的参数**，让"低价池/高价池"不再靠注释猜：

* ``--price-max``  保留 ``close <= 该值``（默认 10.0，即低价池）
* ``--price-min``  保留 ``close >= 该值``（默认 None，不设下限）

两者同时给出即为价格区间。边界语义：**上下限均含等号**（``<=`` / ``>=``）。

安全默认
--------
缺省输出到 ``data/stock/pred_demo_lowprice.parquet``，**不覆盖**生产
``data/stock/pred_demo.parquet``；覆盖生产需显式 ``--overwrite-production``。

用法
----
    python scripts/push_pred_scores.py data/stock/pred_scores_raw_ops_20261030.parquet
    python scripts/push_pred_scores.py raw.parquet --price-max 10 --board main
    python scripts/push_pred_scores.py raw.parquet --price-min 10 --out highprice.parquet
    python scripts/push_pred_scores.py raw.parquet --overwrite-production   # 显式覆盖
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 代码根与数据根分离：在 git worktree 里跑时，代码是本 worktree 的，
# 但 data/ artifacts/ 属未跟踪产物，仍指向共享工作区。
DATA_ROOT = Path(os.environ.get("QUANT_DATA_ROOT", str(PROJECT_ROOT)))
DATA_DIR = DATA_ROOT / "data" / "stock"

# 主板代码前缀（与 backend.routers.paper.JQ_UNIVERSE「全A主板」口径一致）：
# 沪市 600/601/603/605，深市 000/001/002/003。排除 688/689（科创）、
# 300/301/302（创业）、8/4（北交所）。
MB_PREFIX = ("600", "601", "603", "605", "000", "001", "002", "003")

# 低价池默认阈值（元）。改这里 = 改口径，须同步更新单测与本模块文档。
DEFAULT_PRICE_MAX = 10.0

DEFAULT_OUT = DATA_DIR / "pred_demo_lowprice.parquet"
PROD_PRED = DATA_DIR / "pred_demo.parquet"


def normalize_codes(series: pd.Series) -> pd.Series:
    """任意 code 表示 → 6 位数字字符串。"""
    extracted = series.astype(str).str.extract(r"(\d{6})", expand=False)
    return extracted.where(extracted.notna(), series.astype(str)).str.zfill(6)


def mainboard_mask(codes: pd.Series) -> pd.Series:
    """主板前缀命中掩码。"""
    return codes.astype(str).str[:3].isin(MB_PREFIX)


def apply_price_pool(
    df: pd.DataFrame,
    price_max: float | None = DEFAULT_PRICE_MAX,
    price_min: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """按收盘价保留区间内的行，其余置 NaN 后剔除。

    参数
    ----
    df : 需含 ``close`` 与 ``score`` 两列。
    price_max : 保留 ``close <= price_max``；None 表示不设上限。
    price_min : 保留 ``close >= price_min``；None 表示不设下限。

    close 为 NaN 的行一律剔除（无法判定价格区间时不做赌）。

    返回 (过滤后 df, 统计 dict)。
    """
    if "close" not in df.columns or "score" not in df.columns:
        raise ValueError("apply_price_pool 需要 close 与 score 两列")

    out = df.copy()
    n0 = int(out["score"].notna().sum())

    missing = out["close"].isna()
    if price_max is not None:
        above = out["close"] > float(price_max)
    else:
        above = pd.Series(False, index=out.index)
    if price_min is not None:
        below = out["close"] < float(price_min)
    else:
        below = pd.Series(False, index=out.index)

    drop = missing | above | below
    out.loc[drop, "score"] = np.nan

    stats = {
        "input_rows": int(len(df)),
        "input_scored": n0,
        "dropped_price_missing": int(missing.sum()),
        "dropped_over_max": int(above.sum()),
        "dropped_below_min": int(below.sum()),
        "kept_scored": int(out["score"].notna().sum()),
        "price_min": price_min,
        "price_max": price_max,
    }
    return out, stats


def load_close_long(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """从 CNE 日线取 ``date, code, close`` 长表（只读需要的年份文件）。"""
    import pyarrow.parquet as pq

    root = DATA_ROOT / "data" / "quant_dataset"
    frames: list[pd.DataFrame] = []
    for ydir in sorted(root.glob("[0-9][0-9][0-9][0-9]")):
        try:
            year = int(ydir.name)
        except ValueError:
            continue
        if year < start.year or year > end.year:
            continue
        f = ydir / ydir.name / "day" / "stock_daily.parquet"
        if not f.exists():
            continue
        tbl = pq.read_table(f, columns=["trade_date", "ts_code", "close"])
        d = tbl.to_pandas()
        d["date"] = pd.to_datetime(d["trade_date"])
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        d["code"] = normalize_codes(d["ts_code"])
        frames.append(d[["date", "code", "close"]])
    if not frames:
        return pd.DataFrame(columns=["date", "code", "close"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "code"])


def build_pool(
    raw: pd.DataFrame,
    close_long: pd.DataFrame,
    board: str = "main",
    price_max: float | None = DEFAULT_PRICE_MAX,
    price_min: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """raw scores + close → 过滤后的 ``date, code, score``。"""
    s = raw.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["code"] = normalize_codes(s["code"])

    merged = s.merge(close_long, on=["date", "code"], how="left")

    stats: dict = {"raw_rows": int(len(s)), "board": board}
    if board == "main":
        mask = mainboard_mask(merged["code"])
        merged = merged[mask]
        stats["after_board"] = int(len(merged))
    elif board == "all":
        stats["after_board"] = int(len(merged))
    else:
        raise ValueError(f"未知 board={board!r}，只能是 main / all")

    merged, price_stats = apply_price_pool(merged, price_max=price_max, price_min=price_min)
    stats.update(price_stats)

    out = merged[["date", "code", "score"]].dropna(subset=["score"])
    out = out.sort_values(["date", "code"]).reset_index(drop=True)
    stats["final_rows"] = int(len(out))
    stats["final_dates"] = int(out["date"].nunique()) if len(out) else 0
    stats["final_codes"] = int(out["code"].nunique()) if len(out) else 0
    return out, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pred 分数下发（价格池口径显式化）")
    ap.add_argument("raw", help="raw scores parquet（date/code/score）")
    ap.add_argument("--out", default=None, help=f"输出路径（默认 {DEFAULT_OUT}）")
    ap.add_argument("--board", default="main", choices=["main", "all"],
                    help="主板白名单过滤（默认 main）")
    ap.add_argument("--price-max", type=float, default=DEFAULT_PRICE_MAX,
                    help=f"保留 close <= 该值（默认 {DEFAULT_PRICE_MAX}，低价池；传 -1 关闭上限）")
    ap.add_argument("--price-min", type=float, default=None,
                    help="保留 close >= 该值（默认不设下限）")
    ap.add_argument("--overwrite-production", action="store_true",
                    help="允许覆盖 data/stock/pred_demo.parquet（默认拒绝）")
    args = ap.parse_args(argv)

    price_max = None if args.price_max is not None and args.price_max < 0 else args.price_max
    out_path = Path(args.out) if args.out else DEFAULT_OUT

    if out_path.resolve() == PROD_PRED.resolve() and not args.overwrite_production:
        print(f"[拒绝] 目标为生产 pred_demo：{PROD_PRED}", file=sys.stderr)
        print("       确认口径无误后加 --overwrite-production 显式覆盖。", file=sys.stderr)
        return 2

    raw = pd.read_parquet(args.raw)
    print(f"[0] raw: {raw.shape}  dates {raw['date'].min()} -> {raw['date'].max()}")

    start = pd.to_datetime(raw["date"]).min()
    end = pd.to_datetime(raw["date"]).max()
    close_long = load_close_long(start, end)
    print(f"[1] close: {close_long.shape}  dates {start.date()} -> {end.date()}")

    out, stats = build_pool(raw, close_long, board=args.board,
                            price_max=price_max, price_min=args.price_min)
    print(f"[2] board={stats['board']}: {stats['raw_rows']} -> {stats.get('after_board')}")
    print(f"[3] price pool (min={stats['price_min']}, max={stats['price_max']}): "
          f"kept {stats['kept_scored']}/{stats['input_scored']}  "
          f"(低于下限 {stats['dropped_below_min']}, 高于上限 {stats['dropped_over_max']}, "
          f"close缺失 {stats['dropped_price_missing']})")
    print(f"[4] final: {out.shape}  dates {stats['final_dates']}  codes {stats['final_codes']}")

    if len(out):
        print(f"[5] 价格留存量校验：保留池 price_max={price_max}")
    out.to_parquet(out_path, index=False)
    print(f"已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
