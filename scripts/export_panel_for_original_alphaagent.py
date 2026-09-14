#!/usr/bin/env python3
"""把 quant_ui CNE 缓存面板导出为原版 AlphaAgent (D:/Quant/AlphaAgent) 的 panel_1d.parquet。

用途：同数据对比实验——原版 Agent 与 quant_ui Agent 在同一份面板上各自挖掘，
盲测段（默认 2025-01-01 起）不导出，双方都不可见。

- 源：artifacts/panel/cache/panel_v3_*.parquet（CNE adapter 磁盘缓存，datetime/instrument 为普通列）
- 目标：原版 load_panel 期望 MultiIndex(datetime, instrument) + OUTPUT_COLUMNS 全集
- 缺列补 NaN（原版 _finalize_panel 同语义）；数值列转 float32 控制体积
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUANT_UI_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path(r"D:\Quant\AlphaAgent")
sys.path.insert(0, str(ORIGINAL_ROOT))

from alphaagent.core.types import OUTPUT_COLUMNS  # noqa: E402
from alphaagent.core.paths import PANEL_PATH  # noqa: E402

TEXT_LIKE = {"is_trade", "not_st"}


def latest_cache_panel() -> Path:
    caches = sorted((QUANT_UI_ROOT / "artifacts" / "panel" / "cache").glob("panel_v3_*.parquet"))
    if not caches:
        raise FileNotFoundError("quant_ui 无 CNE 面板缓存，请先跑一次挖掘或因子实验室构建面板")
    return caches[-1]


def export(src: Path, dst: Path, start: str, end: str) -> None:
    df = pd.read_parquet(src)
    for col in ("datetime", "instrument"):
        if col not in df.columns:
            raise ValueError(f"缓存面板缺少列 {col}（格式与预期不符）: {src}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    df = df.loc[mask]
    if df.empty:
        raise ValueError(f"裁剪后为空: {start}~{end}")
    df = df.set_index(["datetime", "instrument"]).sort_index()

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    out = df.reindex(columns=OUTPUT_COLUMNS).copy()
    for c in missing:
        out[c] = np.nan
    for c in out.columns:
        if c in TEXT_LIKE:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst)
    dt = out.index.get_level_values("datetime")
    print(f"exported: {dst}")
    print(f"  rows={len(out):,} instruments={out.index.get_level_values('instrument').nunique():,}")
    print(f"  range={dt.min().date()} -> {dt.max().date()}")
    print(f"  missing_cols_filled_nan={missing}")
    print(f"  size={dst.stat().st_size / 1e9:.2f} GB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=None, help="CNE 缓存面板路径（默认取最新 panel_v3_*.parquet）")
    p.add_argument("--dst", type=Path, default=PANEL_PATH, help="原版面板输出路径")
    p.add_argument("--start", default="2020-01-01", help="导出起始日（默认 2020-01-01，与 quant_ui 挖掘窗口一致）")
    p.add_argument("--end", default="2024-12-31", help="导出截止日（默认 2024-12-31，2025+ 留作共同盲测段）")
    args = p.parse_args()
    src = args.src or latest_cache_panel()
    export(src, args.dst, args.start, args.end)


if __name__ == "__main__":
    main()
