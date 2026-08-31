# -*- coding: utf-8 -*-
"""股票日线原始帧(data/quant_dataset/<年>/<年>/day/stock_daily.parquet)。

返回 (raw, meta) 两帧:
- raw  域内(前缀过滤)前复权前的原始日线, 供 load_panel 做前复权/am20
- meta 全市场未复权过滤列(涨跌停/市值/ST/上市天数), 不限前缀
按 (start, end, buffer_days) 缓存。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from core.event_engine.jq.datalake.base import JQDataPlugin, QDATA

PLUGIN = JQDataPlugin(
    name="stock_daily",
    description="CNE tushare wide 股票日线(年度 parquet, 原始帧)",
    date_column=None,          # 带参数加载, 不走通用 asof
)

_COLS = ["ts_code", "trade_date", "open", "close", "high", "low",
         "pre_close", "amount", "adj_factor", "up_limit", "down_limit",
         "total_mv", "turnover_rate", "is_st", "listed_days"]
_META_KEEP = ["ts_code", "trade_date", "pre_close", "up_limit", "down_limit",
              "total_mv", "is_st", "listed_days"]
_PANEL_KEEP = ["ts_code", "trade_date", "open", "close", "amount",
               "adj_factor", "turnover_rate"]


def load(start: str, end: str, buffer_days: int = 45,
         prefixes: tuple[str, ...] = ("00", "60")) -> tuple[pd.DataFrame,
                                                            pd.DataFrame]:
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=buffer_days)
    end_ts = pd.Timestamp(end)
    frames, meta_frames = [], []
    years = sorted({p.name for p in QDATA.iterdir()
                    if p.is_dir() and p.name.isdigit()})
    for y in years:
        f = QDATA / y / y / "day" / "stock_daily.parquet"
        if not f.exists():
            continue
        schema = set(pq.read_schema(f).names)
        usecols = [c for c in _COLS if c in schema]
        df = pd.read_parquet(f, columns=usecols)
        if "listed_days" not in df.columns:
            df["listed_days"] = np.nan
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)]
        if df.empty:
            continue
        keep = [c for c in _META_KEEP if c in df.columns]
        meta_frames.append(df[keep].copy())
        code6 = df["ts_code"].str[:6]
        df = df[code6.str.startswith(tuple(prefixes))]
        if df.empty:
            continue
        frames.append(df[[c for c in _PANEL_KEEP if c in df.columns]])
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta = (pd.concat(meta_frames, ignore_index=True)
            if meta_frames else pd.DataFrame())
    if not len(raw):
        raise ValueError(f"quant_dataset 无 {start}~{end} 日线数据")
    return raw, meta
