# -*- coding: utf-8 -*-
"""指数日线(CNE curated index_bars, TDX 源; cne backfill index_bars 补历史)。"""
from __future__ import annotations

import pandas as pd

from core.event_engine.jq.datalake.base import CNE_CURATED, JQDataPlugin

PLUGIN = JQDataPlugin(
    name="index_bars",
    description="CNE 指数日线(TDX 源): symbol/trade_date/OHLCV/amount",
    date_column="trade_date",
    entity_keys=("symbol",),
)


def load() -> pd.DataFrame:
    root = CNE_CURATED / "index_bars"
    files = sorted(root.glob("trade_date=*/*.parquet"))
    if not files:
        return pd.DataFrame(columns=["symbol", "trade_date", "open", "high",
                                     "low", "close", "volume", "amount"])
    frames = [pd.read_parquet(f, columns=["symbol", "trade_date", "open",
                                          "high", "low", "close",
                                          "volume", "amount"])
              for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (df.sort_values(["symbol", "trade_date"], kind="stable")
              .reset_index(drop=True))
