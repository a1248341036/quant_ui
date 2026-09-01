# -*- coding: utf-8 -*-
"""现金流量表(CNE curated cashflow, 合并报表, ann_date 点时口径)。"""
from __future__ import annotations

import pandas as pd

from core.event_engine.jq.datalake.base import CNE_CURATED, JQDataPlugin

PLUGIN = JQDataPlugin(
    name="cashflow",
    description="现金流量表(合并报表): 经营活动现金流净额等, ann_date 点时",
    date_column="ann_date",
    entity_keys=("code",),
)


def _iter_parquets() -> list[str]:
    base = CNE_CURATED / "cashflow"
    out: list[str] = []
    if base.exists():
        for part in sorted(base.iterdir()):
            if part.is_dir():
                out.extend(str(p) for p in sorted(part.glob("*.parquet")))
    return out


def load() -> pd.DataFrame:
    """全量合并, 按 (code, end_date) 去重保留最新公告。"""
    frames = []
    for f in _iter_parquets():
        df = pd.read_parquet(f)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    cf = pd.concat(frames, ignore_index=True)
    cf["code"] = cf["symbol"].astype(str).str[:6]
    cf["ann_date"] = pd.to_datetime(cf["ann_date"], errors="coerce")
    cf["end_date"] = pd.to_datetime(cf["end_date"], errors="coerce")
    cf = cf.dropna(subset=["ann_date"])
    cf = cf.sort_values(["code", "end_date", "ann_date"], kind="stable")
    cf = cf.drop_duplicates(["code", "end_date"], keep="last")
    cf = cf.sort_values(["code", "ann_date"], kind="stable")
    return cf.reset_index(drop=True)
