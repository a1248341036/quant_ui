# -*- coding: utf-8 -*-
"""资产负债表(CNE curated balancesheet, 合并报表, ann_date 点时口径)。"""
from __future__ import annotations

import pandas as pd

from core.event_engine.jq.datalake.base import CNE_CURATED, JQDataPlugin

PLUGIN = JQDataPlugin(
    name="balancesheet",
    description="资产负债表(合并报表): 归母净资产等, ann_date 点时",
    date_column="ann_date",
    entity_keys=("code",),
)


def _iter_parquets() -> list[str]:
    base = CNE_CURATED / "balancesheet"
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
        if "ts_code" not in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ts_code"})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    bal = pd.concat(frames, ignore_index=True)
    bal = bal[pd.to_numeric(bal.get("report_type"), errors="coerce") == 1]
    bal["code"] = bal["ts_code"].astype(str).str[:6]
    ann = pd.to_datetime(bal["f_ann_date"], errors="coerce").fillna(
        pd.to_datetime(bal["ann_date"], errors="coerce"))
    bal["ann_date"] = ann
    bal = bal.dropna(subset=["ann_date"])
    bal = bal.sort_values(["code", "end_date", "ann_date"], kind="stable")
    bal = bal.drop_duplicates(["code", "end_date"], keep="last")
    bal = bal.sort_values(["code", "ann_date"], kind="stable")
    return bal.reset_index(drop=True)
