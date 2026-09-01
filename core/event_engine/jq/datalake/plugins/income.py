# -*- coding: utf-8 -*-
"""利润表(CNE curated income, 合并报表, ann_date 点时口径)。"""
from __future__ import annotations

import pandas as pd

from core.event_engine.jq.datalake.base import CNE_CURATED, JQDataPlugin

PLUGIN = JQDataPlugin(
    name="income",
    description="利润表(合并报表): 归母净利/净利/营收, ann_date 点时",
    date_column="ann_date",
    entity_keys=("code",),
)


def _iter_parquets() -> list[str]:
    base = CNE_CURATED / "income"
    out: list[str] = []
    if base.exists():
        for part in sorted(base.iterdir()):
            if part.is_dir():
                out.extend(str(p) for p in sorted(part.glob("*.parquet")))
    return out


def load() -> pd.DataFrame:
    """全量合并, 按 (code, end_date) 去重保留最新公告, 按 (code, ann_date) 排序。"""
    frames = []
    for f in _iter_parquets():
        df = pd.read_parquet(f)
        if "ts_code" not in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ts_code"})
        cols = [c for c in ("ts_code", "ann_date", "end_date", "n_income",
                            "n_income_attr_p", "revenue", "report_type")
                if c in df.columns]
        frames.append(df[cols])
    if not frames:
        return pd.DataFrame()
    inc = pd.concat(frames, ignore_index=True)
    inc = inc[pd.to_numeric(inc.get("report_type"), errors="coerce") == 1]
    inc["code"] = inc["ts_code"].astype(str).str[:6]
    inc["ann_date"] = pd.to_datetime(inc["ann_date"], errors="coerce")
    inc["end_date"] = pd.to_datetime(inc["end_date"], errors="coerce")
    inc = inc.dropna(subset=["ann_date"])
    inc = inc.sort_values(["code", "end_date", "ann_date"], kind="stable")
    inc = inc.drop_duplicates(["code", "end_date"], keep="last")
    inc = inc.sort_values(["code", "ann_date"], kind="stable")
    return inc.reset_index(drop=True)
