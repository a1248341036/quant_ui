# -*- coding: utf-8 -*-
"""行业分类(CNE curated industry_members, 月度快照, 点时口径)。

系统: sw(申万, 6 位层级 code, 前缀拆级) + eastmoney(东财行业, 带中文名,
粒度近似聚宽二级/证监会)。eastmoney 快照只覆盖早期月份, 2021+ 以 sw 回退。
"""
from __future__ import annotations

import pandas as pd

from core.event_engine.jq.datalake.base import CNE_CURATED, JQDataPlugin

PLUGIN = JQDataPlugin(
    name="industry_members",
    description="行业分类月度快照(sw 申万 + eastmoney 东财)",
    date_column="as_of_date",
    entity_keys=("code", "system"),
    asof_fallback_first=True,
)

_COLS = ["symbol", "classification_system", "industry_code",
         "industry_name", "as_of_date"]


def load() -> pd.DataFrame:
    root = CNE_CURATED / "industry_members"
    files = sorted(root.glob("**/*.parquet"))
    if not files:
        return pd.DataFrame(columns=["code", "system", "industry_code",
                                     "industry_name", "as_of_date"])
    df = pd.concat([pd.read_parquet(f, columns=_COLS) for f in files],
                   ignore_index=True)
    df["code"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    df["system"] = df["classification_system"].astype(str)
    df["industry_code"] = df["industry_code"].astype(str)
    df["industry_name"] = df["industry_name"].astype(str)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return (df[["code", "system", "industry_code", "industry_name",
                "as_of_date"]]
            .sort_values(["code", "system", "as_of_date"], kind="stable")
            .reset_index(drop=True))
