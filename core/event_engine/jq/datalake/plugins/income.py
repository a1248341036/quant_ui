# -*- coding: utf-8 -*-
"""利润表(data/pg_parquet/income.parquet, 合并报表, ann_date 点时口径)。"""
from __future__ import annotations

import pandas as pd

from core.event_engine.jq.datalake.base import PG, JQDataPlugin

PLUGIN = JQDataPlugin(
    name="income",
    description="利润表(合并报表): 归母净利/净利/营收, ann_date 点时",
    date_column="ann_date",
    entity_keys=("code",),
)


def load() -> pd.DataFrame:
    """按 (code, end_date) 去重保留最新公告, 按 (code, ann_date) 稳定排序。"""
    inc = pd.read_parquet(PG / "income.parquet",
                          columns=["ts_code", "ann_date", "end_date",
                                   "n_income", "n_income_attr_p", "revenue",
                                   "report_type"])
    inc = inc[pd.to_numeric(inc["report_type"], errors="coerce") == 1]
    inc["code"] = inc["ts_code"].str[:6]
    inc["ann_date"] = pd.to_datetime(inc["ann_date"], errors="coerce")
    inc["end_date"] = pd.to_datetime(inc["end_date"], errors="coerce")
    inc = inc.dropna(subset=["ann_date"])
    inc = inc.sort_values(["code", "end_date", "ann_date"], kind="stable")
    inc = inc.drop_duplicates(["code", "end_date"], keep="last")
    inc = inc.sort_values(["code", "ann_date"], kind="stable")
    return inc.reset_index(drop=True)
