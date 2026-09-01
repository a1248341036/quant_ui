# -*- coding: utf-8 -*-
"""财务指标指标表(CNE curated fina_indicator, ann_date 点时口径)。

tushare fina_indicator 按报告期分区, 全量加载后按 (code, end_date) 去重。
"""
from __future__ import annotations

import pandas as pd

from core.event_engine.jq.datalake.base import CNE_CURATED, JQDataPlugin

PLUGIN = JQDataPlugin(
    name="fina_indicator",
    description="财务指标(roe/eps/bps/毛利率等 112 列, ann_date 点时)",
    date_column="ann_date",
    entity_keys=("code",),
)

_COLS = ["symbol", "ann_date", "f_ann_date", "end_date", "eps", "dt_eps",
         "bps", "roe", "roe_waa", "roe_dt", "q_roe", "q_dt_roe",
         "gross_margin", "current_ratio", "quick_ratio",
         "ocfps", "cfps", "revenue_ps", "op_income",
         "q_ocf_to_sales", "ocf_to_debt", "netprofit_margin"]


def _iter_parquets() -> list[str]:
    base = CNE_CURATED / "fina_indicator"
    out: list[str] = []
    if base.exists():
        for part in sorted(base.iterdir()):
            if part.is_dir():
                out.extend(str(p) for p in sorted(part.glob("*.parquet")))
    return out


def load() -> pd.DataFrame:
    """全量合并, 按 (code, end_date) 去重保留最新公告, 按 ann_date 排序。"""
    keep = None
    frames = []
    for f in _iter_parquets():
        df = pd.read_parquet(f)
        if keep is None:
            keep = [c for c in _COLS if c in df.columns]
        frames.append(df[[c for c in keep if c in df.columns]])
    if not frames:
        return pd.DataFrame()
    fi = pd.concat(frames, ignore_index=True)
    fi["code"] = fi["symbol"].astype(str).str[:6]
    fi["ann_date"] = pd.to_datetime(fi["ann_date"], errors="coerce")
    fi["end_date"] = pd.to_datetime(fi["end_date"], errors="coerce")
    fi = fi.dropna(subset=["ann_date"])
    fi = fi.sort_values(["code", "end_date", "ann_date"], kind="stable")
    fi = fi.drop_duplicates(["code", "end_date"], keep="last")
    fi = fi.sort_values(["code", "ann_date"], kind="stable")
    return fi.reset_index(drop=True)
