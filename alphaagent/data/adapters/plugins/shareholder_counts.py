"""股东人数插件：CNE curated `shareholder_counts`（PIT 对齐）。

股东户数是筹码集中度的经典代理：户数下降 = 筹码向大资金集中。
数据为不定期披露（季报/半年报/年报附注，或公司自愿披露），以
announce_date（公告日）为 PIT 锚点展开为日频阶跃序列。

注意：count_date 是统计截止日，公告可能滞后数周；必须以 announce_date
为可用时点（PIT），而非 count_date。
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd
import polars as pl

from alphaagent.data.adapters.plugins.fundamental import _curated_root
from alphaagent.data.adapters.registry import DataSourcePlugin

logger = logging.getLogger(__name__)

# ── 列映射：衍生日频列（identity 映射，load() 内完成计算）──────────────

_HOLDERS_COLS = {
    "holder_count": "holder_count",
    "holder_count_chg_pct": "holder_count_chg_pct",
    "holder_avg_float_shares": "holder_avg_float_shares",
    "holder_avg_value": "holder_avg_value",
    "holder_days_since": "holder_days_since",
}

_ALL_HOLDER_COLS = list(_HOLDERS_COLS.values())

PLUGIN = DataSourcePlugin(
    name="shareholder_counts",
    dataset="shareholder_counts",  # 虚拟名；实际读 curated/shareholder_counts
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map=_HOLDERS_COLS,
    priority=32,
)


# ── 加载函数 ──────────────────────────────────────────────────────────

def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """加载股东人数并 PIT 展开为日频阶跃序列。"""
    root = _curated_root() / "shareholder_counts"
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("CNE curated shareholder_counts 无 parquet 文件")
    raw = pl.read_parquet(root, hive_partitioning=False)

    # PIT 锚点 = announce_date（公告日），缺公告日的记录用 count_date 兜底；
    # 原始列名 → 面板列名（holder_count_change_pct → holder_count_chg_pct 等）
    events = (
        raw.rename({
            "holder_count_change_pct": "holder_count_chg_pct",
            "avg_float_shares": "holder_avg_float_shares",
            "avg_holding_value": "holder_avg_value",
        })
        .with_columns(
            pl.coalesce(["announce_date", "count_date"]).alias("_pit_date"),
        )
        .sort(["symbol", "_pit_date", "count_date"])
        .group_by(["symbol", "_pit_date"])
        .agg([pl.col(c).last() for c in _ALL_HOLDER_COLS if c != "holder_days_since"])
        .filter(pl.col("_pit_date").is_not_null())
    )
    # 兜底：个别记录 announce_date/count_date 双缺
    events = events.filter(pl.col("_pit_date").is_not_null())
    if events.height == 0:
        raise ValueError("shareholder_counts: 无有效记录")

    e = datetime.date.fromisoformat(end) if end else datetime.date.today()
    s = datetime.date.fromisoformat(start) if start else datetime.date(2015, 1, 1)
    events = events.filter(pl.col("_pit_date") <= e)
    if start:
        start_d = datetime.date.fromisoformat(start)
        cutoff = start_d - datetime.timedelta(days=800)
        events = events.filter(pl.col("_pit_date") >= cutoff)
    if events.height == 0:
        raise ValueError(f"shareholder_counts: 过滤后无数据 (start={start}, end={end})")

    symbols = events["symbol"].unique().to_list()

    all_days: list[datetime.date] = []
    d = s
    while d <= e:
        if d.weekday() < 5:
            all_days.append(d)
        d += datetime.timedelta(days=1)

    grid = (
        pl.DataFrame({"date": pl.Series(all_days, dtype=pl.Date)})
        .join(pl.DataFrame({"symbol": pl.Series(symbols, dtype=pl.Utf8)}), how="cross")
        .sort(["symbol", "date"])
    )

    expanded = grid.join_asof(
        events.sort(["symbol", "_pit_date"]),
        left_on="date",
        right_on="_pit_date",
        by="symbol",
        strategy="backward",
    ).with_columns(
        (pl.col("date") - pl.col("_pit_date"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("holder_days_since")
    ).filter(
        # 首次公告之前的行全为 NaN，丢弃
        pl.col("holder_days_since").is_not_null()
    )

    result = expanded.select(
        [pl.col("date").alias("trade_date"), pl.col("symbol").alias("ts_code"),
         *_ALL_HOLDER_COLS]
    )
    pdf = result.to_pandas()
    logger.info("shareholder_counts adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
