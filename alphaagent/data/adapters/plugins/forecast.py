"""业绩预告插件：CNE curated `forecast`（PIT 对齐）。

以 ann_date（公告日）为 point-in-time 锚点，把业绩预告展开为日频阶跃序列
后 join 到核心行情 Panel。与 fundamental 插件同构（join_asof backward）。

设计要点：
- PIT：公告日 D 当天收盘后才知道的预告，D 当天即视为可用（预告通常在盘后
  发布，面板列以当日值表示"已知的最近预告"，评估引擎以 close-to-close
  label 使用，无未来函数；若需更保守可在 DSL 用 DELAY($pred_*, 1)）。
- 多次预告（同一报告期修正）：每次公告都是独立事件，asof 取最近一次。
- 类型编码：中文预告类型（预增/预减/扭亏/首亏/续亏/略增/略减/不确定）
  映射为方向数值 pred_direction ∈ {+1, -1, 0}，面板列统一为 float。
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

# ── 列映射：衍生日频列（load() 内部已完成全部计算，identity 映射）─────────

_FORECAST_COLS = {
    "pred_direction": "pred_direction",
    "pred_change_mid": "pred_change_mid",
    "pred_net_profit_mid": "pred_net_profit_mid",
    "pred_surprise": "pred_surprise",
    "pred_days_since": "pred_days_since",
}

# 预告类型 → 方向编码
_TYPE_DIRECTION: dict[str, float] = {
    "预增": 1.0, "略增": 1.0, "扭亏": 1.0, "续盈": 1.0,
    "预减": -1.0, "略减": -1.0, "首亏": -1.0, "续亏": -1.0, "增亏": -1.0,
    "不确定": 0.0,
}

_ALL_FORECAST_COLS = list(_FORECAST_COLS.values())

PLUGIN = DataSourcePlugin(
    name="forecast",
    dataset="forecast",  # 虚拟名；实际读 curated/forecast
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map=_FORECAST_COLS,
    priority=31,
)


# ── 加载函数 ──────────────────────────────────────────────────────────

def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """加载业绩预告并 PIT 展开为日频阶跃序列。"""
    root = _curated_root() / "forecast"
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("CNE curated forecast 无 parquet 文件")
    raw = pl.read_parquet(root, hive_partitioning=False)

    # 方向编码：优先用预告类型；type 缺失时按变动区间符号推断
    # （min/max 同号为正 → +1，同号为负 → -1，跨零/缺失 → None）
    events = (
        raw.with_columns([
            pl.col("type")
            .replace(_TYPE_DIRECTION, default=None)
            .cast(pl.Float64, strict=False)
            .alias("pred_direction"),
            ((pl.col("p_change_min") + pl.col("p_change_max")) / 2.0)
            .alias("pred_change_mid"),
            ((pl.col("net_profit_min") + pl.col("net_profit_max")) / 2.0)
            .alias("pred_net_profit_mid"),
        ])
        .with_columns(
            pl.col("pred_direction").fill_null(
                pl.when(
                    pl.col("p_change_min").is_not_null()
                    & pl.col("p_change_max").is_not_null()
                    & (pl.col("p_change_min") > 0) & (pl.col("p_change_max") > 0)
                ).then(1.0)
                .when(
                    pl.col("p_change_min").is_not_null()
                    & pl.col("p_change_max").is_not_null()
                    & (pl.col("p_change_min") < 0) & (pl.col("p_change_max") < 0)
                ).then(-1.0)
                .otherwise(None)
            )
        )
        .with_columns(
            pl.when(
                pl.col("pred_net_profit_mid").is_not_null()
                & pl.col("last_parent_net").is_not_null()
                & (pl.col("last_parent_net").abs() > 1e-9)
            )
            .then(pl.col("pred_net_profit_mid") / pl.col("last_parent_net") - 1.0)
            .otherwise(None)
            .alias("pred_surprise")
        )
        # 同一公告日多条（多报告期修正同日发布）取最后一条
        # （pred_days_since 在 asof 之后才计算，不参与聚合）
        .sort(["symbol", "ann_date", "end_date"])
        .group_by(["symbol", "ann_date"])
        .agg([pl.col(c).last() for c in _ALL_FORECAST_COLS if c != "pred_days_since"])
        .filter(pl.col("ann_date").is_not_null())
    )

    if events.height == 0:
        raise ValueError("forecast: 无有效预告记录")

    e = datetime.date.fromisoformat(end) if end else datetime.date.today()
    s = datetime.date.fromisoformat(start) if start else datetime.date(2015, 1, 1)
    events = events.filter(pl.col("ann_date") <= e)
    if start:
        start_d = datetime.date.fromisoformat(start)
        # 保留 start 前最近一次预告（作为初始状态）
        cutoff = start_d - datetime.timedelta(days=800)
        events = events.filter(pl.col("ann_date") >= cutoff)
    if events.height == 0:
        raise ValueError(f"forecast: 过滤后无数据 (start={start}, end={end})")

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
        events.sort(["symbol", "ann_date"]),
        left_on="date",
        right_on="ann_date",
        by="symbol",
        strategy="backward",
    ).with_columns(
        # 距最近一次预告的天数（自然日）；从未预告过的股票为 NaN
        (pl.col("date") - pl.col("ann_date"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("pred_days_since")
    ).filter(
        # 首条预告之前的行全为 NaN，丢弃（与 fundamental 的非空过滤同理）
        pl.col("pred_days_since").is_not_null()
    )

    result = expanded.select(
        [pl.col("date").alias("trade_date"), pl.col("symbol").alias("ts_code"),
         *_ALL_FORECAST_COLS]
    )
    pdf = result.to_pandas()
    logger.info("forecast adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
