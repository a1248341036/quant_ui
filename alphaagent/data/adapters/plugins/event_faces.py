"""事件面插件：CNE curated `dragon_tiger`（龙虎榜）+ `block_trades`（大宗交易）。

两个数据面都是稀疏日事件（每天只有 3-5% 的股票上榜/成交），直接并入面板
覆盖率过低，因此在这里做**稠密化**：展开为全股票 × 工作日网格，计算

- 滚动窗口（90 个交易日）内事件次数 / 事件金额：无事件填 0（语义诚实：
  "过去 90 日没有大宗交易"），覆盖率 100%；
- 距最近一次事件天数 / 最近一次大宗折溢价率（join_asof）：从未发生过事件
  为 NaN，输出行按"近 250 个自然日内有事件或计数非零"裁剪，超期视为无事件。

dragon_tiger 每行是"某日某股因某原因上榜"，同日同股可能多条（多原因），
聚合到 (symbol, date) 粒度：上榜次数按原因条数计，净买入按金额求和。
block_trades 每行一笔大宗成交，同日多笔聚合求和。
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

_EVENT_FACE_COLS = {
    # 龙虎榜（金额单位：元）
    "dt_cnt_90d": "dt_cnt_90d",
    "dt_net_buy_90d": "dt_net_buy_90d",
    "dt_days_since": "dt_days_since",
    # 大宗交易（amount 单位：万元；premium_ratio 为相对收盘价折溢价率，-0.05 = 折价 5%）
    "bt_cnt_90d": "bt_cnt_90d",
    "bt_amt_90d": "bt_amt_90d",
    "bt_premium_last": "bt_premium_last",
    "bt_days_since": "bt_days_since",
}

_ALL_EVENT_COLS = list(_EVENT_FACE_COLS.values())

_ROLLING_ROWS = 90          # 滚动窗口行数（≈90 个交易日）
_EMIT_STALE_DAYS = 250      # 超过该天数无事件且计数为 0 的行不输出（视为无事件）

PLUGIN = DataSourcePlugin(
    name="event_faces",
    dataset="event_faces",  # 虚拟名；实际读 curated/dragon_tiger + block_trades
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map=_EVENT_FACE_COLS,
    priority=33,
)


# ── 内部工具 ──────────────────────────────────────────────────────────

def _read_events(name: str) -> pl.DataFrame:
    root = _curated_root() / name
    if not sorted(root.rglob("*.parquet")):
        raise FileNotFoundError(f"CNE curated {name} 无 parquet 文件")
    return pl.read_parquet(root, hive_partitioning=False)


def _stock_symbols() -> list[str]:
    """全量 A 股股票清单（instruments，asset_type=stock），稠密化网格的行域。"""
    try:
        inst = pl.read_parquet(_curated_root() / "instruments", hive_partitioning=False)
        return (
            inst.filter(pl.col("asset_type") == "stock")["symbol"]
            .unique()
            .to_list()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_faces: 读取 instruments 失败，退回事件股票域: %s", exc)
        return []


def _weekday_grid(start: datetime.date, end: datetime.date,
                  symbols: list[str]) -> pl.DataFrame:
    days: list[datetime.date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return (
        pl.DataFrame({"date": pl.Series(days, dtype=pl.Date)})
        .join(pl.DataFrame({"symbol": pl.Series(symbols, dtype=pl.Utf8)}), how="cross")
        .sort(["symbol", "date"])
    )


def _densify(
    grid: pl.DataFrame,
    events: pl.DataFrame,
    *,
    cnt_col: str,
    amt_col: str,
    since_col: str,
    amt_expr: pl.Expr | None = None,
) -> pl.DataFrame:
    """事件表 → 日频网格特征：滚动计数/金额（fill 0）+ 距最近事件天数。"""
    agg_cols = [pl.len().alias(cnt_col)]
    if amt_expr is not None:
        agg_cols.append(amt_expr.alias(amt_col))
    daily = (
        events.sort(["symbol", "trade_date"])
        .group_by(["symbol", "trade_date"])
        .agg(agg_cols)
    )
    merged = (
        grid.join(daily, left_on=["symbol", "date"],
                  right_on=["symbol", "trade_date"], how="left")
        .with_columns([
            pl.col(cnt_col).fill_null(0.0).cast(pl.Float64),
        ])
    )
    if amt_expr is not None:
        merged = merged.with_columns(pl.col(amt_col).fill_null(0.0).cast(pl.Float64))
    merged = merged.sort(["symbol", "date"]).with_columns(
        pl.col(cnt_col).rolling_sum(window_size=_ROLLING_ROWS).over("symbol").alias(cnt_col),
    )
    if amt_expr is not None:
        merged = merged.with_columns(
            pl.col(amt_col).rolling_sum(window_size=_ROLLING_ROWS).over("symbol").alias(amt_col),
        )
    # 距最近一次事件天数（asof）。polars 对 right_on 键列不加后缀：
    # 右表 trade_date 原名进入，聚合列带 suffix（如 dt_cnt_90d_dt_days_since）
    extra = [pl.col(amt_col)] if amt_expr is not None else []
    asof_src = daily.sort(["symbol", "trade_date"])
    merged = merged.join_asof(
        asof_src,
        left_on="date", right_on="trade_date", by="symbol",
        strategy="backward",
        suffix=f"_{since_col}",
    ).with_columns(
        (pl.col("date") - pl.col("trade_date"))
        .dt.total_days().cast(pl.Float64).alias(since_col)
    ).drop(
        ["trade_date", f"{cnt_col}_{since_col}"]
        + ([f"{amt_col}_{since_col}"] if amt_expr is not None else [])
    )
    return merged


# ── 加载函数 ──────────────────────────────────────────────────────────

def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """加载龙虎榜 + 大宗交易并稠密化为日频特征。"""
    s = datetime.date.fromisoformat(start) if start else datetime.date(2015, 1, 1)
    e = datetime.date.fromisoformat(end) if end else datetime.date.today()
    # 网格向左多铺 ~130 个自然日（≈90 个交易日），保证面板起点处滚动窗口已满
    grid_start = s - datetime.timedelta(days=130)
    # 事件只取网格前 60 日之前以后的部分（更早的事件对任何输出行都不再有贡献）
    evt_cutoff = grid_start - datetime.timedelta(days=60)

    symbols = _stock_symbols()
    if not symbols:
        raise ValueError("event_faces: 无可用股票清单")

    grid = _weekday_grid(grid_start, e, symbols)

    parts: list[pl.DataFrame] = []

    # ── 龙虎榜 ──
    try:
        dt = _read_events("dragon_tiger").select(
            ["symbol", "trade_date", "net_amount"]
        ).filter(pl.col("trade_date") >= evt_cutoff)
        parts.append(
            _densify(
                grid, dt,
                cnt_col="dt_cnt_90d", amt_col="dt_net_buy_90d",
                since_col="dt_days_since",
                amt_expr=pl.col("net_amount").sum(),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_faces: dragon_tiger 跳过: %s", exc)

    # ── 大宗交易 ──
    try:
        bt = _read_events("block_trades").select(
            ["symbol", "trade_date", "amount", "premium_ratio"]
        ).filter(pl.col("trade_date") >= evt_cutoff)
        # premium 列需保留最近一笔的折溢价（asof 语义），单独处理：
        # daily 聚合里带 last(premium_ratio)，_densify 的 asof 已把 daily 行带上
        bt = bt.with_columns(pl.col("amount").alias("_amt"))
        dens = _densify(
            grid, bt,
            cnt_col="bt_cnt_90d", amt_col="bt_amt_90d",
            since_col="bt_days_since",
            amt_expr=pl.col("amount").sum(),
        )
        # 最近一笔折溢价：再对 premium 做一次 asof（按事件表原序取 last）
        prem_daily = (
            bt.sort(["symbol", "trade_date", "premium_ratio"])
            .group_by(["symbol", "trade_date"])
            .agg(pl.col("premium_ratio").last())
            .sort(["symbol", "trade_date"])
        )
        dens = (
            dens.join_asof(
                prem_daily, left_on="date", right_on="trade_date",
                by="symbol", strategy="backward", suffix="_prem",
            )
            .with_columns(pl.col("premium_ratio").alias("bt_premium_last"))
            .drop(["trade_date", "premium_ratio"])
        )
        parts.append(dens)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_faces: block_trades 跳过: %s", exc)

    if not parts:
        raise ValueError("event_faces: 龙虎榜与大宗交易均无数据")

    merged = parts[0]
    for p in parts[1:]:
        merged = merged.join(p, on=["symbol", "date"], how="full", coalesce=True)

    for c in _ALL_EVENT_COLS:
        if c not in merged.columns:
            merged = merged.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))

    # 裁剪输出行：有事件痕迹（计数非零 / 250 日内有事件）才输出，
    # 长期无事件的行丢弃——计数语义仍为 0（缺失即 0），days_since 超期视为无事件。
    merged = merged.filter(
        (pl.col("dt_cnt_90d") > 0) | (pl.col("bt_cnt_90d") > 0)
        | pl.col("dt_days_since").is_not_null() & (pl.col("dt_days_since") <= _EMIT_STALE_DAYS)
        | pl.col("bt_days_since").is_not_null() & (pl.col("bt_days_since") <= _EMIT_STALE_DAYS)
    ).filter(pl.col("date") >= s)

    result = merged.select(
        [pl.col("date").alias("trade_date"), pl.col("symbol").alias("ts_code"),
         *_ALL_EVENT_COLS]
    )
    pdf = result.to_pandas()
    logger.info("event_faces adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
