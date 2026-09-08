# -*- coding: utf-8 -*-
"""事件面数据插件：现金分红（实施公告口径，PIT 日频）。

数据源：``data/pg_parquet/dividend.parquet``（tushare 分红送配全历史，1991 起）。
curated/dividend 目前只保留近期 tip（2026-08 起），历史分红在主仓 pg_parquet。

分红事件在 A 股走 预案 → 股东大会 → 实施公告 多个阶段，只有**实施公告**
（``div_proc = "实施"``、带 ``imp_ann_date``/``ex_date``）才同时确定除息日和
每股现金额，因此本插件以 ``imp_ann_date``（实施公告日）为 PIT 锚点：
- 公告前任何字段均不可见（无未来函数）；
- ``div_cash_div`` 为最近一份已公告实施的每股现金股利（税前，元/股），
  公告后保持为最新值直到下一份分红公告（约年频）；
- ``div_days_to_ex`` 仅在 [实施公告日, 除息日) 窗口内有效，= 距除息日自然日
  倒数（除息日后为空）——供"即将除息/分红抢权"类事件因子使用；
- ``div_days_since_ann`` = 距最近一次实施公告的自然日。

注意：送转比例（stk_div/stk_bo_rate）源口径为每 10 股比例、且部分行缺失，
本插件先不接；股息率类因子可在 DSL 内用 ``$div_cash_div`` 与价格/市值组合。
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

# alphaagent/data/adapters/plugins/dividend.py → quant_ui 根
_QUANT_UI_ROOT = Path(__file__).resolve().parents[4]
_DIVIDEND_PG = _QUANT_UI_ROOT / "data" / "pg_parquet" / "dividend.parquet"

_ALL_DIV_COLS = ["div_cash_div", "div_days_to_ex", "div_days_since_ann"]

PLUGIN = DataSourcePlugin(
    name="dividend",
    dataset="dividend",
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map={c: c for c in _ALL_DIV_COLS},
    priority=40,
)


def _implementation_events() -> pl.DataFrame:
    """已实施分红（每股现金 + 实施公告日 + 除息日），按 (symbol, 公告日) 去重。"""
    if not _DIVIDEND_PG.is_file():
        raise FileNotFoundError(
            f"dividend: 缺少 {_DIVIDEND_PG}（curated/dividend 仅有近期 tip，"
            "历史分红请用 pg_parquet 全历史文件）"
        )
    raw = pl.read_parquet(_DIVIDEND_PG).select(
        ["ts_code", "end_date", "div_proc", "cash_div", "imp_ann_date", "ex_date"]
    )
    if raw.is_empty():
        raise ValueError("dividend: 无数据")

    impl = raw.filter(
        (pl.col("div_proc") == "实施")
        & pl.col("cash_div").is_not_null()
        & pl.col("imp_ann_date").is_not_null()
        & pl.col("ex_date").is_not_null()
    )
    if impl.is_empty():
        raise ValueError("dividend: 无已实施记录")
    impl = (
        impl.sort(["ts_code", "end_date", "imp_ann_date"])
        .unique(subset=["ts_code", "end_date", "imp_ann_date"], keep="last")
        .with_columns(pl.col("cash_div").cast(pl.Float64))
    )
    return impl.rename({"ts_code": "symbol"})


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """加载已实施分红，按实施公告日展开为日频（含除息倒计时窗口）。"""
    impl = _implementation_events()
    step_events = (
        impl.sort(["symbol", "imp_ann_date"])
        .unique(subset=["symbol", "imp_ann_date"], keep="last")
        .select(["symbol", "imp_ann_date", "cash_div"])
        .rename({"imp_ann_date": "_pit_date", "cash_div": "div_cash_div"})
    )

    base = _pitlib.expand_pit_daily(
        step_events,
        start=start,
        end=end,
        anchor="_pit_date",
        since_col="div_days_since_ann",
        value_cols=["div_cash_div"],
    )

    # ── 除息倒计时窗口：[imp_ann_date, ex_date) 内逐自然日 ──
    s, e = _pitlib.parse_window(start, end)
    windows = impl.filter(
        (pl.col("imp_ann_date") >= s - datetime.timedelta(days=_pitlib.PIT_LOOKBACK_DAYS))
        & (pl.col("imp_ann_date") <= e)
        & (pl.col("ex_date") > pl.col("imp_ann_date"))
    )
    rows: list[tuple] = []
    for row in windows.iter_rows(named=True):
        d = row["imp_ann_date"]
        ex = row["ex_date"]
        while d < ex:
            rows.append((row["symbol"], d, (ex - d).days))
            d += datetime.timedelta(days=1)
    if rows:
        ex_windows = (
            pl.DataFrame(
                rows,
                schema={"symbol": pl.Utf8, "trade_date": pl.Date, "_left": pl.Int32},
                orient="row",
            )
            .rename({"symbol": "ts_code"})
            .filter(pl.col("trade_date") >= s)
        )
        base = base.join(
            ex_windows, on=["ts_code", "trade_date"], how="left", coalesce=True
        )
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.Int32).alias("_left"))

    base = base.with_columns(
        pl.col("_left").cast(pl.Float64).alias("div_days_to_ex")
    ).drop("_left")

    pdf = base.to_pandas()
    logger.info("dividend adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
