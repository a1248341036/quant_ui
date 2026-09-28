# -*- coding: utf-8 -*-
"""财报明细项插件：CNE curated ``financial_statement_items``（长表 → PIT 宽列）。

源为长表（``symbol``/``report_period``/``statement_type``/``item_code``/``item_value``/
``announce_date``），本插件挑选 10 个高频明细项透视成日频阶跃面板列（单位：元）：

| 面板列 | 源 (statement_type, item_code) |
|---|---|
| ``fsi_revenue`` | income, revenue |
| ``fsi_net_profit`` | income, net_profit |
| ``fsi_operating_profit`` | income, operating_profit |
| ``fsi_total_assets`` | balance, total_assets |
| ``fsi_total_liabilities`` | balance, total_liabilities |
| ``fsi_total_equity`` | balance, total_equity |
| ``fsi_monetary_funds`` | balance, monetary_funds |
| ``fsi_fixed_assets`` | balance, fixed_assets |
| ``fsi_net_cash_operate`` | cashflow, net_cash_operate |
| ``fsi_capex`` | cashflow, capex |

**PIT 锚点是 ``announce_date``**（实际公告日），不是报告期期末。
**口径注意**：利润表/现金流量表为**年初至今累计**（Q2 行 = 上半年累计），跨季比较须用
同比（`DELTA(..., 252)`）；资产负债表为期末时点值。
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

from . import _pitlib

logger = logging.getLogger(__name__)

# 源 (statement_type, item_code) → 面板列名（item_code 取值已按 curated 实测核对）
_ITEM_MAP: dict[tuple[str, str], str] = {
    ("income", "revenue"): "fsi_revenue",
    ("income", "net_profit"): "fsi_net_profit",
    ("income", "operating_profit"): "fsi_operating_profit",
    ("balance", "total_assets"): "fsi_total_assets",
    ("balance", "total_liabilities"): "fsi_total_liabilities",
    ("balance", "total_equity"): "fsi_total_equity",
    ("balance", "monetary_funds"): "fsi_monetary_funds",
    ("balance", "fixed_assets"): "fsi_fixed_assets",
    ("cashflow", "net_cash_operate"): "fsi_net_cash_operate",
    ("cashflow", "capex"): "fsi_capex",
}
_VALUE_COLS = list(_ITEM_MAP.values())

PLUGIN = DataSourcePlugin(
    name="financial_items",
    dataset="financial_statement_items",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={**{c: c for c in _VALUE_COLS}, "fsi_days_since": "fsi_days_since"},
    priority=63,
)


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> Any:
    """长表 → 按公告日 PIT 展开的日频宽列。"""
    raw = _pitlib.read_curated_pit("financial_statement_items", start=start, end=end)
    needed = {"symbol", "statement_type", "item_code", "item_value", "announce_date"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"financial_statement_items 缺列: {sorted(missing)}")

    wanted = pl.DataFrame(
        {
            "statement_type": [k[0] for k in _ITEM_MAP],
            "item_code": [k[1] for k in _ITEM_MAP],
        }
    )
    long = (
        raw.select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("statement_type").cast(pl.Utf8),
            pl.col("item_code").cast(pl.Utf8),
            pl.col("item_value").cast(pl.Float64, strict=False).alias("_value"),
            pl.col("announce_date").cast(pl.Date).alias("_pit_date"),
        )
        .filter(pl.col("_pit_date").is_not_null() & pl.col("_value").is_not_null())
        .join(wanted, on=["statement_type", "item_code"], how="semi")
        .drop("statement_type")
    )
    if long.is_empty():
        raise ValueError(f"financial_items: 窗口 {start}~{end} 内无目标明细项")

    # 同一 (symbol, 公告日) 出现多期/多项时：后到的行覆盖先到的（sort+unique keep=last）
    wide = (
        long.sort(["symbol", "_pit_date", "item_code"])
        .pivot(on="item_code", index=["symbol", "_pit_date"], values="_value", aggregate_function="last")
    )
    rename = {k[1]: v for k, v in _ITEM_MAP.items() if k[1] in wide.columns}
    wide = wide.rename(rename)
    for col in _VALUE_COLS:
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))

    out = _pitlib.expand_pit_daily(
        wide.select(["symbol", "_pit_date", *_VALUE_COLS]),
        start=start,
        end=end,
        anchor="_pit_date",
        since_col="fsi_days_since",
        value_cols=_VALUE_COLS,
        out_symbol="symbol",
    )
    pdf = out.to_pandas()
    logger.info("financial_items adapter: %d rows × %d cols (start=%s end=%s)", *pdf.shape, start, end)
    return pdf
