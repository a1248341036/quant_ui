"""Tushare 风险警示板（``stock_st``）：``trading_status`` 历史 ST 的快速回填源。

与 ``adapters/baostock/st_history.py`` 的差别在**查询粒度**：

- baostock 逐票扫（一票一次调用覆盖整窗，5557 票 + 免费层限速 ≈ 11 小时）；
- 中间件 ``stock_st`` 返回**按日全市场名单**，一次区间查询 + offset 翻页即可覆盖
  全历史（2016→今约 44 万行 / ~440 次调用 / ~15 分钟）。

证据语义：整窗名单里没有某只票 = 该票在该窗口内从未 ST。这与逐票扫的"零行响应"
等价（都在完整覆盖的前提下成立），所以调用方对零行标的记 0 条证据即可。
"""

from __future__ import annotations

from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.external.tushare_fetch import _get_pro, fetch_range_paged

# trading_status 的行契约（provenance 由 write_fetched 补）。
_OUTPUT_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "is_trading": pl.Boolean,
    "status": pl.Utf8,
}


def fetch_st_history(
    start: date,
    end: date,
    *,
    symbols: list[str] | None = None,
    config: Config | None = None,
) -> pl.DataFrame:
    """整窗全市场 ST 名单 → ``trading_status`` 行（``status='st'``）。

    ``symbols`` 非空时只保留这些标的（与既有 scope 对齐）；名单里没有的标的不会
    出现在结果里，调用方据此记 0 行证据。
    """
    if config is None:
        raise ValueError("Tushare ST 回填需要 config（token / interval 来源）")

    pro = _get_pro(config)
    raw = fetch_range_paged(
        pro,
        "stock_st",
        interval=config.external_tushare_wide_interval,
        start=start,
        end=end,
    )
    if raw.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    frame = raw.rename({"ts_code": "symbol"}) if "ts_code" in raw.columns else raw
    if symbols:
        frame = frame.filter(pl.col("symbol").is_in(sorted(set(symbols))))
    if frame.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    return (
        frame.select(
            pl.col("symbol").cast(pl.Utf8),
            # 中间件把 trade_date 返回成 "YYYYMMDD" 字符串（有时是 "YYYY-MM-DD"），
            # 直接 cast(Date) 会报 InvalidOperationError —— 先归一化成紧凑数字再解析。
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.replace_all("-", "")
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
            .alias("trade_date"),
            pl.lit(True).alias("is_trading"),
            pl.lit("st").alias("status"),
        )
        .unique(subset=["symbol", "trade_date"], keep="last")
        .sort(["trade_date", "symbol"])
    )
