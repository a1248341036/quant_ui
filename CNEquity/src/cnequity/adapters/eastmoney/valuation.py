"""EastMoney valuation metrics (PE/PB/PS/market cap)."""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from cnequity.adapters.eastmoney.clist import clist_rows_to_symbols, fetch_clist_pages
from cnequity.adapters.eastmoney.common import _to_float
from cnequity.adapters.eastmoney.em_auth import EastMoneyClient

logger = logging.getLogger(__name__)

_VALUATION_FIELDS = "f12,f13,f9,f23,f45,f20,f21"


def fetch_valuation_metrics(
    trade_date: date,
    *,
    client: EastMoneyClient | None = None,
    config=None,
) -> pl.DataFrame:
    owns = client is None
    if client is None:
        client = EastMoneyClient(config=config)
    try:
        rows_raw = fetch_clist_pages(client, fields=_VALUATION_FIELDS)
        mapped_rows = clist_rows_to_symbols(rows_raw)
        skipped = len(rows_raw) - len(mapped_rows)
        if skipped:
            logger.warning(
                "EastMoney valuation_metrics clist skipped "
                f"{skipped} unmappable security row(s)"
            )
        rows = []
        for sym, item in mapped_rows:
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": trade_date,
                    "pe_ttm": _to_float(item.get("f9")),
                    "pb": _to_float(item.get("f23")),
                    "ps_ttm": _to_float(item.get("f45")),
                    "total_mv": _to_float(item.get("f20")),
                    "float_mv": _to_float(item.get("f21")),
                }
            )
    finally:
        if owns:
            client.close()
    return (
        pl.DataFrame(rows).unique(subset=["symbol", "trade_date"], keep="last")
        if rows
        else pl.DataFrame()
    )


def fetch_valuation_metrics_tushare(
    trade_date: date,
    *,
    universe: set[str] | None = None,
    config=None,
) -> pl.DataFrame:
    """Tushare ``daily_basic`` fallback for the daily valuation snapshot.

    EastMoney's clist is the primary source, but it is WAF-blocked for some
    egress routes (``Server disconnected``).  Tushare's ``daily_basic`` returns
    a whole-market per-day valuation row (pe_ttm/pb/ps_ttm/total_mv/circ_mv)
    through the same middleware the tushare_wide_daily step already uses, so
    this is a genuinely independent source with a working rate limiter.

    Column mapping keeps the curated schema honest: ``float_mv`` comes from
    Tushare's ``circ_mv`` (circulating market cap), which is the closest match
    to EastMoney's f21.  Units: EastMoney f20/f21 are 元, but Tushare
    total_mv/circ_mv are 万元 — so they are ×1e4 to keep both sources
    comparable in the same schema (consistent with tushare_capital.py).
    Rows outside ``universe`` are dropped to stay in lock-step with daily_bars
    (audit: valuation_bars_orphan_symbol).
    """
    from cnequity.external.tushare_fetch import _fetch_with_retry, _get_pro

    pro = _get_pro(config)
    df = _fetch_with_retry(
        pro,
        "daily_basic",
        interval=config.external_tushare_wide_interval,
        trade_date=trade_date.strftime("%Y%m%d"),
    )
    if df.is_empty():
        return pl.DataFrame()

    # 确保列存在（mock 数据可能缺列），缺失则补空
    for col, dtype in (("trade_date", pl.Utf8), ("ts_code", pl.Utf8),
                       ("pe_ttm", pl.Float64), ("pb", pl.Float64),
                       ("ps_ttm", pl.Float64), ("total_mv", pl.Float64),
                       ("circ_mv", pl.Float64)):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # trade_date 从 tushare 返回的是 "20260924" 字符串，需显式 strptime；
    # 若已是 Date 类型（如测试 mock）则直接保留
    if df.schema["trade_date"] == pl.Date:
        pass
    else:
        df = df.with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8, strict=False)
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
            .alias("trade_date"),
        )
    df = df.with_columns(
        pl.col("ts_code").cast(pl.Utf8, strict=False).str.replace_all(r"\..*$", ""),
        pl.col("pe_ttm").cast(pl.Float64, strict=False),
        pl.col("pb").cast(pl.Float64, strict=False),
        pl.col("ps_ttm").cast(pl.Float64, strict=False),
        # Tushare total_mv/circ_mv are 万元; the curated lake stores 元 → ×1e4.
        (pl.col("total_mv").cast(pl.Float64, strict=False) * 10000).alias("total_mv"),
        (pl.col("circ_mv").cast(pl.Float64, strict=False) * 10000).alias("float_mv"),
    )
    if "ts_code" not in df.columns or "trade_date" not in df.columns:
        return pl.DataFrame()

    # Map ts_code (e.g. "600519") back to the canonical 6-digit.suffix symbol.
    df = df.with_columns(
        pl.col("ts_code")
        .map_elements(
            lambda code: _code_to_symbol(code) if code else None,
            return_dtype=pl.Utf8,
        )
        .alias("symbol")
    )
    df = df.filter(pl.col("symbol").is_not_null())
    if universe:
        df = df.filter(pl.col("symbol").is_in(list(universe)))
    if df.is_empty():
        return pl.DataFrame()
    return (
        df.select([
            "symbol",
            "trade_date",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "total_mv",
            "float_mv",
        ])
        .unique(subset=["symbol", "trade_date"], keep="last")
        .with_columns(pl.lit("tushare").alias("source"))
    )


def _code_to_symbol(code: str) -> str | None:
    """6-digit Tushare code → canonical symbol via exchange inference."""
    from cnequity.domain.symbols import format_symbol, infer_exchange_from_code, is_all_a_symbol

    code = str(code).strip().zfill(6)
    if len(code) != 6 or not code.isdigit():
        return None
    exchange = infer_exchange_from_code(code)
    if not is_all_a_symbol(code, exchange):
        return None
    return format_symbol(code, exchange)
