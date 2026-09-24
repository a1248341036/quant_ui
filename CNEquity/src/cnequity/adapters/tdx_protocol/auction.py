"""TDX 0x056A 集合竞价过程快照 → auction_series schema.

The wire reports virtual matched/unmatched volume in 手; the lake's unit
contract is 股 everywhere (cnequity.domain.units), so the adapter multiplies
by 100. The series covers both the opening (09:15-09:25) and closing
(14:57-15:00) auction process; ``session`` splits them.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime

import polars as pl

from cnequity.domain.rate_limit import RateLimitSpec, wait_spec

logger = logging.getLogger(__name__)

_LOTS_TO_SHARES = 100

# 0x056A is served by the money-flow host group: eltdx routes auction queries
# to these 35 hosts (money_flow_servers.json) rather than the ordinary pool.
# The standard pool is tried first; these are the fallback for failed symbols.
AUCTION_FALLBACK_HOSTS = (
    "180.153.18.170",
    "58.34.106.207",
    "116.128.207.164",
    "120.253.221.207",
    "202.108.254.67",
    "8.160.37.6",
    "60.191.117.167",
    "115.238.56.198",
    "218.75.126.9",
    "115.238.90.165",
    "183.131.224.21",
    "183.131.224.27",
    "60.12.136.250",
    "120.199.2.122",
    "120.199.2.123",
    "117.149.2.68",
    "117.149.2.70",
    "219.146.254.27",
    "221.0.195.48",
    "111.15.15.43",
    "121.33.228.164",
    "210.21.65.136",
    "120.196.72.45",
    "218.17.146.5",
    "101.52.238.101",
    "123.88.147.5",
    "114.141.177.44",
    "123.125.108.103",
    "45.116.35.251",
    "58.67.221.146",
    "182.118.8.4",
    "42.177.92.37",
    "27.151.2.37",
    "183.201.231.84",
    "183.242.231.6",
)


def _market_for(symbol: str) -> int:
    _, _, exch = symbol.partition(".")
    return 1 if exch == "SH" else (0 if exch == "SZ" else 2)


def fetch_auction_series(
    client,
    symbol: str,
    *,
    trade_date: date,
    rate_limit: RateLimitSpec | None = None,
) -> pl.DataFrame:
    """集合竞价过程快照 for one symbol on *trade_date* (0x056A)."""
    wait_spec(rate_limit)
    code, _, _ = symbol.partition(".")
    stamp = int(trade_date.strftime("%Y%m%d"))
    try:
        raw = client.auction_series(symbol=code, market=_market_for(symbol), on_date=stamp)
    except Exception as exc:
        logger.debug("TDX auction failed for %s: %s", symbol, exc)
        return pl.DataFrame()

    rows: list[dict] = []
    for rec in raw or []:
        minute_of_day = int(rec["minute_of_day"])
        second = int(rec["second"])
        price = float(rec["price"])
        if not math.isfinite(price):
            continue
        matched = int(rec["matched_volume"]) * _LOTS_TO_SHARES
        unmatched_signed = int(rec["unmatched_signed"]) * _LOTS_TO_SHARES
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "auction_time": datetime(
                    trade_date.year,
                    trade_date.month,
                    trade_date.day,
                    minute_of_day // 60,
                    minute_of_day % 60,
                    second,
                ),
                "session": "open" if minute_of_day < 12 * 60 else "close",
                "price": price,
                "matched_volume": matched,
                "unmatched_volume": abs(unmatched_signed),
                "unmatched_direction": (
                    1 if unmatched_signed > 0 else (-1 if unmatched_signed < 0 else 0)
                ),
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)