"""TDX 0x000F 股本变迁 → capital_changes schema.

Same wire command as the xdxr path, but this adapter keeps every category
(1..15) with the four raw fields decoded per eltdx's unit semantics: the
share-count categories (2/3/5/7/8/9/10) carry 万股 on the wire and are
multiplied by 10000 to 股; 增发新股 (6) is mixed (only c3 is 万股); the rest
are float32 values as-is (category 1 keeps the 每10股 dividend/rights ratios).
"""

from __future__ import annotations

import logging
import math
from datetime import date

import polars as pl

from cnequity.domain.rate_limit import RateLimitSpec, wait_spec

logger = logging.getLogger(__name__)

_SHARE_COUNT_CATEGORIES = frozenset({2, 3, 5, 7, 8, 9, 10})


def _market_for(symbol: str) -> int:
    _, _, exch = symbol.partition(".")
    return 1 if exch == "SH" else (0 if exch == "SZ" else 2)


def _values(
    category: int, c1: float, c2: float, c3: float, c4: float
) -> tuple[float, float, float, float]:
    if category in _SHARE_COUNT_CATEGORIES:
        return c1 * 10000, c2 * 10000, c3 * 10000, c4 * 10000
    if category == 6:
        return c1, c2, c3 * 10000, c4
    return c1, c2, c3, c4


def fetch_capital_changes(
    client,
    symbol: str,
    *,
    rate_limit: RateLimitSpec | None = None,
    on_date: date | None = None,
    strict: bool = False,
) -> pl.DataFrame:
    """股本变迁事件 for one symbol (0x000F); *on_date* filters client-side."""
    wait_spec(rate_limit)
    code, _, _ = symbol.partition(".")
    try:
        raw = client.capital_changes(symbol=code, market=_market_for(symbol))
    except Exception as exc:
        logger.debug("TDX capital changes failed for %s: %s", symbol, exc)
        if strict:
            raise RuntimeError(f"TDX capital changes failed for {symbol}") from exc
        return pl.DataFrame()
    if not raw:
        return pl.DataFrame()

    rows: list[dict] = []
    for rec in raw:
        date_raw = int(rec["date"])
        try:
            event_date = date(date_raw // 10000, (date_raw % 10000) // 100, date_raw % 100)
        except ValueError:
            continue
        category = int(rec["category"])
        c1, c2, c3, c4 = _values(
            category, float(rec["c1"]), float(rec["c2"]), float(rec["c3"]), float(rec["c4"])
        )
        if not all(math.isfinite(v) for v in (c1, c2, c3, c4)):
            continue
        rows.append(
            {
                "symbol": symbol,
                "event_date": event_date,
                "category": category,
                "category_name": rec["name"],
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "c4": c4,
            }
        )
    if on_date is not None:
        rows = [r for r in rows if r["event_date"] == on_date]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["symbol", "event_date", "category"], keep="last")