"""TDX daily bars with pagination beyond the 800-bar API limit.

TDX reports daily-K ``vol`` in 手; the lake stores 股 (see
:mod:`cnequity.domain.units`), so the stock path multiplies by 100 here, at
the boundary. Measured over 12,182,204 curated rows, ``amount / close / vol``
had a median of 100.000 before the conversion — a lot, not a share.

The index path deliberately does **not** convert. ``client.index()`` is a
different wire call from ``client.bars()``, and its ``vol`` does not reconcile
against the sum of its constituents at any power of 100 (checked on
000001.SH: index amount is 77% of the SH stock-sum amount, but the volumes are
~300× apart, which no shares/lots reading explains). Until that unit is
pinned down, ``index_bars`` and ``sector_bars`` keep the value TDX sent and
their own contract; scaling it on a guess would only move the break.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable
from datetime import date

import polars as pl

from cnequity.adapters.numeric import finite_int64
from cnequity.adapters.tdx_protocol._decode import decoded_quantity
from cnequity.domain.rate_limit import RateLimitSpec, wait_spec
from cnequity.domain.units import lots_to_shares

logger = logging.getLogger(__name__)

_PAGE_SIZE = 800
_MAX_PAGES = 1000


class TdxBarsPaginationError(RuntimeError):
    """Raised when a TDX bars page fails and the caller requires complete history."""


def _date_column(pdf: pl.DataFrame) -> str:
    return "datetime" if "datetime" in pdf.columns else "date"


def _coerce_date(val) -> date:
    if isinstance(val, date):
        return val
    if hasattr(val, "date"):
        return val.date()
    if isinstance(val, str):
        return date.fromisoformat(val[:10])
    raise TypeError(f"unsupported bar date value: {val!r}")


def _page_min_date(pdf: pl.DataFrame) -> date | None:
    col = _date_column(pdf)
    if col not in pdf.columns or pdf.is_empty():
        return None
    series = pdf[col]
    if series.dtype == pl.Date:
        return series.min()
    mins: list[date] = []
    for val in series:
        if val is None:
            continue
        try:
            mins.append(_coerce_date(val))
        except (TypeError, ValueError, OverflowError):
            continue
    return min(mins) if mins else None


def _parse_bar_rows(
    pdf: pl.DataFrame,
    sym: str,
    start: date,
    end: date,
    *,
    volume_in_lots: bool = True,
) -> list[dict]:
    date_col = _date_column(pdf)
    rows: list[dict] = []
    for row in pdf.iter_rows(named=True):
        try:
            td = _coerce_date(row[date_col])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if td < start or td > end:
            continue
        try:
            open_ = float(row.get("open", 0))
            high = float(row.get("high", 0))
            low = float(row.get("low", 0))
            close = float(row.get("close", 0))
            volume = decoded_quantity(row.get("volume", row.get("vol", 0)))
            amount = decoded_quantity(row.get("amount", 0))
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in (open_, high, low, close, volume, amount)):
            continue
        try:
            raw_volume = finite_int64(
                volume,
                minimum=0,
                maximum=(2**63 - 1) // 100 if volume_in_lots else 2**63 - 1,
            )
        except ValueError:
            continue
        rows.append(
            {
                "symbol": sym,
                "trade_date": td,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": lots_to_shares(raw_volume) if volume_in_lots else raw_volume,
                "amount": amount,
            }
        )
    return rows


def fetch_bars_paginated(
    client,
    sym: str,
    start: date,
    end: date,
    *,
    rate_limit: RateLimitSpec | None = None,
    backfill: bool = False,
    on_page: Callable[[], None] | None = None,
    is_index: bool = False,
) -> list[dict]:
    """Fetch daily bars for *sym* in [start, end], paging through TDX history.

    Indices must use the ``index()`` call — ``bars()`` with a stock
    market id returns corrupt datetimes for index codes (e.g. 399001.SZ).

    Stock rows come back with ``volume`` in 股; index rows keep TDX's own unit.
    See the module docstring for why the two differ.
    """
    code, exch = sym.split(".")
    market = 1 if exch == "SH" else (0 if exch == "SZ" else 2)
    offset_pos = 0
    all_rows: list[dict] = []
    seen_pages: set[str] = set()
    page_count = 0

    while True:
        page_count += 1
        if page_count > _MAX_PAGES:
            raise TdxBarsPaginationError(
                f"TDX bars pagination exceeded {_MAX_PAGES} pages for {sym}"
            )
        wait_spec(rate_limit)
        try:
            if is_index:
                raw = client.index(
                    symbol=code,
                    frequency=9,
                    start=offset_pos,
                    offset=_PAGE_SIZE,
                )
            else:
                raw = client.bars(
                    symbol=code,
                    frequency=9,
                    market=market,
                    start=offset_pos,
                    offset=_PAGE_SIZE,
                )
        except Exception as exc:
            # A page is part of the requested symbol/window contract. Returning
            # earlier pages as a successful incremental fetch silently advances
            # the watermark past the missing tail, so every page failure must
            # propagate to the batch/failover layer.
            raise TdxBarsPaginationError(
                f"TDX bars page failed for {sym} at start={offset_pos}"
            ) from exc

        if raw is None or len(raw) == 0:
            break

        if isinstance(raw, pl.DataFrame):
            pdf = raw
        elif hasattr(raw, "columns"):
            pdf = pl.from_pandas(raw)
        else:
            pdf = pl.DataFrame(raw)

        page_signature = hashlib.sha256(repr(pdf.to_dicts()).encode()).hexdigest()
        if page_signature in seen_pages:
            raise TdxBarsPaginationError(
                f"TDX bars pagination did not advance for {sym} at start={offset_pos}"
            )
        seen_pages.add(page_signature)

        page_rows = _parse_bar_rows(pdf, sym, start, end, volume_in_lots=not is_index)
        if page_rows:
            all_rows.extend(page_rows)

        page_min = _page_min_date(pdf)
        if page_min is not None and page_min < start:
            break

        if len(pdf) < _PAGE_SIZE:
            break
        offset_pos += _PAGE_SIZE
        if on_page is not None:
            on_page()

    if not all_rows:
        return []

    df = pl.DataFrame(all_rows).unique(subset=["symbol", "trade_date"], keep="last")
    return df.sort("trade_date").to_dicts()
