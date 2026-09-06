"""EastMoney regulatory / compliance events (filtered from the notice stream).

Same extraction semantics as ``adapters.cninfo.regulatory`` — the day's full
announcement list, keyword-filtered and classified — but paged through the
EastMoney notice API (100 rows/page, deep pagination, 0.5s pacing) instead
of CNINFO's 30-row-per-exchange-column endpoint. The event stream is the
same universe of exchange/filings announcements, so daily keyword-hit
counts are comparable between the two sources.
"""

from __future__ import annotations

import logging
import re
from datetime import date

import polars as pl

from cnequity.adapters.cninfo.regulatory import _classify_event, _KEYWORD_TYPES
from cnequity.adapters.eastmoney.announcements import (
    _iter_announcement_items,
    _symbol_from_em_code,
)
from cnequity.adapters.eastmoney.em_auth import EastMoneyClient

logger = logging.getLogger(__name__)

_PATTERN = re.compile("|".join(re.escape(k) for k, _ in _KEYWORD_TYPES))


def fetch_regulatory_events_eastmoney(
    trade_date: date,
    *,
    client: EastMoneyClient | None = None,
    config=None,
) -> pl.DataFrame:
    """One day's regulatory events, CNINFO-adapter schema (``event_id`` onward)."""
    owns = client is None
    if client is None:
        client = EastMoneyClient(config=config)

    rows: list[dict] = []
    for item in _iter_announcement_items(trade_date, client=client):
        art_code = str(item.get("art_code") or "").strip()
        if not art_code:
            continue
        title = str(item.get("title") or "")
        if not _PATTERN.search(title):
            continue
        codes = [c for c in (item.get("codes") or []) if isinstance(c, dict)]
        for code_index, code_entry in enumerate(codes):
            stock_code = str(code_entry.get("stock_code") or "").strip()
            sym = _symbol_from_em_code(stock_code)
            if sym is None:
                continue
            # Composite id for secondary codes mirrors the announcement
            # adapter: the dataset primary key is event_id alone.
            event_id = art_code if code_index == 0 else f"{art_code}:{stock_code}"
            rows.append(
                {
                    "event_id": f"reg-{event_id}",
                    "symbol": sym,
                    "event_date": trade_date,
                    "event_type": _classify_event(title),
                    "title": title,
                }
            )

    if owns:
        client.close()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["event_id"], keep="last")
