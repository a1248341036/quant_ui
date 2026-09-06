"""EastMoney announcement index (np-anotice-stock).

Primary source for ``announcement_index``, replacing the CNINFO walk as the
default. Why: CNINFO's ``hisAnnouncement`` endpoint serves 30 rows per page
per exchange column with a hard ~100-page wrap (a 3,000+ announcement day is
silently truncated), while this endpoint serves 100 rows per page over the
whole market with deep pagination verified to the last row (12,994 rows on
2026-04-30 = 130 fully reachable pages). At the configured 0.5s pacing a
typical day is a handful of requests; CNINFO needed hundreds plus 504
retry chains.

Historical depth matches the CNINFO walk's 2010 floor: the archive serves
2016-era days with correct ``notice_date`` values (the ``ANyyyymmdd`` prefix
inside ``art_code`` is the indexing date, not the notice date — old filings
were re-archived later, so it must never be used as the date filter).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from datetime import date

import polars as pl

from cnequity.adapters.eastmoney.em_auth import EastMoneyClient
from cnequity.domain.symbols import format_symbol, infer_exchange_from_code, is_all_a_symbol

logger = logging.getLogger(__name__)

_NOTICE_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
_PAGE_SIZE = 100
_GET_RETRIES = 4
_GET_BACKOFF_SECONDS = 3.0


def _get_with_retry(client: EastMoneyClient, params: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(_GET_RETRIES):
        try:
            resp = client.get(_NOTICE_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
                raise RuntimeError("EastMoney announcements response shape unexpected")
            return payload
        except Exception as exc:  # noqa: BLE001 — retried uniformly, re-raised below
            last_exc = exc
            if attempt + 1 < _GET_RETRIES:
                time.sleep(_GET_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _symbol_from_em_code(code: str) -> str | None:
    code = str(code).strip().zfill(6)
    if len(code) != 6 or not code.isdigit():
        return None
    exch = infer_exchange_from_code(code)
    if not is_all_a_symbol(code, exch):
        return None
    return format_symbol(code, exch)


def _detail_url(stock_code: str, art_code: str) -> str:
    return f"https://data.eastmoney.com/notices/detail/{stock_code}/{art_code}.html"


def _iter_announcement_items(
    trade_date: date,
    *,
    client: EastMoneyClient,
) -> "Iterator[dict]":
    """Yield every raw announcement item for *trade_date*, with completeness guards.

    Shared by the announcement-index and regulatory-events adapters: one
    paged sweep over the day's full notice list, failing loud on a short
    read (empty page before the declared end) or a row whose own date
    disagrees with the requested day.
    """
    day = trade_date.strftime("%Y-%m-%d")
    expected_pages: int | None = None
    page = 1
    while True:
        payload = _get_with_retry(
            client,
            {
                "sr": -1,
                "page_size": _PAGE_SIZE,
                "page_index": page,
                "ann_type": "A",
                "client_source": "web",
                "f_node": 0,
                "s_node": 0,
                "begin_time": day,
                "end_time": day,
            },
        )
        data = payload["data"]
        batch = data.get("list") or []
        if not isinstance(batch, list):
            raise RuntimeError(
                f"EastMoney announcements list for {day} page {page} is not a list"
            )
        if expected_pages is None:
            total_hits = data.get("total_hits")
            if not isinstance(total_hits, int) or total_hits < 0:
                raise RuntimeError(f"EastMoney announcements total_hits for {day} invalid")
            expected_pages = max(math.ceil(total_hits / _PAGE_SIZE), 1) if total_hits else 0
        if not batch:
            if expected_pages and page < expected_pages:
                raise RuntimeError(
                    f"EastMoney announcements returned an empty page before the "
                    f"reported end for {day} (page {page}/{expected_pages})"
                )
            return
        for index, item in enumerate(batch):
            if not isinstance(item, dict):
                logger.warning(
                    "EastMoney announcements: skipping non-object row %s on page %s", index, page
                )
                continue
            notice_date = str(item.get("notice_date") or "").strip()[:10]
            if notice_date != day:
                raise RuntimeError(
                    f"EastMoney announcement {item.get('art_code')!r} notice_date "
                    f"{notice_date!r} does not match requested {day}"
                )
            yield item
        if expected_pages and page >= expected_pages:
            return
        page += 1


def fetch_announcement_index_eastmoney(
    trade_date: date,
    *,
    client: EastMoneyClient | None = None,
    config=None,
) -> pl.DataFrame:
    """One trading day's announcement index rows, CNINFO-adapter schema.

    Output columns match ``adapters.cninfo.announcements`` exactly so the
    step, the writer and the dataset schema are source-agnostic. Rows whose
    ``notice_date`` disagrees with the requested day raise — same
    source-contract rule as the CNINFO adapter.
    """
    owns = client is None
    if client is None:
        client = EastMoneyClient(config=config)

    rows: list[dict] = []
    for item in _iter_announcement_items(trade_date, client=client):
        art_code = str(item.get("art_code") or "").strip()
        if not art_code:
            logger.warning("EastMoney announcement missing art_code; skipping")
            continue
        codes = [c for c in (item.get("codes") or []) if isinstance(c, dict)]
        category = ""
        for column in item.get("columns") or []:
            if isinstance(column, dict) and column.get("column_name"):
                category = str(column["column_name"])
                break
        # Joint filings list several codes under one art_code. The dataset
        # primary key is announcement_id alone, so secondary codes get a
        # composite identity instead of colliding or being dropped.
        for code_index, code_entry in enumerate(codes):
            stock_code = str(code_entry.get("stock_code") or "").strip()
            sym = _symbol_from_em_code(stock_code)
            if sym is None:
                continue
            ann_id = art_code if code_index == 0 else f"{art_code}:{stock_code}"
            rows.append(
                {
                    "announcement_id": ann_id,
                    "symbol": sym,
                    "title": str(item.get("title") or ""),
                    "announce_date": trade_date,
                    "category": category,
                    "url": _detail_url(stock_code, art_code),
                }
            )

    if owns:
        client.close()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["announcement_id"], keep="last")
