"""CNINFO regulatory / compliance events (filtered from announcements)."""

from __future__ import annotations

import logging
import re
from datetime import date

import httpx
import polars as pl

from cnequity.adapters.cninfo.announcements import (
    _PAGE_SIZE,
    _announcement_batch,
    _announcement_id,
    _pagination_has_more,
    _pagination_page_signature,
    _pagination_total_pages,
    _symbol_from_cninfo,
    _validate_source_date,
    post_with_retry,
)

logger = logging.getLogger(__name__)

_CNINFO_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"

_KEYWORD_TYPES: list[tuple[str, str]] = [
    ("行政处罚", "penalty"),
    ("处罚决定", "penalty"),
    ("立案", "investigation"),
    ("调查", "investigation"),
    ("监管函", "regulatory_letter"),
    ("警示函", "warning_letter"),
    ("处分", "disciplinary"),
]


def _classify_event(title: str) -> str:
    for keyword, event_type in _KEYWORD_TYPES:
        if keyword in title:
            return event_type
    return "regulatory"


def fetch_regulatory_events(
    trade_date: date,
    *,
    client: httpx.Client | None = None,
    config=None,
) -> pl.DataFrame:
    owns = client is None
    if client is None:
        client = httpx.Client(timeout=30.0, headers={"User-Agent": "Mozilla/5.0"})

    ds = trade_date.strftime("%Y-%m-%d")
    pattern = re.compile("|".join(re.escape(k) for k, _ in _KEYWORD_TYPES))
    rows: list[dict] = []

    for column in ("szse", "sse"):
        page = 1
        seen_page_signatures: set[str] = set()
        while True:
            if config is not None:
                config.rate_limit("cninfo")
            payload = {
                "pageNum": page,
                "pageSize": 30,
                "column": column,
                "tabName": "fulltext",
                "plate": "",
                "stock": "",
                "searchkey": "",
                "secid": "",
                "category": "",
                "trade": "",
                "seDate": f"{ds}~{ds}",
            }
            try:
                data = post_with_retry(client, _CNINFO_URL, data=payload)
                batch = _announcement_batch(data, column=column, page=page)
                total_pages = _pagination_total_pages(data, column=column, page=page)
                has_more = _pagination_has_more(data, column=column, page=page)
                has_more_present = data.get("hasMore") is not None
                if batch and total_pages == 0:
                    raise RuntimeError(
                        f"CNINFO regulatory for {column} page {page} declared "
                        "totalpages=0 but returned rows"
                    )
                if batch:
                    page_signature = _pagination_page_signature(batch)
                    if page_signature in seen_page_signatures:
                        # See announcements.fetch_announcement_index: CNINFO
                        # wraps around after ~100 pages.
                        logger.warning(
                            "CNINFO regulatory pagination wrapped at %s "
                            "page %s (totalpages=%s); stopping early",
                            column,
                            page,
                            total_pages,
                        )
                        break
                    seen_page_signatures.add(page_signature)
            except Exception as exc:
                # Don't truncate: short pages drop blacklist rows. Fail loud
                # once retries (post_with_retry) are exhausted.
                logger.warning("CNINFO regulatory page failed (%s p%s): %s", column, page, exc)
                if owns:
                    client.close()
                raise RuntimeError(
                    f"CNINFO regulatory pagination failed for {column} page {page}: {exc}"
                ) from exc

            if not batch:
                if (isinstance(total_pages, int) and total_pages > 0 and page <= total_pages) or (
                    total_pages is None and has_more
                ):
                    raise RuntimeError(
                        f"CNINFO regulatory returned an empty page before the "
                        f"reported end for {column} page {page}"
                    )
                break
            for item in batch:
                _validate_source_date(item, trade_date, column=column)
                title = str(item.get("announcementTitle") or "")
                if not pattern.search(title):
                    continue
                sym = _symbol_from_cninfo(str(item.get("secCode", "")))
                if not sym:
                    continue
                ann_id = _announcement_id(item)
                if ann_id is None:
                    logger.warning("CNINFO regulatory announcement missing identity; skipping")
                    continue
                rows.append(
                    {
                        "event_id": f"reg-{ann_id}",
                        "symbol": sym,
                        "event_date": trade_date,
                        "event_type": _classify_event(title),
                        "title": title,
                    }
                )
            if isinstance(total_pages, int) and page >= total_pages:
                # See announcements.fetch_announcement_index: hasMore cannot
                # be trusted past the server's own reported total — measured
                # live, it stays true forever while replaying page 1's rows.
                break
            if isinstance(total_pages, int):
                # A stale false `hasMore` must not truncate the blacklist when
                # the server still reports additional pages.
                page += 1
                continue
            if not has_more:
                if total_pages is None and not has_more_present and len(batch) >= _PAGE_SIZE:
                    page += 1
                    continue
                break
            page += 1

    if owns:
        client.close()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["event_id"], keep="last")
