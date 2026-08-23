"""EastMoney per-symbol stock news (on-demand + batch sentiment input)."""

from __future__ import annotations

import logging
from datetime import date

from cnequity.adapters.eastmoney.em_auth import EastMoneyClient
from cnequity.domain.sentiment import aggregate_scores, score_text
from cnequity.domain.symbols import parse_symbol

logger = logging.getLogger(__name__)

_NEWS_URL = "https://np-anotice-stock.eastmoney.com/api/security/news"
_MAX_DATE_PAGES = 100

_MARKET_CODES = {"SH": "1", "SZ": "0", "BJ": "2"}


def _parse_publish_date(raw: object) -> date | None:
    if raw is None:
        return None
    text = str(raw).strip().replace("/", "-")
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _normalize_item(item: dict, *, use_snownlp: bool) -> dict | None:
    title = str(item.get("title") or item.get("TITLE") or "").strip()
    if not title:
        return None
    publish_raw = item.get("showtime") or item.get("NOTICE_DATE") or item.get("publish_time")
    pub_date = _parse_publish_date(publish_raw)
    score, method = score_text(title, use_snownlp=use_snownlp)
    news_id = str(item.get("art_code") or item.get("uniqueUrl") or item.get("url") or title)
    return {
        "news_id": news_id,
        "title": title,
        "publish_time": str(publish_raw or ""),
        "publish_date": pub_date.isoformat() if pub_date else None,
        "url": str(item.get("url") or item.get("uniqueUrl") or ""),
        "sentiment_score": score,
        "sentiment_method": method,
    }


def fetch_stock_news(
    symbol: str,
    *,
    on_date: date | None = None,
    limit: int = 30,
    use_snownlp: bool = True,
    client: EastMoneyClient | None = None,
    config=None,
) -> dict:
    """Fetch recent headlines for *symbol*; optionally filter to *on_date*."""
    if limit <= 0:
        raise ValueError("stock_news limit must be positive")
    info = parse_symbol(symbol)
    market = _MARKET_CODES.get(info.exchange, "0")
    owns = client is None
    if client is None:
        client = EastMoneyClient(config=config)

    params = {
        "stock_list": info.code,
        "page_size": str(limit),
        "market_code": market,
        "client": "web",
    }
    items: list[dict] = []
    try:
        seen_pages: set[tuple[str, ...]] = set()
        page_index = 1
        while True:
            params["page_index"] = str(page_index)
            resp = client.get(_NEWS_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise RuntimeError("EastMoney stock_news response is not an object")
            raw_data = payload.get("data")
            if raw_data is None:
                raw_list = []
            elif not isinstance(raw_data, dict):
                raise RuntimeError("EastMoney stock_news response data is not an object")
            else:
                raw_list = raw_data.get("list")
                if raw_list is None:
                    raw_list = []
                elif not isinstance(raw_list, list):
                    raise RuntimeError("EastMoney stock_news response data.list is not a list")
            if not raw_list:
                break

            page_signature = tuple(
                str(
                    item.get("art_code")
                    or item.get("uniqueUrl")
                    or item.get("url")
                    or item.get("title")
                    or ""
                )
                for item in raw_list
                if isinstance(item, dict)
            )
            if page_signature in seen_pages:
                raise RuntimeError("EastMoney stock_news pagination did not advance")
            seen_pages.add(page_signature)

            page_dates: list[date] = []
            for index, raw in enumerate(raw_list):
                if not isinstance(raw, dict):
                    logger.warning(
                        "EastMoney stock_news: skipping non-object row %s for %s",
                        index,
                        symbol,
                    )
                    continue
                norm = _normalize_item(raw, use_snownlp=use_snownlp)
                if norm is None:
                    continue
                pub = _parse_publish_date(norm.get("publish_time"))
                if pub is not None:
                    page_dates.append(pub)
                if on_date is not None:
                    # A date-scoped request must never attribute an undated item
                    # to the requested session. Keep undated items only for the
                    # explicitly unscoped on-demand feed.
                    if pub != on_date:
                        continue
                items.append(norm)

            # The unscoped API contract is "recent headlines", so preserve its
            # one-page limit. Date-scoped sentiment needs to walk older pages;
            # EastMoney returns newest-first, so crossing before the target is
            # the safe stopping point.
            if on_date is None or len(raw_list) < limit:
                break
            if page_dates and min(page_dates) < on_date:
                break
            page_index += 1
            if page_index > _MAX_DATE_PAGES:
                raise RuntimeError(
                    f"EastMoney stock_news pagination exceeded {_MAX_DATE_PAGES} pages"
                )
    except Exception as exc:
        logger.warning("EastMoney stock_news failed for %s: %s", symbol, exc)
        if owns:
            client.close()
        return {
            "symbol": symbol,
            "source": "eastmoney",
            "items": [],
            "headline_count": 0,
            "aggregate_sentiment": 0.0,
            "error": str(exc),
        }

    if owns:
        client.close()

    # The endpoint can repeat an article across adjacent result pages or expose
    # a revised row under the same stable id. Repeated headlines would inflate
    # both headline_count and the aggregate sentiment used by the daily batch.
    items = list({item["news_id"]: item for item in items}.values())
    scores = [float(i["sentiment_score"]) for i in items]
    return {
        "symbol": symbol,
        "source": "eastmoney",
        "items": items,
        "headline_count": len(items),
        "aggregate_sentiment": aggregate_scores(scores),
        "on_date": on_date.isoformat() if on_date else None,
    }
