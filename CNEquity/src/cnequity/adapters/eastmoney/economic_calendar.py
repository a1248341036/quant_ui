"""Economic calendar — rolling window snapshot (forecast/previous/actual)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl

from cnequity.adapters.eastmoney.common import _to_int
from cnequity.adapters.eastmoney.datacenter import fetch_datacenter
from cnequity.adapters.eastmoney.em_auth import EastMoneyClient

_REPORT = "RPT_ECONOMICCALENDAR"
_COLUMNS = "PUBLISH_DATE,TIME,COUNTRY,INDICATOR,STAR,FORECAST,PREVIOUS,ACTUAL,UNIT"


def _first_present(*values: object) -> object | None:
    """Return the first value that is present without treating numeric zero as missing."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _parse_float(value: object) -> float | None:
    if value is None or value == "" or value == "--":
        return None
    try:
        parsed = float(str(value).replace(",", "").replace("%", ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def fetch_economic_calendar_window(start: date, end: date, *, config=None) -> pl.DataFrame:
    """Fetch calendar events in [start, end]; empty if source unavailable."""
    rows: list[dict] = []
    client_kwargs = {"config": config} if config is not None else {}
    with EastMoneyClient(**client_kwargs) as client:
        data = fetch_datacenter(
            client,
            _REPORT,
            columns=_COLUMNS,
            page_size=500,
            sort_columns="PUBLISH_DATE,TIME",
        )

    for item in data or []:
        pub = str(_first_present(item.get("PUBLISH_DATE"), item.get("publish_date")) or "")[:10]
        if not pub:
            continue
        try:
            event_date = date.fromisoformat(pub)
        except ValueError:
            continue
        if event_date < start or event_date > end:
            continue
        indicator = str(_first_present(item.get("INDICATOR"), item.get("indicator")) or "").strip()
        country = str(_first_present(item.get("COUNTRY"), item.get("country")) or "").strip()
        if not indicator:
            # Without an indicator there is no stable event identity; allowing
            # it through would collapse unrelated rows onto the same key.
            continue
        event_time = str(_first_present(item.get("TIME"), item.get("time")) or "").strip()
        star = _first_present(item.get("STAR"), item.get("star"))
        importance = _to_int(star)
        event_id = f"{event_date.isoformat()}|{event_time}|{country}|{indicator}"
        rows.append(
            {
                "event_id": event_id,
                "event_date": event_date,
                "event_time": event_time,
                "country": country,
                "indicator": indicator,
                "importance": importance,
                "forecast": _parse_float(
                    _first_present(item.get("FORECAST"), item.get("forecast"))
                ),
                "previous": _parse_float(
                    _first_present(item.get("PREVIOUS"), item.get("previous"))
                ),
                "actual": _parse_float(_first_present(item.get("ACTUAL"), item.get("actual"))),
                "unit": str(_first_present(item.get("UNIT"), item.get("unit")) or "").strip()
                or None,
            }
        )

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["event_id"], keep="last")


def fetch_economic_calendar(trade_date: date, *, config=None) -> pl.DataFrame:
    """Rolling window [trade_date-2, trade_date+14] for snapshot daily runs."""
    start = trade_date - timedelta(days=2)
    end = trade_date + timedelta(days=14)
    if config is None:
        return fetch_economic_calendar_window(start, end)
    return fetch_economic_calendar_window(start, end, config=config)
