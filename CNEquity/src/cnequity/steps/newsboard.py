"""News wire + economic calendar archive steps (daily batch)."""

from __future__ import annotations

import logging
from datetime import date

from cnequity.adapters.eastmoney.economic_calendar import fetch_economic_calendar
from cnequity.adapters.eastmoney.news_wire import fetch_flash_news_wire
from cnequity.config import Config
from cnequity.orchestrator.registry import register_step
from cnequity.steps.common import SnapshotBackfillError
from cnequity.steps.http_common import empty_ok, run_incremental_fetched, write_fetched

logger = logging.getLogger(__name__)


@register_step("flash_news_wire", group="research")
def step_flash_news_wire(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("flash_news_wire: eastmoney source disabled in config")
    # Fail-loud on empty: an empty success left the dataset unregistered in curated
    # and permanently failed lake_health (exists error) while the step looked green.

    def _fetch(d: date):
        return fetch_flash_news_wire(d, config=config)

    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "flash_news_wire",
        _fetch,
        source="eastmoney",
        allow_empty=False,
        date_col="publish_date",
    )


@register_step("economic_calendar", group="macro_risk")
def step_economic_calendar(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("economic_calendar: eastmoney source disabled in config")
    # This is a rolling live window, not historical by-date data. It cannot
    # be routed through the daily helper because its event_date intentionally
    # contains future dates; reject backfill explicitly before fetching.
    if getattr(config, "_backfill", False):
        raise SnapshotBackfillError(
            "economic_calendar: backfill not supported — fetch semantics are snapshot "
            "(rolling live window; historical values unavailable)"
        )
    try:
        df = fetch_economic_calendar(trade_date, config=config)
    except Exception as exc:
        if "9501" in str(exc):
            logger.warning(
                "economic_calendar: EastMoney RPT_ECONOMICCALENDAR endpoint is "
                "unavailable (code=9501); skipping this run"
            )
            return {"rows_read": 0, "rows_written": 0, "skipped": "endpoint unavailable"}
        raise
    empty_ok(df, "economic_calendar", trade_date)
    return write_fetched(config, run_id, "economic_calendar", df, source="eastmoney")
