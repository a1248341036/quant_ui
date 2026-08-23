"""Exchange-local clock helpers used by ingestion defaults and gates."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

# Mainland China has used UTC+08:00 without daylight-saving changes for the
# exchange dates handled by this project. A fixed offset also works on Windows
# without requiring an IANA timezone database or an extra runtime dependency.
SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
# Leave a small settlement buffer after the 15:00 closing auction before a
# daily dataset is expected to publish the session.
A_SHARE_FINAL_AT = time(15, 5)


def shanghai_now(now: datetime | None = None) -> datetime:
    """Return an aware current timestamp represented in exchange time."""
    anchor = now or datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        raise ValueError("market clock must be timezone-aware")
    return anchor.astimezone(SHANGHAI_TZ)


def shanghai_today(now: datetime | None = None) -> date:
    """Return the exchange-local calendar date, independent of host timezone."""
    return shanghai_now(now).date()


def is_session_final(as_of: date, now: datetime | None = None) -> bool:
    """Whether *as_of* is a completed exchange session at the current clock.

    Historical/future dates are treated as final. Only the current Shanghai
    date is provisional before the settlement buffer, which prevents a Monday
    pre-open health check from calling Friday's complete data stale.
    """
    local_now = shanghai_now(now)
    return as_of != local_now.date() or local_now.time() >= A_SHARE_FINAL_AT
