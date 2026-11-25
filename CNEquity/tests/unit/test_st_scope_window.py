"""ST 证据续签窗口：daily 走滚动窗口，显式 --start / 全量 backfill 保持原语义。

2026-11-16 事故：daily 续签沿用 BACKFILL_START(2016) 作 scope 起点，导致每天都要
校验"2016 起全窗口证据已在 curated 里齐备"。201 只票历史行数不足时整张 receipt
签不出，historical all-A 长期 BLOCKED，只能人工重签。
"""

from datetime import date, timedelta

from cnequity.steps import reference
from cnequity.steps.common import BACKFILL_START


def _cfg(backfill: bool = False, start: date | None = None):
    class _C:
        pass

    c = _C()
    c._backfill = backfill
    c._backfill_start = start
    return c


def test_daily_uses_rolling_window():
    trade_date = date(2026, 11, 16)
    got = reference._st_scope_start(_cfg(), trade_date)
    assert got == max(BACKFILL_START, trade_date - timedelta(days=reference._ST_DAILY_LOOKBACK_DAYS))
    # 必须明显晚于 2016，否则又回到每天校验十年全量的老路
    assert got > BACKFILL_START


def test_explicit_start_wins():
    explicit = date(2026, 9, 1)
    assert reference._st_scope_start(_cfg(start=explicit), date(2026, 11, 16)) == explicit


def test_full_backfill_keeps_history():
    assert reference._st_scope_start(_cfg(backfill=True), date(2026, 11, 16)) == BACKFILL_START
