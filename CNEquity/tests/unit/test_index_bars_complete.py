"""index_bars must fail-loud on a partial symbol set (no watermark poison)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import polars as pl
import pytest

from cnequity.adapters.tdx_protocol.client import INDEX_SYMBOLS, TdxSourceError, fetch_index_bars
from cnequity.config import Config
from cnequity.steps.bars import _validate_index_bar_coverage, step_index_bars


def test_fetch_index_bars_rejects_partial_symbol_set(monkeypatch):
    """One surviving index must not advance daily coverage for the other seven."""

    def fake_paginated(client, sym, start, end, **kwargs):
        if sym == "000852.SH":
            return [
                {
                    "symbol": sym,
                    "trade_date": start,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                    "amount": 1.0,
                }
            ]
        raise RuntimeError(f"tdx down for {sym}")

    monkeypatch.setattr(
        "cnequity.adapters.tdx_protocol.client.fetch_bars_paginated",
        fake_paginated,
    )
    monkeypatch.setattr(
        "cnequity.adapters.tdx_protocol.client._quotes_client",
        lambda config=None: MagicMock(),
    )
    monkeypatch.setattr(
        "cnequity.adapters.tdx_protocol.client._close_quotes_client",
        lambda client: None,
    )
    monkeypatch.setattr(
        "cnequity.adapters.tdx_protocol.client.reset_tdx_server_cache",
        lambda: None,
    )
    # Exhaust retries quickly without sleeping on mock path.
    monkeypatch.setattr(
        "cnequity.adapters.tdx_protocol.client._TDX_FETCH_ATTEMPTS",
        1,
    )

    with pytest.raises(TdxSourceError, match="000001.SH"):
        fetch_index_bars(date(2026, 7, 10), date(2026, 7, 10), allow_mock=False)


def test_index_bar_coverage_rejects_interior_symbol_session_gap(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    symbols = [f"{code}.{exchange}" for code, exchange in INDEX_SYMBOLS]
    days = [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 5)]
    rows = [
        {"symbol": symbol, "trade_date": day}
        for symbol in symbols
        for day in days
        if not (symbol == "000300.SH" and day == date(2024, 6, 4))
    ]
    frame = pl.DataFrame(rows)
    with pytest.raises(RuntimeError, match="000300.SH@2024-06-04"):
        _validate_index_bar_coverage(cfg, frame, days[0], days[-1])


def test_step_index_bars_applies_finality_guard_before_fetch(config, monkeypatch):
    events: list[str] = []

    def stop_before_fetch(*args, **kwargs):
        events.append("guard")
        raise RuntimeError("stop before fetch")

    monkeypatch.setattr(
        "cnequity.steps.bars._reject_unfinished_daily_bar_window",
        stop_before_fetch,
    )
    monkeypatch.setattr(
        "cnequity.steps.bars.incremental_window",
        lambda *args, **kwargs: date(2026, 8, 10),
    )
    monkeypatch.setattr(
        "cnequity.steps.bars.fetch_index_bars",
        lambda *args, **kwargs: events.append("fetch"),
    )

    with pytest.raises(RuntimeError, match="stop before fetch"):
        step_index_bars(config, date(2026, 8, 10), "run-index-guard", {})

    assert events == ["guard"]
