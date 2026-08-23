"""Unit tests for Sina offshore commodity bars (COMEX gold)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cnequity.adapters.sina.global_futures import (
    OFFSHORE_CONTRACTS,
    fetch_offshore_commodity_bars_range,
)
from cnequity.domain.schemas import validate_dataframe, with_provenance


def test_offshore_contracts_unique():
    syms = [c[0] for c in OFFSHORE_CONTRACTS]
    assert len(syms) == len(set(syms))
    assert ("GC0.CMX", "GC", "COMEX黄金", "CMX") in OFFSHORE_CONTRACTS


def test_explicit_empty_offshore_contracts_are_a_noop():
    client = MagicMock()
    client.get.side_effect = AssertionError("empty contract selection must not fetch defaults")
    df = fetch_offshore_commodity_bars_range(
        date(2026, 7, 21), date(2026, 7, 21), contracts=(), client=client
    )
    assert df.is_empty()


def test_fetch_offshore_parses_sina_payload():
    payload = [
        {
            "date": "2026-07-20",
            "open": "4022.700",
            "high": "4046.000",
            "low": "3986.500",
            "close": "4011.800",
            "volume": "0",
            "position": "12",
            "settlement": "0",
        },
        {
            "date": "2026-07-21",
            "open": "4011.800",
            "high": "4088.400",
            "low": "4003.300",
            "close": "4077.700",
            "volume": "0",
            "position": "0",
            "settlement": "4077.700",
        },
        {
            "date": "2019-01-02",
            "open": "1",
            "high": "1",
            "low": "1",
            "close": "1",
            "volume": "0",
            "position": "0",
            "settlement": "0",
        },
    ]
    fake = MagicMock()
    fake.get.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value=payload),
    )
    with patch("cnequity.adapters.sina.global_futures.httpx.Client", return_value=fake):
        fake.__enter__ = MagicMock(return_value=fake)
        fake.__exit__ = MagicMock(return_value=None)
        # Client is constructed as context? Our code doesn't use context manager —
        # it constructs Client() and calls .get / .close
        client = MagicMock()
        client.get.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value=payload),
        )
        df = fetch_offshore_commodity_bars_range(
            date(2026, 7, 20),
            date(2026, 7, 21),
            client=client,
        )
    assert df.height == 2
    assert set(df["symbol"].to_list()) == {"GC0.CMX"}
    assert df.filter(pl.col("trade_date") == date(2026, 7, 21))["close"][0] == 4077.7
    assert df.filter(pl.col("trade_date") == date(2026, 7, 21))["open_interest"][0] is None
    assert df["source"].unique().to_list() == ["sina"]
    validated = validate_dataframe(
        with_provenance(df, source="eastmoney", data_version="v1"),
        "commodity_bars",
    )
    assert validated["source"].unique().to_list() == ["sina"]


def test_fetch_offshore_skips_non_object_rows_and_keeps_valid_rows():
    client = MagicMock()
    client.get.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(
            return_value=[
                None,
                {
                    "date": "2026-07-21",
                    "open": "4011.8",
                    "high": "4088.4",
                    "low": "4003.3",
                    "close": "4077.7",
                    "volume": "10",
                },
            ]
        ),
    )
    df = fetch_offshore_commodity_bars_range(date(2026, 7, 21), date(2026, 7, 21), client=client)
    assert df.height == 1


def test_fetch_offshore_empty_on_bad_payload():
    client = MagicMock()
    client.get.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value={"error": True}),
    )
    df = fetch_offshore_commodity_bars_range(date(2026, 7, 21), date(2026, 7, 21), client=client)
    assert df.is_empty()


def test_fetch_offshore_malformed_payload_fails_strict_fetch():
    client = MagicMock()
    client.get.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value={"error": True}),
    )
    with pytest.raises(RuntimeError, match="offshore commodity_bars failed for GC0.CMX"):
        fetch_offshore_commodity_bars_range(
            date(2026, 7, 21), date(2026, 7, 21), client=client, strict=True
        )


def test_missing_ohlcv_is_not_replaced_with_close_or_zero():
    client = MagicMock()
    client.get.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(
            return_value=[
                {
                    "date": "2026-07-21",
                    "open": None,
                    "high": "4088.400",
                    "low": "4003.300",
                    "close": "4077.700",
                    "volume": None,
                }
            ]
        ),
    )

    df = fetch_offshore_commodity_bars_range(date(2026, 7, 21), date(2026, 7, 21), client=client)

    assert df.is_empty()


def test_nonfinite_open_interest_is_dropped():
    client = MagicMock()
    client.get.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(
            return_value=[
                {
                    "date": "2026-07-21",
                    "open": "4011.8",
                    "high": "4088.4",
                    "low": "4003.3",
                    "close": "4077.7",
                    "volume": "10",
                    "position": "inf",
                }
            ]
        ),
    )
    df = fetch_offshore_commodity_bars_range(date(2026, 7, 21), date(2026, 7, 21), client=client)
    assert df.height == 1
    assert df["open_interest"][0] is None


def test_int64_overflow_volume_is_dropped():
    client = MagicMock()
    client.get.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(
            return_value=[
                {
                    "date": "2026-07-21",
                    "open": "4011.8",
                    "high": "4088.4",
                    "low": "4003.3",
                    "close": "4077.7",
                    "volume": "1e300",
                }
            ]
        ),
    )
    assert fetch_offshore_commodity_bars_range(
        date(2026, 7, 21), date(2026, 7, 21), client=client
    ).is_empty()


def test_strict_transport_failure_is_not_treated_as_empty_history():
    client = MagicMock()
    client.get.side_effect = RuntimeError("route down")

    with pytest.raises(RuntimeError, match="offshore commodity_bars failed"):
        fetch_offshore_commodity_bars_range(
            date(2026, 7, 21),
            date(2026, 7, 21),
            client=client,
            strict=True,
        )
