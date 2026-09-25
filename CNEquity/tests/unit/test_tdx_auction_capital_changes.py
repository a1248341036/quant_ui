"""Offline coverage for TDX 0x056A auction + 0x000F capital changes.

Wire-level tests build the exact byte layouts from the eltdx Rust reference;
adapter tests exercise the unit semantics (手→股, 万股→股) against fake
clients, so nothing here touches the network.
"""

from __future__ import annotations

import struct
from datetime import date, datetime

import polars as pl
import pytest

from cnequity.adapters.tdx_protocol import auction as auction_mod
from cnequity.adapters.tdx_protocol import capital_changes as cc_mod
from cnequity.adapters.tdx_protocol._wire.parser.std.get_auction_series import (
    GetAuctionSeriesCmd,
)
from cnequity.adapters.tdx_protocol._wire.parser.std.get_capital_changes import (
    GetCapitalChangesCmd,
)


def _auction_body(records: list[tuple]) -> bytes:
    body = struct.pack("<H", len(records))
    for minute_of_day, second, price, matched, unmatched, reserved in records:
        body += struct.pack("<HfIiBB", minute_of_day, price, matched, unmatched, reserved, second)
    return body


def test_auction_parser_request_bytes():
    cmd = GetAuctionSeriesCmd(None)
    cmd.setParams(0, "000988", date=20260814, mode=3, start=0, count=500)
    pkg = bytes(cmd.send_pkg)
    assert len(pkg) == 12 + 28
    assert pkg[:12] == struct.pack("<HIHHH", 0x10C, 0x01016408, 30, 30, 0x056A)
    assert pkg[12:] == struct.pack(
        "<BB6sIIIII", 0, 0, b"000988", 20260814, 3, 0, 0, 500
    )


def test_auction_parser_parse_response():
    body = _auction_body(
        [
            (555, 0, 10.5, 100, -50, 0),  # 09:15:00 opening auction
            (897, 30, 10.6, 200, 75, 0),  # 14:57:30 closing auction
        ]
    )
    rows = GetAuctionSeriesCmd(None).parseResponse(body)
    assert len(rows) == 2
    assert rows[0]["minute_of_day"] == 555
    assert rows[0]["second"] == 0
    assert rows[0]["price"] == pytest.approx(10.5)
    assert rows[0]["matched_volume"] == 100
    assert rows[0]["unmatched_signed"] == -50
    assert rows[1]["minute_of_day"] == 897
    assert rows[1]["second"] == 30


def test_auction_parser_parse_response_truncated():
    assert GetAuctionSeriesCmd(None).parseResponse(b"\x01\x00\x00") == []


def _capital_changes_body(records: list[tuple]) -> bytes:
    body = struct.pack("<H", 1)  # one reported block
    body += struct.pack("<B6sH", 0, b"000001", len(records))
    for market, code, reserved, date_raw, category, c1, c2, c3, c4 in records:
        body += struct.pack("<B6sBIBffff", market, code, reserved, date_raw, category, c1, c2, c3, c4)
    return body


def test_capital_changes_parser_request_bytes():
    cmd = GetCapitalChangesCmd(None)
    cmd.setParams(0, "000001")
    pkg = bytes(cmd.send_pkg)
    assert len(pkg) == 12 + 9
    assert pkg[:12] == struct.pack("<HIHHH", 0x10C, 0x01016408, 11, 11, 0x000F)
    assert pkg[12:] == struct.pack("<HB6s", 1, 0, b"000001")


def test_capital_changes_parser_parse_response():
    body = _capital_changes_body(
        [
            (0, b"000001", 0, 19910403, 5, 0.0, 0.0, 2650.0, 4850.0171),
            (0, b"000001", 0, 19900301, 1, 3.56, 0.0, 0.0, 1.0),
            (0, b"000001", 0, 20200101, 15, 1.0, 2.0, 3.0, 4.0),
        ]
    )
    rows = GetCapitalChangesCmd(None).parseResponse(body)
    assert len(rows) == 3
    assert rows[0]["date"] == 19910403
    assert rows[0]["category"] == 5
    assert rows[0]["name"] == "股本变化"
    assert rows[0]["c3"] == pytest.approx(2650.0)
    assert rows[1]["name"] == "除权除息"
    assert rows[2]["name"] == "重整调整"


def test_capital_changes_parser_parse_response_truncated():
    assert GetCapitalChangesCmd(None).parseResponse(b"\x01\x00") == []


class _AuctionClient:
    def __init__(self, raw, error=False):
        self.raw = raw
        self.error = error
        self.calls = []

    def auction_series(self, symbol, market=None, on_date=0):
        self.calls.append((symbol, market, on_date))
        if self.error:
            raise ConnectionError("socket reset")
        return self.raw


def test_auction_adapter_normalizes_rows(monkeypatch):
    monkeypatch.setattr(auction_mod, "wait_spec", lambda *a, **k: None)
    client = _AuctionClient(
        [
            {"minute_of_day": 555, "second": 0, "price": 10.5, "matched_volume": 10, "unmatched_signed": -5, "reserved": 0},
            {"minute_of_day": 897, "second": 30, "price": 10.6, "matched_volume": 20, "unmatched_signed": 7, "reserved": 0},
        ]
    )
    df = auction_mod.fetch_auction_series(
        client, "000001.SZ", trade_date=date(2026, 8, 14)
    )
    assert client.calls == [("000001", 0, 20260814)]
    assert df.height == 2
    assert df["session"].to_list() == ["open", "close"]
    assert df["matched_volume"].to_list() == [1000, 2000]  # 手 → 股
    assert df["unmatched_volume"].to_list() == [500, 700]
    assert df["unmatched_direction"].to_list() == [-1, 1]
    assert df["auction_time"].to_list() == [
        datetime(2026, 8, 14, 9, 15, 0),
        datetime(2026, 8, 14, 14, 57, 30),
    ]
    assert df["price"].to_list() == [10.5, 10.6]


def test_auction_adapter_empty_on_error_and_empty(monkeypatch):
    monkeypatch.setattr(auction_mod, "wait_spec", lambda *a, **k: None)
    assert auction_mod.fetch_auction_series(
        _AuctionClient([], error=True), "000001.SZ", trade_date=date(2026, 8, 14)
    ).is_empty()
    assert auction_mod.fetch_auction_series(
        _AuctionClient(None), "000001.SZ", trade_date=date(2026, 8, 14)
    ).is_empty()


class _CapitalChangesClient:
    def __init__(self, raw, error=False):
        self.raw = raw
        self.error = error
        self.calls = []

    def capital_changes(self, symbol, market=None):
        self.calls.append((symbol, market))
        if self.error:
            raise ConnectionError("socket reset")
        return self.raw


def test_capital_changes_adapter_unit_semantics(monkeypatch):
    monkeypatch.setattr(cc_mod, "wait_spec", lambda *a, **k: None)
    client = _CapitalChangesClient(
        [
            {"date": 19910403, "category": 5, "name": "股本变化", "c1": 0.0, "c2": 0.0, "c3": 2650.0, "c4": 4850.0171},
            {"date": 20200101, "category": 6, "name": "增发新股", "c1": 1.5, "c2": 2.5, "c3": 300.0, "c4": 4.5},
            {"date": 19900301, "category": 1, "name": "除权除息", "c1": 3.56, "c2": 0.0, "c3": 0.0, "c4": 1.0},
        ]
    )
    df = cc_mod.fetch_capital_changes(client, "000001.SZ")
    assert client.calls == [("000001", 0)]
    assert df.height == 3
    by_date = {r["event_date"]: r for r in df.to_dicts()}
    # 股本数量类: 万股 × 10000 → 股
    assert by_date[date(1991, 4, 3)]["c3"] == pytest.approx(26_500_000.0)
    assert by_date[date(1991, 4, 3)]["c4"] == pytest.approx(48_500_171.0)
    # 增发新股: 只有 c3 是万股
    assert by_date[date(2020, 1, 1)]["c1"] == pytest.approx(1.5)
    assert by_date[date(2020, 1, 1)]["c3"] == pytest.approx(3_000_000.0)
    assert by_date[date(2020, 1, 1)]["c4"] == pytest.approx(4.5)
    # 除权除息: 原值 (每10股口径)
    assert by_date[date(1990, 3, 1)]["c1"] == pytest.approx(3.56)
    assert by_date[date(1990, 3, 1)]["c4"] == pytest.approx(1.0)


def test_capital_changes_adapter_filters_on_date(monkeypatch):
    monkeypatch.setattr(cc_mod, "wait_spec", lambda *a, **k: None)
    client = _CapitalChangesClient(
        [
            {"date": 19910403, "category": 5, "name": "股本变化", "c1": 0.0, "c2": 0.0, "c3": 2650.0, "c4": 0.0},
            {"date": 19900301, "category": 1, "name": "除权除息", "c1": 3.56, "c2": 0.0, "c3": 0.0, "c4": 1.0},
        ]
    )
    df = cc_mod.fetch_capital_changes(
        client, "000001.SZ", on_date=date(1991, 4, 3)
    )
    assert df.height == 1
    assert df["event_date"].to_list() == [date(1991, 4, 3)]


def test_capital_changes_adapter_strict_raises(monkeypatch):
    monkeypatch.setattr(cc_mod, "wait_spec", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="TDX capital changes failed for 000001.SZ"):
        cc_mod.fetch_capital_changes(
            _CapitalChangesClient([], error=True), "000001.SZ", strict=True
        )


def test_capital_changes_adapter_skips_bad_dates(monkeypatch):
    monkeypatch.setattr(cc_mod, "wait_spec", lambda *a, **k: None)
    client = _CapitalChangesClient(
        [
            {"date": 19911303, "category": 5, "name": "股本变化", "c1": 0.0, "c2": 0.0, "c3": 1.0, "c4": 0.0},
            {"date": 19910403, "category": 5, "name": "股本变化", "c1": 0.0, "c2": 0.0, "c3": 1.0, "c4": 0.0},
        ]
    )
    df = cc_mod.fetch_capital_changes(client, "000001.SZ")
    assert df.height == 1
    assert df["event_date"].to_list() == [date(1991, 4, 3)]