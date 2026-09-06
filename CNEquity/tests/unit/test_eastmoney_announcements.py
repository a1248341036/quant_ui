"""EastMoney announcement index — pagination, mapping, completeness guards."""

from __future__ import annotations

from datetime import date

import pytest

from cnequity.adapters.eastmoney.announcements import (
    _symbol_from_em_code,
    fetch_announcement_index_eastmoney,
)


def test_symbol_from_em_code_maps_exchanges():
    assert _symbol_from_em_code("600519") == "600519.SH"
    assert _symbol_from_em_code("000001") == "000001.SZ"
    assert _symbol_from_em_code("920001") == "920001.BJ"


def test_symbol_from_em_code_rejects_non_a_and_malformed():
    assert _symbol_from_em_code("810001") is None
    assert _symbol_from_em_code("abc") is None


def _item(art: str, code: str, notice: str = "2024-06-28", **extra):
    return {
        "art_code": art,
        "codes": [{"stock_code": code, "market_code": "0", "short_name": "x"}],
        "columns": [{"column_code": "c", "column_name": "定期报告"}],
        "notice_date": f"{notice} 00:00:00",
        "title": f"标题{art}",
        **extra,
    }


class _FakeClient:
    """Serves canned pages keyed by page_index, mirroring the live endpoint."""

    def __init__(self, pages: list[list[dict]], total_hits: int | None = None):
        self.pages = pages
        self.total_hits = total_hits if total_hits is not None else sum(len(p) for p in pages)
        self.page_requests: list[int] = []
        self.closed = False

    def get(self, url, **kwargs):
        params = kwargs.get("params") or {}
        page = int(params["page_index"])
        self.page_requests.append(page)
        batch = self.pages[page - 1] if page - 1 < len(self.pages) else []
        payload = {"data": {"list": batch, "total_hits": self.total_hits}}
        return _FakeResponse(payload)

    def close(self):
        self.closed = True


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_fetch_maps_columns_and_paginates():
    pages = [
        [_item("AN1", "000001"), _item("AN2", "600519")],
        [_item("AN3", "000001")],
        [_item("AN4", "600519")],
    ]
    # 201 declared hits at 100/page → 3 expected pages, matching the page list.
    client = _FakeClient(pages, total_hits=201)
    df = fetch_announcement_index_eastmoney(date(2024, 6, 28), client=client)
    assert client.closed is False  # caller owns the client, must not be closed
    assert client.page_requests == [1, 2, 3]
    assert df.height == 4
    assert df.columns == [
        "announcement_id",
        "symbol",
        "title",
        "announce_date",
        "category",
        "url",
    ]
    row = df.filter(df["announcement_id"] == "AN1").row(0, named=True)
    assert row["symbol"] == "000001.SZ"
    assert row["category"] == "定期报告"
    assert row["url"] == "https://data.eastmoney.com/notices/detail/000001/AN1.html"
    assert row["announce_date"] == date(2024, 6, 28)


def test_fetch_dedupes_repeated_art_codes():
    pages = [[_item("AN1", "000001"), _item("AN1", "000001")]]
    df = fetch_announcement_index_eastmoney(date(2024, 6, 28), client=_FakeClient(pages))
    assert df.height == 1


def test_fetch_composite_ids_for_joint_filings():
    joint = _item("AN9", "000001")
    joint["codes"] = [
        {"stock_code": "000001", "market_code": "0", "short_name": "a"},
        {"stock_code": "600519", "market_code": "1", "short_name": "b"},
    ]
    df = fetch_announcement_index_eastmoney(date(2024, 6, 28), client=_FakeClient([[joint]]))
    assert sorted(df["announcement_id"].to_list()) == ["AN9", "AN9:600519"]
    assert sorted(df["symbol"].to_list()) == ["000001.SZ", "600519.SH"]


def test_fetch_rejects_date_mismatch():
    pages = [[_item("AN1", "000001", notice="2024-06-27")]]
    with pytest.raises(RuntimeError, match="does not match requested"):
        fetch_announcement_index_eastmoney(date(2024, 6, 28), client=_FakeClient(pages))


def test_fetch_rejects_empty_page_before_declared_end():
    # 250 hits → 3 expected pages, but the source only serves one then empties.
    client = _FakeClient(pages=[[_item("AN1", "000001")]], total_hits=250)
    with pytest.raises(RuntimeError, match="empty page before the reported end"):
        fetch_announcement_index_eastmoney(date(2024, 6, 28), client=client)


def test_fetch_returns_empty_frame_for_no_hits():
    df = fetch_announcement_index_eastmoney(date(2024, 6, 28), client=_FakeClient(pages=[]))
    assert df.height == 0


def test_fetch_skips_non_a_share_codes():
    row = _item("AN1", "000001")
    row["codes"] = [
        {"stock_code": "810001", "market_code": "0", "short_name": "bad"},
        {"stock_code": "600519", "market_code": "1", "short_name": "good"},
    ]
    df = fetch_announcement_index_eastmoney(date(2024, 6, 28), client=_FakeClient([[row]]))
    assert df.height == 1
    assert df["symbol"][0] == "600519.SH"
