"""EastMoney regulatory events — keyword filter, classification, dedupe."""

from __future__ import annotations

from datetime import date

import pytest

from cnequity.adapters.eastmoney.regulatory import fetch_regulatory_events_eastmoney


def _item(art: str, code: str, title: str):
    return {
        "art_code": art,
        "codes": [{"stock_code": code, "market_code": "0", "short_name": "x"}],
        "columns": [{"column_code": "c", "column_name": "监管"}],
        "notice_date": "2024-06-28 00:00:00",
        "title": title,
    }


class _FakeClient:
    def __init__(self, pages: list[list[dict]], total_hits: int | None = None):
        self.pages = pages
        self.total_hits = total_hits if total_hits is not None else sum(len(p) for p in pages)
        self.closed = False

    def get(self, url, **kwargs):
        params = kwargs.get("params") or {}
        page = int(params["page_index"])
        batch = self.pages[page - 1] if page - 1 < len(self.pages) else []
        return _FakeResponse({"data": {"list": batch, "total_hits": self.total_hits}})

    def close(self):
        self.closed = True


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_fetch_filters_by_keyword_and_classifies():
    pages = [
        [
            _item("AN1", "000001", "关于对某公司给予通报批评处分的决定"),
            _item("AN2", "600519", "2023年年度报告"),
            _item("AN3", "000002", "收到中国证监会立案告知书"),
        ],
        [_item("AN4", "600519", "收到监管函的公告")],
    ]
    # 201 declared hits at 100/page → 2 expected pages, matching the page list.
    df = fetch_regulatory_events_eastmoney(
        date(2024, 6, 28), client=_FakeClient(pages, total_hits=201)
    )
    assert df.height == 3  # AN2 年报 filtered out
    assert set(df["event_id"]) == {"reg-AN1", "reg-AN3", "reg-AN4"}
    by_id = {r["event_id"]: r for r in df.iter_rows(named=True)}
    assert by_id["reg-AN1"]["event_type"] == "disciplinary"
    assert by_id["reg-AN3"]["event_type"] == "investigation"
    assert by_id["reg-AN4"]["event_type"] == "regulatory_letter"
    assert by_id["reg-AN1"]["symbol"] == "000001.SZ"
    assert df.columns == ["event_id", "symbol", "event_date", "event_type", "title"]


def test_fetch_empty_when_no_keyword_hits():
    df = fetch_regulatory_events_eastmoney(
        date(2024, 6, 28), client=_FakeClient([[_item("AN1", "000001", "半年度报告")]])
    )
    assert df.height == 0


def test_fetch_rejects_date_mismatch():
    bad = _item("AN1", "000001", "处罚决定")
    bad["notice_date"] = "2024-06-27 00:00:00"
    with pytest.raises(RuntimeError, match="does not match requested"):
        fetch_regulatory_events_eastmoney(date(2024, 6, 28), client=_FakeClient([[bad]]))


def test_fetch_dedupes_and_composite_ids():
    joint = _item("AN9", "000001", "警示函")
    joint["codes"] = [
        {"stock_code": "000001", "market_code": "0", "short_name": "a"},
        {"stock_code": "600519", "market_code": "1", "short_name": "b"},
    ]
    pages = [[joint, joint]]
    df = fetch_regulatory_events_eastmoney(date(2024, 6, 28), client=_FakeClient(pages))
    assert sorted(df["event_id"].to_list()) == ["reg-AN9", "reg-AN9:600519"]
