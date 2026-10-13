# -*- coding: utf-8 -*-
"""report_rc backfill (Tushare full-market date-range): contract, backfill bypasses
quarterly cadence, PK dedup. Old impl was a no-date full-market snapshot leaving a
single report_date=2026 partition; now _make_date_range_step + `cne backfill`.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import cnequity.steps  # noqa: F401  # trigger step registration (tushare_wide imported last)
from cnequity.config import Config
from cnequity.domain import schemas
from cnequity.domain.datasets import get_dataset
from cnequity.orchestrator.registry import get_step
from cnequity.steps import tushare_wide as tw
from cnequity.storage.state import StateStore


@pytest.fixture
def cfg(tmp_path):
    c = Config(
        data_root=tmp_path / "data",
        external_tushare_wide_token="test-token",
        external_tushare_wide_interval=0.0,
    )
    c.staging_root.mkdir(parents=True, exist_ok=True)
    c.meta_root.mkdir(parents=True, exist_ok=True)
    return c


def _rc_rows(count: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": [f"{600000 + i % 5000:06d}.SH" for i in range(count)],
            "name": ["test"] * count,
            "report_date": [date(2026, 6, 30)] * count,
            "org_name": ["test-sec"] * count,
            "report_title": ["deep-report"] * count,
            "rating": ["buy"] * count,
            "roe": [12.5] * count,
            "eps": [1.2] * count,
        }
    )


def _watermark(cfg, d: date) -> None:
    StateStore(cfg.meta_root).set_date("report_rc", d)


# --- contract ---------------------------------------------------------


def test_report_rc_contract_registered():
    spec = get_dataset("report_rc")
    assert spec.primary_source == "tushare"
    assert spec.partition_col == "report_date"
    assert spec.cadence == "quarterly"
    assert spec.fetch_semantics == "by_date"  # supports `cne backfill`


def test_report_rc_step_registered_in_fundamentals_group():
    entry = get_step("report_rc")
    assert entry.group == "fundamentals"
    assert entry.depends_on == ["instruments"]


# --- backfill bypasses cadence ---------------------------------------


def test_backfill_bypasses_cadence_skip_and_uses_explicit_window(monkeypatch, cfg):
    """Same-quarter watermark would cadence-skip the old snapshot impl; backfill
    must bypass it and hand the explicit _backfill_start/_backfill_end window to
    the dedicated fetch (verified via _ts_date strings on the first page call)."""
    _watermark(cfg, date(2026, 10, 12))  # same quarter (Q4) as trade_date
    cfg._backfill = True
    cfg._backfill_start = date(2026, 1, 1)
    cfg._backfill_end = date(2026, 1, 1)  # single window -> deterministic

    calls: list[dict] = []
    pages = [_rc_rows(3)]

    def fake_fetch(pro, api, interval=0.0, **kw):
        calls.append(kw)
        return pages.pop(0) if pages else pl.DataFrame()

    monkeypatch.setattr(tw, "_fetch_with_retry", fake_fetch)

    def fake_finalize(combined, dataset, run_id, trade_date, config):
        return {"rows_read": combined.height, "rows_written": combined.height}

    monkeypatch.setattr(tw, "_finalize_and_write", fake_finalize)
    out = get_step("report_rc").fn(cfg, date(2026, 10, 13), "run-backfill", {})
    assert out["rows_written"] == 3
    assert calls[0]["start_date"] == "20260101"
    assert calls[0]["end_date"] == "20260101"


def test_daily_mode_still_respects_cadence_skip(monkeypatch, cfg):
    """Non-backfill daily runs still follow quarterly cadence: same-quarter
    watermark must skip without touching the wire."""
    _watermark(cfg, date(2026, 10, 12))
    cfg._backfill = False
    called = {"n": 0}

    def fake_range(*a, **k):
        called["n"] += 1
        return _rc_rows(3)

    monkeypatch.setattr(tw, "_fetch_report_rc_range", fake_range)
    out = get_step("report_rc").fn(cfg, date(2026, 10, 13), "run-daily", {})
    assert called["n"] == 0
    assert out["rows_written"] == 0
    assert out["status"] == "success"


def test_backfill_dedups_on_primary_keys(monkeypatch, cfg):
    """Range fetch dedups by PRIMARY_KEYS (symbol/report_date/org_name/report_title)."""
    cfg._backfill = True
    cfg._backfill_start = date(2026, 1, 1)
    cfg._backfill_end = date(2026, 1, 1)
    dup = pl.concat([_rc_rows(4), _rc_rows(4)], how="vertical")  # 8 rows, 4 unique PK
    pages = [dup]

    def fake_fetch(pro, api, interval=0.0, **kw):
        return pages.pop(0) if pages else pl.DataFrame()

    monkeypatch.setattr(tw, "_fetch_with_retry", fake_fetch)
    captured: dict = {}

    def fake_finalize(combined, dataset, run_id, trade_date, config):
        captured["rows"] = combined.height
        return {"rows_read": combined.height, "rows_written": combined.height}

    monkeypatch.setattr(tw, "_finalize_and_write", fake_finalize)
    get_step("report_rc").fn(cfg, date(2026, 10, 13), "run-dedup", {})
    assert captured["rows"] == 4


def test_report_rc_windows_split_by_quarter():
    w = tw._report_rc_windows(date(2024, 1, 1), date(2024, 12, 31))
    assert len(w) >= 4
    assert w[0][0] == date(2024, 1, 1)
    assert w[-1][1] == date(2024, 12, 31)


def test_coerce_report_rc_page_nulls_string_numerics():
    """A page whose max_price is entirely String (middleware returns '' for empty)
    must be coerced to Float64 with '' -> null (column dtype is uniform within a
    page; the drift happens across pages)."""
    page = pl.DataFrame(
        {
            "ts_code": ["600000.SH", "600001.SH"],
            "max_price": ["10.5", ""],
            "name": ["a", "b"],
        }
    )
    out = tw._coerce_report_rc_page(page)
    assert out["max_price"].dtype == pl.Float64
    assert out["max_price"].to_list() == [10.5, None]
    assert out["name"].dtype == pl.Utf8


def test_report_rc_numeric_cols_cover_schema_float64s():
    """页面级 cast 集必须覆盖 scheme 全部数值列（op_pr/op_rt/np/rd/ev_ebitda 曾漏掉导致
    跨页 String/Float64 漂移、pl.concat 报 SchemaError）。"""
    expected = {
        n for n, dt in schemas.REPORTRC_SCHEMA.items()
        if dt in (pl.Float64, pl.Int64, pl.Int32)
    }
    assert expected <= set(tw._REPORT_RC_NUMERIC_COLS)


def test_fetch_report_rc_range_coerces_schema_numeric_drift(monkeypatch):
    """op_pr/op_rt 等 schema 数值列跨页漂移（String('') vs Float64）时不再 vstack 崩溃。"""
    pages = [
        pl.DataFrame({"ts_code": ["600000.SH"], "op_pr": [3.0], "op_rt": [5.0]}),
        pl.DataFrame({"ts_code": ["600001.SH"], "op_pr": [""], "op_rt": [""]}),
    ]

    fake = lambda *a, **k: pages.pop(0) if pages else pl.DataFrame()  # noqa: E731

    monkeypatch.setattr(tw, "_fetch_with_retry", fake)
    out = tw._fetch_report_rc_range(
        object(), "report_rc", interval=0.0, start=date(2026, 1, 1), end=date(2026, 1, 1)
    )
    assert out.height == 2
    assert out["op_pr"].dtype == pl.Float64
    assert out["op_pr"].to_list() == [3.0, None]
    assert out["op_rt"].to_list() == [5.0, None]


def test_fetch_report_rc_range_coerces_and_concats_pages(monkeypatch):
    """跨页出现 max_price String（中间件空值）时不再 SchemError：页级 cast 后 concat。"""
    pages = [
        pl.DataFrame({"ts_code": ["600000.SH"], "max_price": [10.5]}),
        pl.DataFrame({"ts_code": ["600001.SH"], "max_price": [""]}),
    ]

    fake = lambda *a, **k: pages.pop(0) if pages else pl.DataFrame()  # noqa: E731

    monkeypatch.setattr(tw, "_fetch_with_retry", fake)
    out = tw._fetch_report_rc_range(
        object(), "report_rc", interval=0.0, start=date(2026, 1, 1), end=date(2026, 1, 1)
    )
    assert out.height == 2
    assert out["max_price"].dtype == pl.Float64
    assert out["max_price"].to_list() == [10.5, None]
