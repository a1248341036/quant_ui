"""margin_trading backfill — date walk, curated-date resume, range bounds."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from cnequity.config import Config
from cnequity.steps.capital import _backfill_margin_trading


class _DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    def close(self) -> None:
        pass


def _fake_row(d: date) -> dict:
    return {
        "symbol": "600000.SH",
        "trade_date": d,
        "margin_balance": 1.0,
        "margin_buy": 1.0,
        "short_balance": 0.0,
        "short_sell_volume": 0.0,
    }


def _fake_rows(d: date, count: int = 50) -> pl.DataFrame:
    return pl.DataFrame([{**_fake_row(d), "symbol": f"{600000 + i:06d}.SH"} for i in range(count)])


def _setup(monkeypatch, cfg: Config, *, empty_days: set[date] = frozenset()):
    fetched: list[date] = []

    def fake_fetch(d: date, *, client=None) -> pl.DataFrame:
        fetched.append(d)
        if d in empty_days:
            return pl.DataFrame()
        return _fake_rows(d)

    monkeypatch.setattr("cnequity.steps.capital.fetch_margin_trading", fake_fetch)
    monkeypatch.setattr("cnequity.adapters.eastmoney.em_auth.EastMoneyClient", _DummyClient)
    monkeypatch.setattr(cfg, "rate_limit", lambda source: None)
    return fetched


def test_walks_range_and_stages_rows(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    cfg._backfill_start = date(2026, 6, 1)
    cfg._backfill_end = date(2026, 6, 5)
    fetched = _setup(monkeypatch, cfg)

    out = _backfill_margin_trading(cfg, date(2026, 7, 1), "run-1")

    # no curated calendar in tmp lake → Mon–Fri fallback: 6/1..6/5 = 5 weekdays
    assert fetched == [date(2026, 6, d) for d in (1, 2, 3, 4, 5)]
    assert out["days_fetched"] == 5
    assert out["rows_written"] == 250
    staged = list((cfg.staging_root / "margin_trading").glob("**/*.parquet"))
    assert len(staged) == 1
    df = pl.read_parquet(staged[0])
    assert df.height == 250
    assert set(df["source"]) == {"eastmoney"}


def test_skips_dates_already_curated(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    cfg._backfill_start = date(2026, 6, 1)
    cfg._backfill_end = date(2026, 6, 3)
    fetched = _setup(monkeypatch, cfg)

    curated = cfg.curated_root / "margin_trading" / "trade_date=2026-06-02"
    curated.mkdir(parents=True)
    pl.DataFrame(
        [{**_fake_row(date(2026, 6, 2)), "symbol": f"{i:06d}.SH"} for i in range(50)]
    ).write_parquet(curated / "part-0.parquet")

    out = _backfill_margin_trading(cfg, date(2026, 7, 1), "run-1")

    assert date(2026, 6, 2) not in fetched
    assert out["days_fetched"] == 2
    assert out["days_skipped"] == 1


def test_partial_existing_day_is_not_considered_complete(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    cfg._backfill_start = date(2026, 6, 1)
    cfg._backfill_end = date(2026, 6, 2)
    fetched = _setup(monkeypatch, cfg)

    curated = cfg.curated_root / "margin_trading" / "trade_date=2026-06-02"
    curated.mkdir(parents=True)
    pl.DataFrame([_fake_row(date(2026, 6, 2))]).write_parquet(curated / "partial.parquet")

    out = _backfill_margin_trading(cfg, date(2026, 7, 1), "run-1")

    assert date(2026, 6, 2) in fetched
    assert out["days_skipped"] == 0


def test_empty_days_reported_not_fatal(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    cfg._backfill_start = date(2026, 6, 1)
    cfg._backfill_end = date(2026, 6, 2)
    _setup(monkeypatch, cfg, empty_days={date(2026, 6, 1)})

    out = _backfill_margin_trading(cfg, date(2026, 7, 1), "run-1")

    assert out["days_empty"] == 1
    assert out["days_fetched"] == 1
    assert out["rows_written"] == 50
    finding = out["context_updates"]["audit_findings"][0]
    assert finding["check"] == "backfill_empty_days"
    assert finding["severity"] == "warning"
    assert finding["sample_dates"] == ["2026-06-01"]


def test_partial_response_is_not_staged_and_is_retryable(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    cfg._backfill_start = date(2026, 6, 1)
    cfg._backfill_end = date(2026, 6, 1)
    monkeypatch.setattr(
        "cnequity.steps.capital.fetch_margin_trading",
        lambda d, **kwargs: pl.DataFrame([_fake_row(d)]),
    )
    monkeypatch.setattr("cnequity.adapters.eastmoney.em_auth.EastMoneyClient", _DummyClient)
    monkeypatch.setattr(cfg, "rate_limit", lambda source: None)

    out = _backfill_margin_trading(cfg, date(2026, 7, 1), "run-partial")

    assert out["status"] == "warning"
    assert out["days_fetched"] == 0
    assert out["failed_days"] == 1
    assert out["rows_written"] == 0
    assert not list(cfg.staging_root.glob("margin_trading/**/*.parquet"))
    finding = out["context_updates"]["audit_findings"][0]
    assert finding["check"] == "backfill_incomplete_days"
    assert finding["days"] == [{"trade_date": "2026-06-01", "symbols": 1}]


def test_rejects_rows_from_a_different_requested_date(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    cfg._backfill_start = date(2026, 6, 1)
    cfg._backfill_end = date(2026, 6, 2)
    _setup(monkeypatch, cfg)

    def wrong_date_fetch(d: date, *, client=None) -> pl.DataFrame:
        if d == date(2026, 6, 1):
            return _fake_rows(d)
        return pl.DataFrame([_fake_row(d.replace(day=d.day + 1))])

    monkeypatch.setattr("cnequity.steps.capital.fetch_margin_trading", wrong_date_fetch)

    with pytest.raises(RuntimeError, match="different or invalid trade_date"):
        _backfill_margin_trading(cfg, date(2026, 7, 1), "run-wrong-date")

    staged = list(cfg.staging_root.glob("margin_trading/**/*.parquet"))
    assert len(staged) == 1
    assert pl.read_parquet(staged[0])["trade_date"].to_list() == [date(2026, 6, 1)] * 50


def test_end_clamped_to_trade_date(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    cfg._backfill_start = date(2026, 6, 29)
    cfg._backfill_end = date(2026, 7, 10)
    fetched = _setup(monkeypatch, cfg)

    _backfill_margin_trading(cfg, date(2026, 6, 30), "run-1")

    assert fetched == [date(2026, 6, 29), date(2026, 6, 30)]


def test_parallel_workers_fetch_all_days(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    cfg._backfill_start = date(2026, 6, 1)
    cfg._backfill_end = date(2026, 6, 12)
    cfg._backfill_workers = 3
    fetched = _setup(monkeypatch, cfg)

    out = _backfill_margin_trading(cfg, date(2026, 7, 1), "run-1")

    expected = {date(2026, 6, d) for d in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12)}
    assert set(fetched) == expected
    assert out["days_fetched"] == 10
    staged = pl.concat(
        [pl.read_parquet(f) for f in (cfg.staging_root / "margin_trading").glob("**/*.parquet")]
    )
    assert set(staged["trade_date"]) == expected
