"""universe_survivorship_findings — does the lake keep the names that died?"""

from datetime import date, timedelta

import polars as pl

from cnequity.config import Config
from cnequity.quality.cross_checks import (
    RETIRED_GAP_DAYS,
    universe_survivorship_findings,
)

_START = date(2020, 1, 6)
_END = date(2024, 6, 28)


def _write_bars(root, series: dict[str, tuple[date, date]]) -> None:
    """series: symbol -> (first_bar, last_bar); one bar every 7 days between."""
    by_day: dict[date, list[str]] = {}
    for sym, (first, last) in series.items():
        d = first
        while d <= last:
            by_day.setdefault(d, []).append(sym)
            d += timedelta(days=7)
    for d, syms in by_day.items():
        part = root / "curated" / "daily_bars" / f"trade_date={d.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"symbol": syms, "trade_date": [d] * len(syms), "volume": [100] * len(syms)}
        ).write_parquet(part / "part-merged.parquet")


def _write_instruments(root, rows: dict[str, date | None]) -> None:
    part = root / "curated" / "instruments"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": list(rows),
            "delist_date": pl.Series(list(rows.values()), dtype=pl.Date),
        }
    ).write_parquet(part / "part-merged.parquet")


def _by_check(findings: list[dict]) -> dict[str, dict]:
    return {f["check"]: f for f in findings}


def test_no_findings_when_daily_bars_absent(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    assert universe_survivorship_findings(cfg, _END) == []


def test_short_span_is_not_judged(tmp_path):
    """Over a few months a real market may genuinely retire nobody."""
    cfg = Config(data_root=tmp_path / "data")
    short_start = _END - timedelta(days=200)
    _write_bars(cfg.data_root, {"A": (short_start, _END), "B": (short_start, _END)})

    assert universe_survivorship_findings(cfg, _END) == []


def test_flags_lake_where_no_symbol_ever_stops_trading(tmp_path):
    """The current-listing-snapshot backfill signature: everyone is a survivor."""
    cfg = Config(data_root=tmp_path / "data")
    _write_bars(cfg.data_root, {sym: (_START, _END) for sym in ("A", "B", "C")})

    findings = universe_survivorship_findings(cfg, _END)

    assert len(findings) == 1
    assert findings[0]["check"] == "universe_survivorship_absent"
    assert findings[0]["severity"] == "error"
    assert findings[0]["symbols"] == 3
    assert findings[0]["span_years"] > 2


def test_retired_names_reported_and_clean_when_marked_delisted(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    dead_last = _END - timedelta(days=RETIRED_GAP_DAYS + 30)
    _write_bars(cfg.data_root, {"A": (_START, _END), "DEAD": (_START, dead_last)})
    _write_instruments(cfg.data_root, {"A": None, "DEAD": dead_last})

    checks = _by_check(universe_survivorship_findings(cfg, _END))

    assert "universe_survivorship_absent" not in checks
    assert "retired_symbol_missing_delist_date" not in checks
    assert checks["universe_survivorship"]["retired_symbols"] == 1
    assert checks["universe_survivorship"]["total_symbols"] == 2


def test_zero_volume_placeholder_does_not_hide_retired_symbol(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    dead_last = _END - timedelta(days=RETIRED_GAP_DAYS + 30)
    _write_bars(cfg.data_root, {"A": (_START, _END), "DEAD": (_START, dead_last)})

    tail = cfg.data_root / "curated" / "daily_bars" / f"trade_date={_END.isoformat()}"
    tail.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["A", "DEAD"],
            "trade_date": [_END, _END],
            "volume": [100, 0],
        }
    ).write_parquet(tail / "part-tail.parquet")

    checks = _by_check(universe_survivorship_findings(cfg, _END))

    assert "universe_survivorship_absent" not in checks
    assert checks["universe_survivorship"]["retired_symbols"] == 1


def test_flags_retired_symbol_with_no_delist_date(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    dead_last = _END - timedelta(days=RETIRED_GAP_DAYS + 30)
    _write_bars(cfg.data_root, {"A": (_START, _END), "DEAD": (_START, dead_last)})
    _write_instruments(cfg.data_root, {"A": None, "DEAD": None})

    checks = _by_check(universe_survivorship_findings(cfg, _END))
    finding = checks["retired_symbol_missing_delist_date"]

    assert finding["severity"] == "warning"
    assert finding["unmarked_count"] == 1
    assert finding["sample"][0]["symbol"] == "DEAD"


def test_display_only_etf_is_not_research_survivorship_gap(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    dead_last = _END - timedelta(days=RETIRED_GAP_DAYS + 30)
    _write_bars(cfg.data_root, {"A": (_START, _END), "159915.SZ": (_START, dead_last)})
    _write_instruments(cfg.data_root, {"A": None, "159915.SZ": None})
    instruments = cfg.data_root / "curated" / "instruments" / "part-merged.parquet"
    pl.DataFrame(
        {"symbol": ["A", "159915.SZ"], "delist_date": [None, None], "asset_type": ["stock", "etf"]},
        schema={"symbol": pl.Utf8, "delist_date": pl.Date, "asset_type": pl.Utf8},
    ).write_parquet(instruments)

    checks = _by_check(universe_survivorship_findings(cfg, _END))

    assert "retired_symbol_missing_delist_date" not in checks


def test_ordinary_suspension_is_not_counted_as_retired(tmp_path):
    """A gap shorter than the retirement threshold is a halt, not a delisting."""
    cfg = Config(data_root=tmp_path / "data")
    halted_last = _END - timedelta(days=RETIRED_GAP_DAYS - 30)
    _write_bars(cfg.data_root, {"A": (_START, _END), "HALT": (_START, halted_last)})

    findings = universe_survivorship_findings(cfg, _END)

    assert len(findings) == 1
    assert findings[0]["check"] == "universe_survivorship_absent"
