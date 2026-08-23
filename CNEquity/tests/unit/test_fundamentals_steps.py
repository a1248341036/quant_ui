"""Offline coverage for fundamentals step wrappers and valuation backfill."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import cnequity.steps  # noqa: F401
from cnequity.config import Config
from cnequity.steps import fundamentals as fund
from cnequity.steps.common import load_bar_universe


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_root=tmp_path / "data")
    c.staging_root.mkdir(parents=True)
    return c


def test_financial_statement_items_disabled(cfg):
    cfg.sources["eastmoney"] = False
    with pytest.raises(RuntimeError, match="eastmoney source disabled"):
        fund.step_financial_statement_items(cfg, date(2024, 6, 28), "run-1", {})


def test_financial_statement_items_empty(cfg, monkeypatch):
    monkeypatch.setattr(
        fund,
        "fetch_financial_statement_items",
        lambda trade_date, backfill=False, config=None: pl.DataFrame(),
    )
    result = fund.step_financial_statement_items(cfg, date(2024, 6, 28), "run-1", {})
    assert result == {"rows_read": 0, "rows_written": 0}


def test_financial_statement_items_backfill_surfaces_missing_periods(cfg, monkeypatch):
    cfg._backfill = True
    cfg._backfill_start = date(2024, 1, 1)
    cfg._backfill_end = date(2024, 6, 30)
    monkeypatch.setattr(fund, "fetch_financial_statement_items", lambda *a, **k: pl.DataFrame())

    result = fund.step_financial_statement_items(cfg, date(2026, 6, 30), "run-fsi-gap", {})

    assert result["status"] == "warning"
    assert result["missing_periods"] == 2
    assert result["context_updates"]["audit_findings"][0]["check"] == (
        "backfill_missing_report_periods"
    )


def test_financial_statement_items_writes_staging(cfg, monkeypatch):
    seen = {}

    def fake_fetch(trade_date, backfill=False, config=None):
        seen["backfill"] = backfill
        return pl.DataFrame(
            {
                "symbol": ["600519.SH"] * 4,
                "report_period": ["2024Q1"] * 4,
                "statement_type": ["income", "indicator", "balance", "cashflow"],
                "item_code": ["revenue", "roe", "total_assets", "net_cash_operate"],
                "item_value": [1_000_000.0, 0.12, 1_000_000.0, 100_000.0],
                "announce_date": [date(2024, 4, 20)] * 4,
            }
        )

    monkeypatch.setattr(fund, "fetch_financial_statement_items", fake_fetch)
    cfg._backfill = True
    cfg._backfill_start = date(2024, 1, 1)
    cfg._backfill_end = date(2024, 3, 31)
    result = fund.step_financial_statement_items(cfg, date(2024, 6, 28), "run-fsi", {})
    assert seen["backfill"] is True
    assert result["rows_written"] == 4
    assert result.get("status") is None
    assert list(cfg.staging_root.glob("financial_statement_items/**/*.parquet"))


def test_financial_statement_items_backfill_surfaces_partial_report_families(cfg, monkeypatch):
    cfg._backfill = True
    cfg._backfill_start = date(2024, 1, 1)
    cfg._backfill_end = date(2024, 3, 31)
    monkeypatch.setattr(
        fund,
        "fetch_financial_statement_items",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "symbol": ["600519.SH"],
                "report_period": ["2024Q1"],
                "statement_type": ["income"],
                "item_code": ["revenue"],
                "item_value": [1_000_000.0],
                "announce_date": [date(2024, 4, 20)],
            }
        ),
    )

    result = fund.step_financial_statement_items(cfg, date(2024, 6, 28), "run-fsi-partial", {})

    assert result["status"] == "warning"
    assert result["missing_statement_periods"] == 1
    finding = result["context_updates"]["audit_findings"][0]
    assert finding["check"] == "backfill_missing_statement_types"
    assert finding["missing_statement_types"] == [
        {"report_period": "2024Q1", "missing": ["balance", "cashflow", "indicator"]}
    ]


def test_financial_statement_items_backfill_surfaces_missing_income_statement(cfg, monkeypatch):
    """A missing income statement must surface too, not just balance/cashflow.

    fetch_financial_statement_items issues four independent requests -
    income, indicator, balance, cashflow - so a period with only
    balance/cashflow present is still incomplete.
    """
    cfg._backfill = True
    cfg._backfill_start = date(2024, 1, 1)
    cfg._backfill_end = date(2024, 3, 31)
    monkeypatch.setattr(
        fund,
        "fetch_financial_statement_items",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "symbol": ["600519.SH", "600519.SH"],
                "report_period": ["2024Q1", "2024Q1"],
                "statement_type": ["balance", "cashflow"],
                "item_code": ["total_assets", "net_cash_operate"],
                "item_value": [1_000_000.0, 100_000.0],
                "announce_date": [date(2024, 4, 20), date(2024, 4, 20)],
            }
        ),
    )

    result = fund.step_financial_statement_items(cfg, date(2024, 6, 28), "run-fsi-income", {})

    assert result["status"] == "warning"
    finding = result["context_updates"]["audit_findings"][0]
    assert finding["check"] == "backfill_missing_statement_types"
    assert finding["missing_statement_types"] == [
        {"report_period": "2024Q1", "missing": ["income", "indicator"]}
    ]


def test_valuation_metrics_disabled(cfg):
    cfg.sources["eastmoney"] = False
    with pytest.raises(RuntimeError, match="eastmoney source disabled"):
        fund.step_valuation_metrics(cfg, date(2024, 6, 28), "run-1", {})


def test_valuation_metrics_rejects_empty_snapshot(cfg, monkeypatch):
    monkeypatch.setattr(fund, "load_bar_universe", lambda _config: {"600519.SH"})
    monkeypatch.setattr(
        fund,
        "fetch_valuation_metrics",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="valuation_metrics: no rows returned"):
        fund.step_valuation_metrics(cfg, date(2024, 6, 28), "run-empty", {})


def test_load_bar_universe_ignores_zero_volume_placeholders(cfg):
    part = cfg.curated_root / "daily_bars" / "trade_date=2024-06-28"
    part.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "trade_date": [date(2024, 6, 28)] * 2,
            "volume": [100, 0],
        }
    ).write_parquet(part / "part-000.parquet")

    assert load_bar_universe(cfg) == {"600519.SH"}


def test_load_bar_universe_keeps_legacy_rows_in_a_mixed_schema_lake(cfg):
    root = cfg.curated_root / "daily_bars"
    root.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH"], "trade_date": [date(2024, 6, 27)]}).write_parquet(
        root / "legacy.parquet"
    )
    pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": [date(2024, 6, 28)],
            "volume": [0],
        }
    ).write_parquet(root / "current.parquet")

    assert load_bar_universe(cfg) == {"600519.SH"}


def test_symbols_needing_backfill_does_not_count_duplicate_rows(cfg):
    part = cfg.curated_root / "valuation_metrics" / "trade_date=2024-06"
    part.mkdir(parents=True)
    dates = [date(2024, 6, day) for day in range(1, 9)]
    rows = []
    for index, day in enumerate(dates):
        rows.append(
            {
                "symbol": "600519.SH",
                "trade_date": day,
                "float_mv": 1.0 if index < 6 else None,
                "total_mv": 1.0 if index < 6 else None,
                "source": "baostock",
                "fetched_at": f"2024-06-{day.day:02d}T00:00:00+00:00",
            }
        )
    rows.extend(
        {**rows[index], "fetched_at": f"2024-06-{index + 1:02d}T01:00:00+00:00"} for index in (0, 1)
    )
    pl.DataFrame(rows).write_parquet(part / "part-000.parquet")

    # Eight unique dates contain only six complete market-cap rows (75%). The
    # two retries must not inflate this to the 80% skip threshold.
    assert fund._symbols_needing_backfill(cfg, ["600519.SH"]) == ["600519.SH"]


def test_backfill_valuation_locked_nothing_to_do(cfg, monkeypatch):
    monkeypatch.setattr(
        "cnequity.storage.valuation_orphans.purge_valuation_orphan_symbols",
        lambda config: {"purged": 0},
    )
    monkeypatch.setattr(fund, "load_symbols", lambda config: ["600519.SH"])
    monkeypatch.setattr(fund, "load_bar_universe", lambda config: {"600519.SH"})
    monkeypatch.setattr(fund, "_symbols_needing_backfill", lambda config, universe, **kwargs: [])
    monkeypatch.setattr(fund, "_valuation_history_end", lambda config, trade_date: date(2024, 6, 1))
    result = fund._backfill_valuation_metrics_locked(cfg, date(2024, 6, 28), "run-v")
    assert result["rows_written"] == 0
    assert "already backfilled" in result["note"]


def test_backfill_valuation_locked_history_end_before_start(cfg, monkeypatch):
    monkeypatch.setattr(
        "cnequity.storage.valuation_orphans.purge_valuation_orphan_symbols",
        lambda config: {"purged": 0},
    )
    monkeypatch.setattr(fund, "load_symbols", lambda config: ["600519.SH"])
    monkeypatch.setattr(fund, "load_bar_universe", lambda config: {"600519.SH"})
    monkeypatch.setattr(
        fund,
        "_symbols_needing_backfill",
        lambda config, universe, **kwargs: ["600519.SH"],
    )
    monkeypatch.setattr(fund, "_valuation_history_end", lambda config, trade_date: date(2010, 1, 1))
    result = fund._backfill_valuation_metrics_locked(cfg, date(2024, 6, 28), "run-v")
    assert "history_end before backfill start" in result["note"]


def test_backfill_valuation_locked_writes_chunks(cfg, monkeypatch):
    monkeypatch.setattr(
        "cnequity.storage.valuation_orphans.purge_valuation_orphan_symbols",
        lambda config: {"purged": 1},
    )
    monkeypatch.setattr(fund, "load_symbols", lambda config: ["600519.SH", "000001.SZ"])
    monkeypatch.setattr(fund, "load_bar_universe", lambda config: {"600519.SH", "000001.SZ"})
    monkeypatch.setattr(
        fund,
        "_symbols_needing_backfill",
        lambda config, universe, **kwargs: ["600519.SH", "000001.SZ"],
    )
    monkeypatch.setattr(fund, "_valuation_history_end", lambda config, trade_date: date(2024, 6, 1))

    def fake_history(batch, start, end, config=None):
        df = pl.DataFrame(
            {
                "symbol": batch,
                "trade_date": [date(2024, 1, 2)] * len(batch),
                "pe_ttm": [10.0] * len(batch),
                "pb": [1.0] * len(batch),
                "ps_ttm": [2.0] * len(batch),
                "total_mv": [1e9] * len(batch),
                "float_mv": [1e9] * len(batch),
            }
        )
        return df, []

    monkeypatch.setattr(
        "cnequity.adapters.baostock.valuation.fetch_valuation_history",
        fake_history,
    )
    # Shrink chunk size so the loop body runs once with our tiny universe.
    monkeypatch.setattr(fund, "_VALUATION_BACKFILL_CHUNK", 50)
    result = fund._backfill_valuation_metrics_locked(cfg, date(2024, 6, 28), "run-chunk")
    assert result["rows_written"] == 2
    assert result["symbols_todo"] == 2
    assert list(cfg.staging_root.glob("valuation_metrics/**/batch-00000/*.parquet")) or list(
        cfg.staging_root.glob("valuation_metrics/**/*.parquet")
    )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"trade_date": date(2015, 12, 31)}, "outside requested window"),
        ({"symbol": "000001.SZ"}, "unexpected symbol"),
    ],
)
def test_backfill_valuation_locked_rejects_out_of_scope_rows(cfg, monkeypatch, update, message):
    monkeypatch.setattr(
        "cnequity.storage.valuation_orphans.purge_valuation_orphan_symbols",
        lambda config: {"purged": 0},
    )
    monkeypatch.setattr(fund, "load_symbols", lambda config: ["600519.SH"])
    monkeypatch.setattr(fund, "load_bar_universe", lambda config: {"600519.SH"})
    monkeypatch.setattr(
        fund,
        "_symbols_needing_backfill",
        lambda config, universe, **kwargs: ["600519.SH"],
    )
    monkeypatch.setattr(fund, "_valuation_history_end", lambda config, trade_date: date(2024, 6, 1))

    def fake_history(batch, start, end, config=None):
        row = {
            "symbol": "600519.SH",
            "trade_date": date(2024, 1, 2),
            "pe_ttm": 10.0,
            "pb": 1.0,
            "ps_ttm": 2.0,
            "total_mv": 1e9,
            "float_mv": 1e9,
        }
        row.update(update)
        return pl.DataFrame([row]), []

    monkeypatch.setattr(
        "cnequity.adapters.baostock.valuation.fetch_valuation_history", fake_history
    )
    with pytest.raises(RuntimeError, match=message):
        fund._backfill_valuation_metrics_locked(cfg, date(2024, 6, 28), "run-invalid")
    assert not list(cfg.staging_root.glob("valuation_metrics/**/*.parquet"))


def test_backfill_valuation_locked_aborts_on_runtime_error(cfg, monkeypatch):
    monkeypatch.setattr(
        "cnequity.storage.valuation_orphans.purge_valuation_orphan_symbols",
        lambda config: {"purged": 0},
    )
    monkeypatch.setattr(fund, "load_symbols", lambda config: ["600519.SH"])
    monkeypatch.setattr(fund, "load_bar_universe", lambda config: {"600519.SH"})
    monkeypatch.setattr(
        fund,
        "_symbols_needing_backfill",
        lambda config, universe, **kwargs: ["600519.SH"],
    )
    monkeypatch.setattr(fund, "_valuation_history_end", lambda config, trade_date: date(2024, 6, 1))

    def boom(*a, **k):
        raise RuntimeError("baostock banned")

    monkeypatch.setattr(
        "cnequity.adapters.baostock.valuation.fetch_valuation_history",
        boom,
    )
    result = fund._backfill_valuation_metrics_locked(cfg, date(2024, 6, 28), "run-abort")
    assert result["rows_written"] == 0
    assert "baostock banned" in result["aborted"]
    finding = result["context_updates"]["audit_findings"][0]
    assert finding["code"] == "baostock_backfill_incomplete"


def test_shareholder_backfill_surfaces_empty_windows(cfg, monkeypatch):
    cfg._backfill = True
    cfg._backfill_start = date(2024, 1, 1)
    cfg._backfill_end = date(2024, 12, 31)
    monkeypatch.setattr(
        "cnequity.adapters.eastmoney.shareholders.fetch_share_structure",
        lambda *args, **kwargs: pl.DataFrame(),
    )

    result = fund.step_share_structure(cfg, date(2026, 6, 30), "run-empty-window", {})

    assert result["status"] == "warning"
    assert result["empty_windows"] == 1
    assert result["context_updates"]["audit_findings"][0]["check"] == ("backfill_empty_windows")
