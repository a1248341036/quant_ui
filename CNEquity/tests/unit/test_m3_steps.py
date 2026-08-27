from datetime import date, timedelta

import polars as pl
import pytest

import cnequity.steps  # noqa: F401
from cnequity.config import Config, ScheduleGroup, WaveConfig, validate_config
from cnequity.domain.schemas import validate_dataframe
from cnequity.orchestrator.registry import get_step


def test_m3_steps_are_registered():
    for name in (
        "fund_flow",
        "northbound_holdings",
        "northbound_flows",
        "margin_trading",
        "valuation_metrics",
        "sector_members",
        "announcement_index",
        "dragon_tiger",
        "block_trades",
    ):
        entry = get_step(name)
        assert entry.fn is not None


def test_fund_flow_schema_normalization():
    raw = pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [date(2024, 6, 28)],
            "main_net_inflow": [1_000_000.0],
            "super_large_net_inflow": [500_000.0],
            "large_net_inflow": [300_000.0],
            "medium_net_inflow": [100_000.0],
            "small_net_inflow": [100_000.0],
            "source": ["eastmoney"],
            "data_version": ["v1"],
            "fetched_at": ["2024-06-28T00:00:00+00:00"],
        }
    )
    out = validate_dataframe(raw, "fund_flow")
    assert out.height == 1


@pytest.fixture
def cfg(tmp_path):
    return Config(data_root=tmp_path / "data")


def test_step_fund_flow_writes_staging(cfg, monkeypatch):
    from cnequity.steps import capital as cap
    from cnequity.storage.state import StateStore

    StateStore(cfg.meta_root).set_date("fund_flow", date(2024, 6, 27))

    def fake_fetch(trade_date, **kwargs):
        return pl.DataFrame(
            {
                "symbol": ["600519.SH"],
                "trade_date": [trade_date],
                "main_net_inflow": [1.0],
                "super_large_net_inflow": [0.0],
                "large_net_inflow": [0.0],
                "medium_net_inflow": [0.0],
                "small_net_inflow": [0.0],
            }
        )

    monkeypatch.setattr(cap, "fetch_fund_flow", fake_fetch)
    cfg.staging_root.mkdir(parents=True)
    result = cap.step_fund_flow(cfg, date(2024, 6, 28), "run-1", {})
    assert result["rows_written"] == 1
    staged = list(cfg.staging_root.glob("fund_flow/**/*.parquet"))
    assert len(staged) == 1


@pytest.mark.parametrize(
    ("dataset", "step_name", "fetch_name"),
    [
        # margin_trading is deliberately absent: its T+1 publication cadence
        # makes an empty response legitimate (see the dedicated tests below).
        ("fund_flow", "step_fund_flow", "fetch_fund_flow"),
    ],
)
def test_capital_steps_reject_empty_canonical_feeds(
    cfg, monkeypatch, dataset, step_name, fetch_name
):
    from cnequity.steps import capital as cap
    from cnequity.storage.state import StateStore

    StateStore(cfg.meta_root).set_date(dataset, date(2024, 6, 27))
    monkeypatch.setattr(cap, fetch_name, lambda *_args, **_kwargs: pl.DataFrame())
    with pytest.raises(RuntimeError, match=f"{dataset}: no rows returned"):
        getattr(cap, step_name)(cfg, date(2024, 6, 28), "run-empty", {})


def test_margin_trading_skips_unpublished_day_without_fetch(cfg, monkeypatch):
    """T+1 publication: the walked tail day has no report yet. Skip it outright —
    clamping the request to the prior session stamps prior-day rows that
    incremental validation then rejects wholesale, failing every evening run
    (regression 2026-08-24/25)."""
    from cnequity.steps import capital as cap
    from cnequity.storage.state import StateStore

    StateStore(cfg.meta_root).set_date("margin_trading", date(2024, 6, 27))

    def fail_fetch(trade_date, **_kwargs):
        raise AssertionError(f"fetch must not be issued for unpublished {trade_date}")

    monkeypatch.setattr(cap, "fetch_margin_trading", fail_fetch)
    result = cap.step_margin_trading(cfg, date(2024, 6, 28), "run-unpublished", {})
    assert result["rows_written"] == 0


def test_margin_trading_writes_published_gap_days(cfg, monkeypatch):
    from cnequity.steps import capital as cap
    from cnequity.storage.state import StateStore

    StateStore(cfg.meta_root).set_date("margin_trading", date(2024, 6, 25))
    symbols = [f"{600000 + i}.SH" for i in range(60)]

    def fake_fetch(trade_date, **_kwargs):
        return pl.DataFrame(
            {
                "symbol": symbols,
                "trade_date": [trade_date] * len(symbols),
                "margin_balance": [1.0] * len(symbols),
                "margin_buy": [0.0] * len(symbols),
                "short_balance": [0.0] * len(symbols),
                "short_sell_volume": [0.0] * len(symbols),
            }
        )

    monkeypatch.setattr(cap, "fetch_margin_trading", fake_fetch)
    cfg.staging_root.mkdir(parents=True)
    result = cap.step_margin_trading(cfg, date(2024, 6, 28), "run-gap", {})
    # 06-26 and 06-27 are published (<= prior session); 06-28 is not.
    assert result["rows_written"] == 2 * len(symbols)
    staged = list(cfg.staging_root.glob("margin_trading/**/*.parquet"))
    assert len(staged) == 1


def test_margin_trading_rejects_partial_daily_feed(cfg, monkeypatch):
    from cnequity.steps import capital as cap
    from cnequity.storage.state import StateStore

    StateStore(cfg.meta_root).set_date("margin_trading", date(2024, 6, 25))

    def fake_fetch(trade_date, **_kwargs):
        return pl.DataFrame(
            {
                "symbol": ["600519.SH"],
                "trade_date": [trade_date],
                "margin_balance": [1.0],
                "margin_buy": [0.0],
                "short_balance": [0.0],
                "short_sell_volume": [0.0],
            }
        )

    monkeypatch.setattr(cap, "fetch_margin_trading", fake_fetch)
    # The first published gap day (06-26) comes back with far too few symbols:
    # a truncated snapshot must abort the step instead of landing.
    with pytest.raises(RuntimeError, match="margin_trading: incomplete daily snapshot"):
        cap.step_margin_trading(cfg, date(2024, 6, 28), "run-partial", {})


def test_northbound_holdings_rejects_empty_daily_feed(cfg, monkeypatch):
    from cnequity.steps import capital as cap

    monkeypatch.setattr(
        cap,
        "fetch_northbound_holdings",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="northbound_holdings: no rows returned"):
        cap.step_northbound_holdings(cfg, date(2024, 6, 28), "run-empty", {})


def test_northbound_holdings_backfill_surfaces_missing_quarters(cfg, monkeypatch):
    from cnequity.steps import capital as cap

    cfg._backfill = True
    cfg._backfill_start = date(2020, 1, 1)
    cfg._backfill_end = date(2020, 6, 30)
    monkeypatch.setattr(cap, "fetch_northbound_holdings", lambda *_args, **_kwargs: pl.DataFrame())

    result = cap.step_northbound_holdings(cfg, date(2026, 6, 30), "run-backfill", {})

    assert result["status"] == "warning"
    assert result["missing_periods"] == 2
    assert result["context_updates"]["audit_findings"][0]["check"] == ("backfill_missing_quarters")


def test_northbound_holdings_rejects_partial_period(cfg, monkeypatch):
    from cnequity.steps import capital as cap

    monkeypatch.setattr(
        cap,
        "fetch_northbound_holdings",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "symbol": ["600519.SH"],
                "trade_date": [date(2026, 6, 30)],
                "channel": ["SH"],
                "holding_shares": [1.0],
                "holding_mv": [2.0],
                "holding_ratio": [0.1],
            }
        ),
    )

    with pytest.raises(RuntimeError, match="northbound_holdings: incomplete quarterly snapshot"):
        cap.step_northbound_holdings(cfg, date(2026, 8, 16), "run-partial", {})


def test_northbound_holdings_rejects_missing_exchange_channel(cfg, monkeypatch):
    from cnequity.steps import capital as cap

    monkeypatch.setattr(
        cap,
        "fetch_northbound_holdings",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "symbol": [f"600{i:03d}.SH" for i in range(120)],
                "trade_date": [date(2026, 6, 30)] * 120,
                "channel": ["SH"] * 120,
                "holding_shares": [1.0] * 120,
                "holding_mv": [2.0] * 120,
                "holding_ratio": [0.1] * 120,
            }
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="northbound_holdings: incomplete quarterly snapshot.*missing SZ",
    ):
        cap.step_northbound_holdings(cfg, date(2026, 8, 16), "run-missing-channel", {})


def test_northbound_flows_skips_retired_window_without_source_request(cfg, monkeypatch):
    from cnequity.steps import capital as cap

    cfg._backfill = True
    cfg._backfill_start = date(2025, 1, 1)
    cfg._backfill_end = date(2026, 1, 1)

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("retired northbound window should not hit EastMoney")

    monkeypatch.setattr(cap, "fetch_northbound_flows_range", unexpected_fetch)

    result = cap.step_northbound_flows(cfg, date(2026, 8, 16), "run-retired", {})

    assert result["rows_written"] == 0
    assert "outside northbound flow publication range" in result["note"]


def test_northbound_flows_clips_backfill_to_published_range(cfg, monkeypatch):
    from cnequity.steps import capital as cap

    cfg._backfill = True
    cfg._backfill_start = date(2010, 1, 1)
    cfg._backfill_end = date(2026, 1, 1)
    seen = {}

    def fake_fetch(start, end, **kwargs):
        seen["window"] = (start, end)
        return pl.DataFrame()

    monkeypatch.setattr(cap, "fetch_northbound_flows_range", fake_fetch)

    result = cap.step_northbound_flows(cfg, date(2026, 8, 16), "run-clipped", {})

    assert result == {"rows_read": 0, "rows_written": 0}
    assert seen["window"] == (date(2014, 11, 17), date(2024, 8, 16))


def test_northbound_flows_rejects_partial_published_range(cfg, monkeypatch):
    from cnequity.steps import capital as cap

    requested = [date(2024, 6, 27), date(2024, 6, 28)]
    monkeypatch.setattr(cap, "incremental_trade_dates", lambda *args: requested)
    monkeypatch.setattr(cap, "list_trading_dates", lambda *args: requested)
    monkeypatch.setattr(
        cap,
        "fetch_northbound_flows_range",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "trade_date": [date(2024, 6, 28)],
                "channel": ["SH"],
                "net_buy": [1.0],
                "buy_amount": [2.0],
                "sell_amount": [1.0],
            }
        ),
    )

    with pytest.raises(RuntimeError, match="northbound_flows: incomplete published range"):
        cap.step_northbound_flows(cfg, date(2024, 6, 28), "run-partial", {})


def test_northbound_flows_tolerates_small_gap_from_untracked_hk_holidays(cfg, monkeypatch):
    """A handful of missing day/channel rows must not fail the whole fetch.

    There is no Hong Kong / Stock Connect holiday calendar in this codebase,
    so a mainland trading day that Stock Connect skips because HKEX (not
    SSE/SZSE) is closed always looks "missing" against the mainland-only
    expected set. That must stay a warning, not a hard failure, as long as
    the gap is small relative to the window.
    """
    from cnequity.steps import capital as cap

    requested = [date(2024, 6, 3) + timedelta(days=i) for i in range(20)]
    monkeypatch.setattr(cap, "incremental_trade_dates", lambda *args: requested)
    monkeypatch.setattr(cap, "list_trading_dates", lambda *args: requested)
    # 40 expected (day, channel) rows (20 days x SH+SZ); observed omits just
    # one HK-holiday-style day entirely, an ~5% gap, well under tolerance.
    hk_holiday = requested[7]
    rows = [(day, channel) for day in requested for channel in ("SH", "SZ") if day != hk_holiday]
    monkeypatch.setattr(
        cap,
        "fetch_northbound_flows_range",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "trade_date": [day for day, _ in rows],
                "channel": [channel for _, channel in rows],
                "net_buy": [1.0] * len(rows),
                "buy_amount": [2.0] * len(rows),
                "sell_amount": [1.0] * len(rows),
            }
        ),
    )

    result = cap.step_northbound_flows(cfg, requested[-1], "run-tolerant", {})
    assert result["rows_written"] == len(rows)


def test_trading_status_rejects_empty_feed(cfg, monkeypatch):
    from cnequity.steps import reference

    monkeypatch.setattr(reference, "load_symbols", lambda _cfg: ["600519.SH"])
    monkeypatch.setattr(
        reference,
        "fetch_trading_status",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="trading_status: no rows returned"):
        reference.step_trading_status(cfg, date(2024, 6, 28), "run-empty", {})


def test_trading_status_rejects_partial_feed(cfg, monkeypatch):
    from cnequity.steps import reference

    monkeypatch.setattr(reference, "load_symbols", lambda _cfg: ["600519.SH", "000001.SZ"])
    monkeypatch.setattr(
        reference,
        "fetch_trading_status",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "symbol": ["600519.SH"],
                "trade_date": [date(2024, 6, 28)],
                "is_trading": [True],
                "status": ["normal"],
            }
        ),
    )
    with pytest.raises(RuntimeError, match="incomplete daily snapshot"):
        reference.step_trading_status(cfg, date(2024, 6, 28), "run-partial", {})


def test_trading_status_preserves_incremental_findings(cfg, monkeypatch):
    from cnequity.steps import reference

    row = pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [date(2024, 6, 28)],
            "is_trading": [True],
            "status": ["N"],
        }
    )
    finding = {"dataset": "trading_status", "check": "coverage_gap"}
    monkeypatch.setattr(reference, "load_symbols", lambda _cfg: ["600519.SH"])
    monkeypatch.setattr(
        reference,
        "fetch_incremental_daily",
        lambda *args, **kwargs: (row, [finding]),
    )

    result = reference.step_trading_status(cfg, date(2024, 6, 28), "run-1", {})

    assert result["context_updates"]["audit_findings"] == [finding]


def test_valuation_snapshot_filters_to_bar_universe(cfg, monkeypatch):
    """The EastMoney clist returns delisted names with no bar; the daily snapshot
    must drop them so valuation stays in lock-step with daily_bars (audit:
    valuation_bars_orphan_symbol)."""
    from cnequity.steps import fundamentals as fund
    from cnequity.storage.state import StateStore

    # Bar universe: only 600519.SH has ever traded.
    bars_part = cfg.curated_root / "daily_bars" / "trade_date=2024-06-28"
    bars_part.mkdir(parents=True)
    pl.DataFrame({"symbol": ["600519.SH"], "trade_date": [date(2024, 6, 28)]}).write_parquet(
        bars_part / "part-merged.parquet"
    )

    StateStore(cfg.meta_root).set_date("valuation_metrics", date(2024, 6, 27))

    def fake_fetch(trade_date, **kwargs):
        return pl.DataFrame(
            {
                # 600519.SH trades; 000003.SZ is a delisted orphan the clist still returns.
                "symbol": ["600519.SH", "000003.SZ"],
                "trade_date": [trade_date, trade_date],
                "pe_ttm": [30.0, 1.0],
                "pb": [9.0, 0.1],
                "ps_ttm": [12.0, 0.5],
                "total_mv": [2.0e12, 1.0e8],
                "float_mv": [2.0e12, 1.0e8],
            }
        )

    monkeypatch.setattr(fund, "fetch_valuation_metrics", fake_fetch)
    cfg.staging_root.mkdir(parents=True)
    result = fund.step_valuation_metrics(cfg, date(2024, 6, 28), "run-1", {})

    assert result["rows_written"] == 1
    staged = pl.read_parquet(list(cfg.staging_root.glob("valuation_metrics/**/*.parquet")))
    assert staged["symbol"].to_list() == ["600519.SH"]


def test_validate_config_accepts_capital_group(tmp_path):
    cfg = Config(
        data_root=tmp_path / "data",
        # Explicit: the Config default is 8, which validate_config rejects on
        # macOS. This test is about the group shape, not the worker count.
        workers=1,
        daily_waves=[WaveConfig(name="core", parallel=True, steps=["instruments"])],
        schedule_groups={
            "capital": ScheduleGroup(at="16:30", steps=["fund_flow", "margin_trading"]),
        },
    )
    assert validate_config(cfg) == []


def test_northbound_reads_the_hsgt_report_not_an_index_fund_flow_kline():
    """Regression: northbound must not be sourced from a fund-flow kline.

    It used to read ``push2his /stock/fflow/kline/get?secid=1.000001`` and map
    f52/f53 onto SH/SZ. Those fields are 上证指数's 主力净流入 and 小单净流入 —
    two legs of a zero-sum decomposition, not two geographic channels — so the
    column carried plausible-looking numbers that were never northbound at all.
    """
    from cnequity.adapters.eastmoney import capital

    assert capital._NORTH_FLOW_REPORT == "RPT_MUTUAL_DEAL_HISTORY"
    assert capital._NORTHBOUND_CHANNELS == {"001": "SH", "003": "SZ"}
    assert not hasattr(capital, "_FFLOW_KLINE_URL")
    assert not hasattr(capital, "_KAMT_URL")


def test_backup_snapshot_failure_does_not_abort_the_ca_backfill(tmp_path, monkeypatch):
    """Regression: a best-effort audit artifact took down the primary fetch.

    `snapshot_corporate_actions_backup` writes an EastMoney snapshot for
    cross-source audit. When EastMoney changed its filter grammar it started
    raising, and the raise propagated out of the step — aborting
    `cne backfill corporate_actions` before TDX, the actual canonical source,
    was contacted at all.
    """
    from datetime import date

    import polars as pl

    from cnequity.steps import events

    monkeypatch.setattr(events, "load_symbols", lambda cfg: ["600519.SH"])

    def _boom(*a, **k):
        raise RuntimeError("EastMoney datacenter rejected schema")

    monkeypatch.setattr(events, "snapshot_corporate_actions_backup", _boom)
    fetched = {}

    def _fake_tdx(trade_date, **kwargs):
        fetched["called"] = True
        return pl.DataFrame(
            [
                {
                    "symbol": "600519.SH",
                    "ex_date": date(2024, 6, 28),
                    "action_type": "cash_dividend",
                    "cash_dividend": 1.0,
                    "bonus_ratio": 0.0,
                    "transfer_ratio": 0.0,
                    "allotment_ratio": None,
                    "allotment_price": None,
                }
            ]
        )

    monkeypatch.setattr(events, "fetch_corporate_actions", _fake_tdx)
    monkeypatch.setattr(events, "write_simple", lambda *a, **k: {"rows_read": 1, "rows_written": 1})

    cfg = Config(data_root=tmp_path / "lake")
    cfg._backfill = True
    out = events.step_corporate_actions(cfg, date(2024, 6, 28), "run-1", {})

    assert fetched.get("called") is True, "TDX must still be contacted"
    assert out["rows_written"] == 1
