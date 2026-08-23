"""L4 capital steps: fund flow, northbound, margin, dragon tiger, block trades."""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from cnequity.adapters.eastmoney.capital import (
    NORTHBOUND_HISTORY_START,
    NORTHBOUND_LAST_PUBLISHED,
    _quarter_end_dates,
    fetch_block_trades,
    fetch_dragon_tiger,
    fetch_fund_flow,
    fetch_margin_trading,
    fetch_northbound_flows_range,
    fetch_northbound_holdings,
)
from cnequity.config import Config
from cnequity.domain.datasets import should_fetch
from cnequity.orchestrator.registry import register_step
from cnequity.steps.common import BACKFILL_START, incremental_trade_dates, list_trading_dates
from cnequity.steps.http_common import run_incremental_fetched, write_fetched
from cnequity.storage.state import StateStore

logger = logging.getLogger(__name__)

_MARGIN_FLUSH_DAYS = 63  # stage a parquet part roughly every quarter of fetched days
_MIN_MARGIN_SYMBOLS_PER_DAY = 50
_NORTHBOUND_SZ_START = date(2016, 12, 5)
# Stock Connect is closed on HK-only holidays that this codebase has no
# calendar for (see step_northbound_flows); that mismatch alone is on the
# order of a few percent of mainland trading days a year. Set well above
# that so it doesn't mask a real fetch failure, but low enough that a
# largely-empty response still trips it.
_NORTHBOUND_GAP_TOLERANCE = 0.15
_MIN_NORTHBOUND_HOLDING_ROWS_PER_PERIOD = 100
_MIN_NORTHBOUND_HOLDING_ROWS_PER_CHANNEL = 50


def _run_capital_step(
    config: Config,
    trade_date: date,
    run_id: str,
    dataset: str,
    fetch_fn,
    *,
    allow_empty: bool = True,
) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError(f"{dataset}: eastmoney source disabled in config")

    # Bind Config so EastMoneyClient uses [sources.eastmoney] shared pacing /
    # proxy / timeout — bare clients only throttle at 1s in-process and trip EM
    # WAF on first-run multi-page clist/datacenter sweeps (fund_flow, margin).
    def _bound(d: date) -> pl.DataFrame:
        return fetch_fn(d, config=config)

    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        dataset,
        _bound,
        source="eastmoney",
        allow_empty=allow_empty,
    )


@register_step("fund_flow", group="capital", depends_on=["instruments"])
def step_fund_flow(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    return _run_capital_step(
        config, trade_date, run_id, "fund_flow", fetch_fund_flow, allow_empty=False
    )


@register_step("northbound_holdings", group="capital", depends_on=["instruments"])
def step_northbound_holdings(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("northbound_holdings: eastmoney source disabled in config")
    # Quarterly since Aug 2024: daily refreshes the latest quarter. Backfill
    # walks all quarter-ends from 2016 but the EM report only serves the most
    # recent quarter(s) — historical TRADE_DATE filters return 0 rows (verified
    # 2026-07), so history accrues forward only, one quarter per disclosure.
    from cnequity.steps.http_common import write_fetched

    backfill = getattr(config, "_backfill", False)
    df = _validate_northbound_holdings_snapshot(
        fetch_northbound_holdings(trade_date, backfill=backfill, config=config)
    )
    missing_periods: list[str] = []
    if backfill:
        expected = set(
            _quarter_end_dates(
                trade_date,
                start=getattr(config, "_backfill_start", None),
                end=getattr(config, "_backfill_end", None),
            )
        )
        observed = (
            {value.isoformat() for value in df.get_column("trade_date").drop_nulls().to_list()}
            if not df.is_empty() and "trade_date" in df.columns
            else set()
        )
        missing_periods = sorted(expected - observed)
    if df.is_empty():
        if not backfill:
            raise RuntimeError(
                f"northbound_holdings: no rows returned for {trade_date.isoformat()}"
            )
        result: dict = {"rows_read": 0, "rows_written": 0}
    else:
        result = write_fetched(config, run_id, "northbound_holdings", df, source="eastmoney")
    if missing_periods:
        result["status"] = "warning"
        result["missing_periods"] = len(missing_periods)
        result["context_updates"] = {
            "audit_findings": [
                {
                    "dataset": "northbound_holdings",
                    "severity": "warning",
                    "check": "backfill_missing_quarters",
                    "message": (
                        f"northbound holdings missing {len(missing_periods)} requested "
                        f"quarter(s): {', '.join(missing_periods[:8])}"
                    ),
                    "missing_periods": missing_periods,
                }
            ]
        }
    return result


@register_step("northbound_flows", group="capital")
def step_northbound_flows(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    """Northbound flows over the whole outstanding window in one request.

    Deliberately not on ``_run_capital_step``: that helper fetches one day at a
    time, and this dataset's watermark is frozen at the last session the
    exchanges published (see ``NORTHBOUND_LAST_PUBLISHED``). Per-day fetching
    would therefore issue one more request every day, forever, all of them
    returning nothing.
    """
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("northbound_flows: eastmoney source disabled in config")

    # Cadence check: source retired (cadence="skip") — skip the fetch entirely.
    state = StateStore(config.meta_root)
    wm = state.get_date("northbound_flows")
    if not should_fetch("northbound_flows", wm, trade_date):
        logger.info("northbound_flows: cadence skip (source retired)")
        state.set_date("northbound_flows", trade_date)
        return {"rows_read": 0, "rows_written": 0, "status": "success"}

    if getattr(config, "_backfill", False):
        start = getattr(config, "_backfill_start", None) or NORTHBOUND_HISTORY_START
        end = getattr(config, "_backfill_end", None) or trade_date
    else:
        dates = incremental_trade_dates(config, "northbound_flows", trade_date)
        if not dates:
            return {"rows_read": 0, "rows_written": 0}
        start, end = dates[0], dates[-1]

    # The report has both a fixed lower bound and a retirement date. Clamp the
    # request before touching the network: otherwise a retired feed is queried
    # in full on every daily run after its watermark freezes, and an explicit
    # pre-2014 backfill spends a request proving that the channel did not exist.
    start = max(start, NORTHBOUND_HISTORY_START)
    end = min(end, NORTHBOUND_LAST_PUBLISHED)
    if start > end:
        logger.info(
            "northbound_flows: requested window %s..%s is outside the published range "
            "%s..%s; skipping source request",
            start.isoformat(),
            end.isoformat(),
            NORTHBOUND_HISTORY_START.isoformat(),
            NORTHBOUND_LAST_PUBLISHED.isoformat(),
        )
        return {
            "rows_read": 0,
            "rows_written": 0,
            "note": "source window is outside northbound flow publication range",
        }

    df = fetch_northbound_flows_range(start, end, config=config)
    if df.is_empty():
        # Expected for any window past the cutoff — not a fetch failure, and
        # not something to zero-fill. The audit reports the frozen watermark.
        logger.info(
            "northbound_flows: no published rows in %s..%s", start.isoformat(), end.isoformat()
        )
        return {"rows_read": 0, "rows_written": 0}
    expected_dates = list_trading_dates(config, start, end)
    expected = {(day, "SH") for day in expected_dates if day >= NORTHBOUND_HISTORY_START}
    expected.update((day, "SZ") for day in expected_dates if day >= _NORTHBOUND_SZ_START)
    required_columns = {"trade_date", "channel"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise RuntimeError(
            "northbound_flows: response is missing required column(s): "
            + ", ".join(missing_columns)
        )
    observed = {
        (day, channel)
        for day, channel in df.select(["trade_date", "channel"]).iter_rows()
        if day is not None and channel is not None
    }
    missing = sorted(expected - observed)
    if missing:
        # `expected` is built from the mainland trading calendar only: there
        # is no Hong Kong / Stock Connect holiday calendar in this codebase.
        # Stock Connect trades only when both mainland exchanges and HKEX are
        # open, so a mainland trading day that is an HK-only holiday (Good
        # Friday, HKSAR Establishment Day, etc.) always looks "missing" here
        # even though the source published a genuinely complete range. That
        # mismatch is small and well known (a handful of HK-only holidays a
        # year); tolerate it, but still raise once the gap is far too large
        # for that explanation to plausibly cover, which is the signature of
        # a real fetch failure rather than a calendar mismatch.
        sample = ", ".join(f"{day.isoformat()}/{channel}" for day, channel in missing[:8])
        gap_ratio = len(missing) / len(expected)
        if gap_ratio > _NORTHBOUND_GAP_TOLERANCE:
            raise RuntimeError(
                "northbound_flows: incomplete published range; missing "
                f"{len(missing)} of {len(expected)} expected day/channel row(s) "
                f"({gap_ratio:.0%}, e.g. {sample})"
            )
        logger.warning(
            "northbound_flows: %s of %s expected day/channel row(s) absent from "
            "%s..%s (e.g. %s) - within HK-holiday tolerance, not raised",
            len(missing),
            len(expected),
            start.isoformat(),
            end.isoformat(),
            sample,
        )
    return write_fetched(config, run_id, "northbound_flows", df, source="eastmoney")


def _existing_margin_dates(
    config: Config,
    *,
    min_symbols: int = _MIN_MARGIN_SYMBOLS_PER_DAY,
) -> set[date]:
    root = config.curated_root / "margin_trading"
    files = list(root.glob("**/*.parquet")) if root.exists() else []
    if not files:
        return set()
    scan = pl.scan_parquet(files)
    if "symbol" not in scan.collect_schema().names():
        return set()
    return set(
        scan.group_by("trade_date")
        .agg(pl.col("symbol").n_unique().alias("_symbol_count"))
        .filter(pl.col("_symbol_count") >= min_symbols)
        .select("trade_date")
        .collect()
        .get_column("trade_date")
        .to_list()
    )


def _margin_symbol_count(df: pl.DataFrame) -> int:
    if "symbol" not in df.columns:
        return 0
    return df.get_column("symbol").drop_nulls().n_unique()


def _validate_northbound_holdings_snapshot(df: pl.DataFrame) -> pl.DataFrame:
    """Reject a non-empty but obviously truncated quarterly holdings response.

    The report contains two independent exchange legs.  A total row floor is
    not enough: a response containing only SH rows can still exceed the floor
    while silently losing the whole SZ leg.
    """
    if df.is_empty():
        return df
    required = {"symbol", "trade_date", "channel"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "northbound_holdings: response is missing required column(s): " + ", ".join(missing)
        )
    unique = df.unique(subset=["symbol", "trade_date", "channel"])
    counts = unique.group_by("trade_date").agg(pl.len().alias("_holding_rows"))
    channel_counts = unique.group_by("trade_date", "channel").agg(pl.len().alias("_channel_rows"))
    missing_channels: list[str] = []
    for row in counts.iter_rows(named=True):
        observed = set(
            unique.filter(pl.col("trade_date") == row["trade_date"])
            .get_column("channel")
            .drop_nulls()
            .to_list()
        )
        expected = {"SH"}
        if row["trade_date"] >= _NORTHBOUND_SZ_START:
            expected.add("SZ")
        missing = sorted(expected - observed)
        if missing:
            missing_channels.append(f"{row['trade_date']} missing {','.join(missing)}")

    incomplete_channels = channel_counts.filter(
        pl.col("_channel_rows") < _MIN_NORTHBOUND_HOLDING_ROWS_PER_CHANNEL
    )
    incomplete_totals = counts.filter(
        pl.col("_holding_rows") < _MIN_NORTHBOUND_HOLDING_ROWS_PER_PERIOD
    )
    if not incomplete_totals.is_empty() or not incomplete_channels.is_empty() or missing_channels:
        details = [
            f"{row['trade_date']} total={row['_holding_rows']}"
            for row in incomplete_totals.iter_rows(named=True)
        ]
        details.extend(
            f"{row['trade_date']} {row['channel']}={row['_channel_rows']}"
            for row in incomplete_channels.iter_rows(named=True)
        )
        details.extend(missing_channels)
        raise RuntimeError(
            "northbound_holdings: incomplete quarterly snapshot; each observed "
            f"period needs at least {_MIN_NORTHBOUND_HOLDING_ROWS_PER_PERIOD} "
            f"unique holding row(s), and each exchange channel needs at least "
            f"{_MIN_NORTHBOUND_HOLDING_ROWS_PER_CHANNEL} row(s) ({'; '.join(details)})"
        )
    return df


def _validate_margin_snapshot(df: pl.DataFrame, trade_date: date) -> pl.DataFrame:
    """Reject a non-empty margin response that is obviously truncated."""
    if df.is_empty():
        return df
    count = _margin_symbol_count(df)
    if count < _MIN_MARGIN_SYMBOLS_PER_DAY:
        raise RuntimeError(
            "margin_trading: incomplete daily snapshot; expected at least "
            f"{_MIN_MARGIN_SYMBOLS_PER_DAY} unique symbols on {trade_date.isoformat()}, "
            f"got {count}"
        )
    return df


def _backfill_margin_trading(config: Config, trade_date: date, run_id: str) -> dict:
    """Walk trading days fetching the EM margin report (history is served).

    Resumable: days already in curated are skipped, so a killed sweep can be
    rerun. ``--start/--end`` on ``cne backfill`` bound the walk; parts are
    staged in chunks so progress survives mid-run failures via compact.
    ``--workers N`` fetches days concurrently — each worker holds its own
    client throttled to 1 req/s (bypasses the shared source limiter, so the
    aggregate rate is up to N req/s; an explicit operator choice for sweeps).
    """
    from concurrent.futures import ThreadPoolExecutor

    from cnequity.adapters.eastmoney.em_auth import EastMoneyClient
    from cnequity.domain.schemas import with_provenance
    from cnequity.steps.common import _backfill_empty_day_finding
    from cnequity.storage import StagingWriter

    start = getattr(config, "_backfill_start", None) or BACKFILL_START
    end = getattr(config, "_backfill_end", None) or trade_date
    workers = max(1, int(getattr(config, "_backfill_workers", 1)))
    days = list_trading_dates(config, start, min(end, trade_date))
    have = _existing_margin_dates(config)
    todo = [d for d in days if d not in have]
    if not todo:
        return {"rows_read": 0, "rows_written": 0, "days_skipped": len(days)}

    writer = StagingWriter(config.staging_root)
    frames: list[pl.DataFrame] = []
    total_rows = 0
    empty_days: list[date] = []
    incomplete_days: list[tuple[date, int]] = []
    n_parts = 0

    def flush() -> None:
        nonlocal frames, total_rows, n_parts
        if not frames:
            return
        part = with_provenance(
            pl.concat(frames, how="diagonal_relaxed"), source="eastmoney", data_version="v1"
        )
        writer.write_batch("margin_trading", run_id, f"bf-{n_parts:04d}", part)
        n_parts += 1
        total_rows += part.height
        frames = []

    import threading

    local = threading.local()
    clients: list[EastMoneyClient] = []
    clients_lock = threading.Lock()

    def fetch_one(d: date) -> pl.DataFrame:
        client = getattr(local, "client", None)
        if client is None:
            # Prefer config so cross-process [sources.eastmoney] pacing applies
            # even with multiple workers (file lock serializes across threads).
            client = EastMoneyClient(config=config)
            local.client = client
            with clients_lock:
                clients.append(client)
        return fetch_margin_trading(d, client=client)

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Submit one flush-chunk at a time: a mid-sweep failure only waits
            # out the current chunk, and staged parts land as the sweep goes.
            for lo in range(0, len(todo), _MARGIN_FLUSH_DAYS):
                chunk = todo[lo : lo + _MARGIN_FLUSH_DAYS]
                for d, df in zip(chunk, pool.map(fetch_one, chunk), strict=True):
                    if df.is_empty():
                        empty_days.append(d)
                    else:
                        if "trade_date" not in df.columns:
                            raise RuntimeError(
                                f"margin_trading: fetch for {d.isoformat()} did not return "
                                "the configured trade_date column"
                            )
                        parsed_dates = df.get_column("trade_date").cast(pl.Date, strict=False)
                        invalid = parsed_dates.is_null() | (parsed_dates != d).fill_null(True)
                        invalid_count = int(invalid.sum())
                        if invalid_count:
                            raise RuntimeError(
                                f"margin_trading: fetch for {d.isoformat()} returned "
                                f"{invalid_count} row(s) with a different or invalid trade_date"
                            )
                        symbol_count = _margin_symbol_count(df)
                        if symbol_count < _MIN_MARGIN_SYMBOLS_PER_DAY:
                            incomplete_days.append((d, symbol_count))
                            continue
                        frames.append(df)
                done += len(chunk)
                flush()
                logger.info(
                    "margin_trading backfill: %d/%d days (at %s, %d rows staged)",
                    done,
                    len(todo),
                    chunk[-1].isoformat(),
                    total_rows,
                )
    except Exception:
        # Preserve successful rows from the current concurrent chunk when a
        # later fetch or response validation fails; the next run can then
        # resume instead of replaying the whole chunk.
        flush()
        raise
    finally:
        for client in clients:
            client.close()

    if empty_days:
        logger.warning(
            "margin_trading backfill: %d trading day(s) returned no rows (e.g. %s) — "
            "left absent; a rerun retries them",
            len(empty_days),
            empty_days[0].isoformat(),
        )
    result = {
        "rows_read": total_rows,
        "rows_written": total_rows,
        "days_fetched": len(todo) - len(empty_days) - len(incomplete_days),
        "days_skipped": len(days) - len(todo),
        "days_empty": len(empty_days),
    }
    if incomplete_days:
        result["status"] = "warning"
        result["failed_days"] = len(incomplete_days)
        result.setdefault("context_updates", {})["audit_findings"] = [
            {
                "dataset": "margin_trading",
                "severity": "warning",
                "check": "backfill_incomplete_days",
                "message": (
                    f"margin_trading: {len(incomplete_days)} day(s) returned fewer than "
                    f"{_MIN_MARGIN_SYMBOLS_PER_DAY} unique symbols; rows were not staged"
                ),
                "days": [
                    {"trade_date": day.isoformat(), "symbols": count}
                    for day, count in incomplete_days
                ],
            }
        ]
    if empty_days:
        result.setdefault("context_updates", {})["audit_findings"] = [
            *(result.get("context_updates", {}).get("audit_findings") or []),
            _backfill_empty_day_finding("margin_trading", empty_days),
        ]
    if empty_days or incomplete_days:
        result.setdefault("status", "warning")
    return result


@register_step("margin_trading", group="capital", depends_on=["instruments"])
def step_margin_trading(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if getattr(config, "_backfill", False):
        return _backfill_margin_trading(config, trade_date, run_id)

    def _fetch(day: date, *, config: Config) -> pl.DataFrame:
        return _validate_margin_snapshot(fetch_margin_trading(day, config=config), day)

    return _run_capital_step(
        config, trade_date, run_id, "margin_trading", _fetch, allow_empty=False
    )


def _backfill_daily_report(
    config: Config, trade_date: date, run_id: str, dataset: str, fetch_fn, floor: date
) -> dict:
    """dragon_tiger / block_trades: each day's fetch works standalone and the
    daily step never walked a range through it — see ``walk_day_backfill``."""
    from cnequity.adapters.eastmoney.em_auth import EastMoneyClient
    from cnequity.steps.common import walk_day_backfill

    client = EastMoneyClient(config=config)
    try:
        return walk_day_backfill(
            config,
            trade_date,
            run_id,
            dataset,
            lambda d: fetch_fn(d, client=client, config=config),
            source="eastmoney",
            floor=floor,
        )
    finally:
        client.close()


@register_step("dragon_tiger", group="signals", depends_on=["instruments"])
def step_dragon_tiger(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if getattr(config, "_backfill", False):
        # Confirmed live 2007-01-04 has rows, 2006-01-04 does not.
        return _backfill_daily_report(
            config, trade_date, run_id, "dragon_tiger", fetch_dragon_tiger, date(2007, 1, 1)
        )
    return _run_capital_step(config, trade_date, run_id, "dragon_tiger", fetch_dragon_tiger)


@register_step("block_trades", group="signals", depends_on=["instruments"])
def step_block_trades(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if getattr(config, "_backfill", False):
        # Confirmed live from 2010-01-04; older single-day probes were
        # ambiguous (block trades are sparse — a quiet day and "no report yet"
        # look identical), so this floor is the conservative, confirmed one.
        return _backfill_daily_report(
            config, trade_date, run_id, "block_trades", fetch_block_trades, date(2010, 1, 1)
        )
    return _run_capital_step(config, trade_date, run_id, "block_trades", fetch_block_trades)
