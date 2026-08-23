from datetime import date, datetime, timezone

import polars as pl

from cnequity.config import Config
from cnequity.derive.trading_status_history import (
    derive_suspension_history,
    status_evidence_rank,
)


def test_status_evidence_precedence_is_pit_aware():
    trade_day = date(2024, 6, 28)
    assert status_evidence_rank({"source": "baostock", "trade_date": trade_day}) == 0
    assert (
        status_evidence_rank(
            {
                "source": "eastmoney",
                "trade_date": trade_day,
                "fetched_at": datetime(2024, 6, 28, 7, 30, tzinfo=timezone.utc),
            }
        )
        == 0
    )
    assert (
        status_evidence_rank(
            {
                "source": "eastmoney",
                "trade_date": trade_day,
                "fetched_at": datetime(2024, 7, 1, 8, 0, tzinfo=timezone.utc),
            }
        )
        == 2
    )
    assert status_evidence_rank({"source": "derived_bar_gap", "trade_date": trade_day}) == 1


def _write(root, dataset, partition_col, val, df):
    d = root / "curated" / dataset / f"{partition_col}={val}"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "part-merged.parquet")


def test_derive_suspension_from_bar_gaps(tmp_path):
    root = tmp_path / "data"
    cfg = Config(data_root=root)
    # calendar: 3 trading days
    days = [date(2024, 6, 26), date(2024, 6, 27), date(2024, 6, 28)]
    for d in days:
        _write(
            root,
            "trading_calendar",
            "trade_date",
            d.isoformat(),
            pl.DataFrame(
                {
                    "trade_date": [d],
                    "is_trading": [True],
                    "source": ["seed"],
                    "data_version": ["v1"],
                    "fetched_at": ["2024-06-28T00:00:00+00:00"],
                }
            ),
        )

    # bars: 600519 trades all 3 days; 000001 missing the middle day (suspended)
    def bar(sym, d):
        return {
            "symbol": sym,
            "trade_date": d,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
            "amount": 1.0,
            "source": "tdx_protocol",
            "data_version": "v1",
            "fetched_at": "2024-06-28T00:00:00+00:00",
        }

    for d in days:
        rows = [bar("600519.SH", d)]
        if d != date(2024, 6, 27):
            rows.append(bar("000001.SZ", d))
        _write(root, "daily_bars", "trade_date", d.isoformat(), pl.DataFrame(rows))
    # instruments: both listed before window, not delisted
    (root / "curated" / "instruments").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "name": ["A"],
            "exchange": ["SH"],
            "asset_type": ["stock"],
            "list_date": [date(2010, 1, 1)],
            "delist_date": [None],
            "prev_symbol": [None],
            "source": ["tdx"],
            "data_version": ["v1"],
            "fetched_at": ["2024-06-28T00:00:00+00:00"],
        }
    ).write_parquet(root / "curated" / "instruments" / "part-merged.parquet")
    nested_instruments = root / "curated" / "instruments" / ".old-fragments"
    nested_instruments.mkdir()
    pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "name": ["B"],
            "exchange": ["SZ"],
            "asset_type": ["stock"],
            "list_date": [date(2010, 1, 1)],
            "delist_date": [None],
            "prev_symbol": [None],
            "source": ["tdx"],
            "data_version": ["v1"],
            "fetched_at": ["2024-06-28T00:00:00+00:00"],
        }
    ).write_parquet(nested_instruments / "part-old.parquet")

    existing = root / "curated" / "trading_status" / "trade_date=2024-06" / "fragments"
    existing.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [date(2024, 6, 26)],
            "is_trading": [True],
            "status": ["N"],
            "source": ["eastmoney"],
            "data_version": ["v1"],
            "fetched_at": ["2024-06-28T00:00:00+00:00"],
        }
    ).write_parquet(existing / "part-old.parquet")

    n = derive_suspension_history(cfg)
    assert n == 1  # only 000001 on 2024-06-27

    # trading_status is month-partitioned (DatasetSpec); never write day dirs.
    month_dir = root / "curated" / "trading_status" / "trade_date=2024-06"
    assert month_dir.is_dir()
    assert not (root / "curated" / "trading_status" / "trade_date=2024-06-27").exists()

    ts = pl.read_parquet(month_dir / "part-merged.parquet")
    assert (
        ts.filter(
            (pl.col("symbol") == "600519.SH") & (pl.col("trade_date") == date(2024, 6, 26))
        ).height
        == 1
    )
    assert [path.name for path in month_dir.rglob("*.parquet")] == ["part-merged.parquet"]
    susp = ts.filter(pl.col("status") == "suspended")
    assert susp.height == 1
    assert susp["symbol"][0] == "000001.SZ"
    assert susp["trade_date"][0] == date(2024, 6, 27)
    assert susp["is_trading"][0] is False
    assert susp["source"][0] == "derived_bar_gap"


def test_derive_suspension_treats_zero_volume_placeholder_as_missing(tmp_path):
    root = tmp_path / "data"
    cfg = Config(data_root=root)
    days = [date(2024, 6, 27), date(2024, 6, 28)]
    for d in days:
        _write(
            root,
            "trading_calendar",
            "trade_date",
            d.isoformat(),
            pl.DataFrame(
                {
                    "trade_date": [d],
                    "is_trading": [True],
                    "source": ["seed"],
                    "data_version": ["v1"],
                    "fetched_at": ["2024-06-28T00:00:00+00:00"],
                }
            ),
        )

    def bar(day: date, volume: int) -> dict:
        return {
            "symbol": "000001.SZ",
            "trade_date": day,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": volume,
            "amount": float(volume) * 10.0,
            "source": "tdx_protocol",
            "data_version": "v1",
            "fetched_at": "2024-06-28T00:00:00+00:00",
        }

    _write(
        root,
        "daily_bars",
        "trade_date",
        "2024-06-27",
        pl.DataFrame([bar(days[0], 0)]),
    )
    _write(
        root,
        "daily_bars",
        "trade_date",
        "2024-06-28",
        pl.DataFrame([bar(days[1], 100)]),
    )
    (root / "curated" / "instruments").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "name": ["B"],
            "exchange": ["SZ"],
            "asset_type": ["stock"],
            "list_date": [date(2010, 1, 1)],
            "delist_date": [None],
            "prev_symbol": [None],
            "source": ["tdx"],
            "data_version": ["v1"],
            "fetched_at": ["2024-06-28T00:00:00+00:00"],
        }
    ).write_parquet(root / "curated" / "instruments" / "part-merged.parquet")

    assert derive_suspension_history(cfg) == 1
    status = pl.read_parquet(
        root / "curated" / "trading_status" / "trade_date=2024-06" / "part-merged.parquet"
    )
    assert status.filter(pl.col("status") == "suspended")["trade_date"].to_list() == [
        date(2024, 6, 27)
    ]


def test_derive_suspension_ignores_placeholder_only_symbol(tmp_path):
    root = tmp_path / "data"
    cfg = Config(data_root=root)
    days = [date(2024, 6, 27), date(2024, 6, 28)]
    for d in days:
        _write(
            root,
            "trading_calendar",
            "trade_date",
            d.isoformat(),
            pl.DataFrame(
                {
                    "trade_date": [d],
                    "is_trading": [True],
                    "source": ["seed"],
                    "data_version": ["v1"],
                    "fetched_at": ["2024-06-28T00:00:00+00:00"],
                }
            ),
        )

    rows = []
    for symbol, volume in (("600519.SH", 100), ("000001.SZ", 0)):
        for d in days:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": d,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": volume,
                    "amount": float(volume),
                    "source": "tdx_protocol",
                    "data_version": "v1",
                    "fetched_at": "2024-06-28T00:00:00+00:00",
                }
            )
    _write(
        root,
        "daily_bars",
        "trade_date",
        "2024-06",
        pl.DataFrame(rows),
    )

    (root / "curated" / "instruments").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "list_date": [date(2010, 1, 1), date(2010, 1, 1)],
            "delist_date": [None, None],
        }
    ).write_parquet(root / "curated" / "instruments" / "part-merged.parquet")

    assert derive_suspension_history(cfg) == 0


def test_derive_suspension_empty_lake(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    assert derive_suspension_history(cfg) == 0


def test_derive_suspension_respects_end_window(tmp_path):
    """Gaps outside [--start,--end] must not be written."""
    root = tmp_path / "data"
    cfg = Config(data_root=root)
    days = [
        date(2015, 12, 30),
        date(2015, 12, 31),
        date(2016, 1, 4),
        date(2016, 1, 5),
    ]

    def bar(sym, d):
        return {
            "symbol": sym,
            "trade_date": d,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
            "amount": 1.0,
            "source": "tdx_protocol",
            "data_version": "v1",
            "fetched_at": "2016-01-05T00:00:00+00:00",
        }

    for d in days:
        _write(
            root,
            "trading_calendar",
            "trade_date",
            d.isoformat(),
            pl.DataFrame(
                {
                    "trade_date": [d],
                    "is_trading": [True],
                    "source": ["seed"],
                    "data_version": ["v1"],
                    "fetched_at": ["2016-01-05T00:00:00+00:00"],
                }
            ),
        )
        # 000001 missing 2015-12-31 and 2016-01-04
        rows = [bar("600519.SH", d)]
        if d not in (date(2015, 12, 31), date(2016, 1, 4)):
            rows.append(bar("000001.SZ", d))
        _write(root, "daily_bars", "trade_date", d.isoformat(), pl.DataFrame(rows))

    (root / "curated" / "instruments").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "name": ["A", "B"],
            "exchange": ["SH", "SZ"],
            "asset_type": ["stock", "stock"],
            "list_date": [date(2010, 1, 1), date(2010, 1, 1)],
            "delist_date": [None, None],
            "prev_symbol": [None, None],
            "source": ["tdx", "tdx"],
            "data_version": ["v1", "v1"],
            "fetched_at": ["2016-01-05T00:00:00+00:00"] * 2,
        }
    ).write_parquet(root / "curated" / "instruments" / "part-merged.parquet")

    n = derive_suspension_history(cfg, start=date(2015, 1, 1), end=date(2015, 12, 31))
    assert n == 1
    ts = pl.read_parquet(
        root / "curated" / "trading_status" / "trade_date=2015-12" / "part-merged.parquet"
    )
    assert ts.filter(pl.col("status") == "suspended")["trade_date"].to_list() == [
        date(2015, 12, 31)
    ]
    assert not (root / "curated" / "trading_status" / "trade_date=2016-01").exists()
