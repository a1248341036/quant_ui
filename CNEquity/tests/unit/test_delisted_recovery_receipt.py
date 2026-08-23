import json
from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.steps import delisted


def test_delisted_recovery_gate_requires_receipt_integrity_and_bars(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    start = end = date(2024, 6, 27)
    targets = {
        "600001.SH": {
            "ownership": "dedicated_fetch",
            "basis": "formal_delist_date",
            "formal_delist_date": end.isoformat(),
        }
    }
    scope = delisted._recovery_scope(start, end, targets)
    receipt_root = cfg.meta_root / "quality" / "coverage" / delisted._RECOVERY_CLAIM
    receipt_root.mkdir(parents=True)
    receipt_path = receipt_root / f"{scope['scope_id']}.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "claim": delisted._RECOVERY_CLAIM,
                "status": "complete",
                "scope": {key: value for key, value in scope.items() if key != "targets"},
                "recovered_symbols": ["600001.SH"],
                "expected_no_data_symbols": [],
                "target_symbols_sha256": delisted._recovery_symbol_hash(["600001.SH"]),
            }
        ),
        encoding="utf-8",
    )

    assert delisted.delisted_recovery_covers(cfg, start, end, ["600001.SH"]) is False

    bars = cfg.curated_root / "daily_bars" / f"trade_date={start.isoformat()}"
    bars.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600001.SH"],
            "trade_date": [start],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [100],
            "amount": [1000.0],
        }
    ).write_parquet(bars / "part.parquet")

    assert delisted.delisted_recovery_covers(cfg, start, end, ["600001.SH"]) is True

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["scope"]["scope_id"] = "tampered"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    assert delisted.delisted_recovery_covers(cfg, start, end, ["600001.SH"]) is False


def test_delisted_recovery_covers_requires_bars_to_span_the_full_window(tmp_path):
    """A merely-overlapping on-disk span must not count as full coverage.

    A wide claimed window with bars on disk for only a narrow sub-range
    (e.g. after a later repair purges rows outside it) must return False,
    not True just because the two ranges touch at all.
    """
    cfg = Config(data_root=tmp_path / "data")
    start, end = date(2024, 1, 1), date(2024, 12, 31)
    targets = {
        "600001.SH": {
            "ownership": "dedicated_fetch",
            "basis": "formal_delist_date",
            "formal_delist_date": end.isoformat(),
        }
    }
    scope = delisted._recovery_scope(start, end, targets)
    receipt_root = cfg.meta_root / "quality" / "coverage" / delisted._RECOVERY_CLAIM
    receipt_root.mkdir(parents=True)
    (receipt_root / f"{scope['scope_id']}.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "claim": delisted._RECOVERY_CLAIM,
                "status": "complete",
                "scope": {key: value for key, value in scope.items() if key != "targets"},
                "recovered_symbols": ["600001.SH"],
                "expected_no_data_symbols": [],
                "target_symbols_sha256": delisted._recovery_symbol_hash(["600001.SH"]),
            }
        ),
        encoding="utf-8",
    )

    # Bars on disk cover only 2024-06-01..2024-06-05 - well inside [start,
    # end], overlapping it, but nowhere near spanning the full claimed year.
    for d in (date(2024, 6, 1), date(2024, 6, 5)):
        part = cfg.curated_root / "daily_bars" / f"trade_date={d.isoformat()}"
        part.mkdir(parents=True)
        pl.DataFrame(
            {
                "symbol": ["600001.SH"],
                "trade_date": [d],
                "open": [10.0],
                "high": [10.0],
                "low": [10.0],
                "close": [10.0],
                "volume": [100],
                "amount": [1000.0],
            }
        ).write_parquet(part / "part.parquet")

    assert delisted.delisted_recovery_covers(cfg, start, end, ["600001.SH"]) is False
