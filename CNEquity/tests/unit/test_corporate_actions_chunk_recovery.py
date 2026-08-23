import json
from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.orchestrator.engine import JobEngine
from cnequity.orchestrator.manifest import Manifest
from cnequity.steps import events


def _rows(symbol: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "ex_date": [date(2024, 6, 28)],
            "action_type": ["cash_dividend"],
            "cash_dividend": [1.0],
            "bonus_ratio": [0.0],
            "transfer_ratio": [0.0],
            "allotment_ratio": [None],
            "allotment_price": [None],
            "source": ["tdx_protocol"],
        }
    )


def test_corporate_actions_stages_successful_chunks_for_retry(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "lake", batch_size=2)
    cfg._backfill = True
    cfg._backfill_start = date(2024, 1, 1)
    cfg._backfill_end = date(2024, 6, 28)
    monkeypatch.setattr(
        events, "load_symbols", lambda _cfg: ["600519.SH", "000001.SZ", "300750.SZ"]
    )
    calls = []

    def fake_fetch(_trade_date, *, symbols, **_kwargs):
        calls.append(symbols)
        if symbols == ["300750.SZ"]:
            raise RuntimeError("simulated chunk outage")
        return _rows(symbols[0])

    monkeypatch.setattr(events, "fetch_corporate_actions", fake_fetch)

    manifest = Manifest(cfg.manifest_path)
    run_id = manifest.start_run("init", {})
    manifest.start_batch(run_id, "parent", "corporate_actions", "corporate_actions")
    out = events.step_corporate_actions(cfg, date(2024, 6, 28), run_id, {"_batch_id": "parent"})

    assert calls == [["600519.SH", "000001.SZ"], ["300750.SZ"]]
    assert out["failed_symbols"] == ["300750.SZ"]
    assert out["status"] == "failed"
    assert out["rows_written"] == 1
    child = [
        row
        for row in manifest.get_batches_for_run(run_id)
        if row["task_id"] == "corporate_actions_chunk"
    ]
    assert len(child) == 1
    assert json.loads(child[0]["symbols_json"]) == ["600519.SH", "000001.SZ"]


def test_engine_retry_only_fetches_unreceipted_chunks(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "lake", batch_size=2)
    cfg._backfill = True
    cfg._backfill_start = date(2024, 1, 1)
    cfg._backfill_end = date(2024, 6, 28)
    monkeypatch.setattr(
        events,
        "load_symbols",
        lambda _cfg: ["600519.SH", "000001.SZ", "300750.SZ"],
    )
    calls = []
    fail_third = {True}

    def fake_fetch(_trade_date, *, symbols, **_kwargs):
        calls.append(list(symbols))
        if symbols == ["300750.SZ"] and fail_third:
            raise RuntimeError("simulated chunk outage")
        return _rows(symbols[0])

    monkeypatch.setattr(events, "fetch_corporate_actions", fake_fetch)

    engine = JobEngine(cfg)
    run_id = engine.manifest.start_run(
        "init",
        {"backfill": True, "phases": ["phase2a_corporate_actions"]},
    )
    first = engine._run_step("corporate_actions", date(2024, 6, 28), run_id, {})
    assert first["status"] == "failed"

    fail_third.clear()
    retry = engine._retry_run_locked(run_id, date(2024, 6, 28), auto_finalize=False)

    assert calls == [["600519.SH", "000001.SZ"], ["300750.SZ"], ["300750.SZ"]]
    assert retry["status"] == "success"
    parent_batches = [
        row
        for row in engine.manifest.get_batches_for_run(run_id)
        if row["task_id"] == "corporate_actions"
    ]
    assert sorted(row["status"] for row in parent_batches) == ["success", "superseded"]
