"""ST 证据的每日续签（daily run 内）：前缀复用 + 增量补证据 + checkpoint 推进。

背景：``trading_status`` 的 ST 历史由 `cne backfill trading_status` 一次性建立；
若只靠那次回填，receipt 的 ``end`` 会停在当天，`cne audit` 每天都会报"没有凭证
覆盖这个窗口"。本步把 daily run 已入库的 `stock_st` 名单续进证据并重签凭证。
"""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

import cnequity.steps  # noqa: F401
from cnequity.config import Config
from cnequity.quality.st_coverage import build_st_scope, load_st_checkpoint, write_st_checkpoint
from cnequity.steps import reference

SYMBOLS = ["000732.SZ", "600519.SH"]


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_root=tmp_path / "data")
    c.staging_root.mkdir(parents=True, exist_ok=True)
    c.curated_root.mkdir(parents=True, exist_ok=True)
    c.trading_status_st_backfill_source = "tushare"
    return c


def _write_stock_st(cfg, rows: dict) -> None:
    part = cfg.curated_root / "stock_st" / "trade_date=2024-06"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(part / "part-merged.parquet")


def _stock_st_rows(days=("2024-06-27", "2024-06-28")) -> dict:
    return {
        "symbol": ["000732.SZ"] * len(days),
        "trade_date": [date.fromisoformat(d) for d in days],
        "name": ["ST泰禾"] * len(days),
        "type": ["ST"] * len(days),
        "type_name": ["风险警示板"] * len(days),
    }


def _seed_checkpoint(cfg, old_end: date, counts: dict) -> None:
    scope = build_st_scope(SYMBOLS, date(2016, 1, 1), old_end, universe="all_a", source="tushare")
    write_st_checkpoint(
        cfg,
        {
            "schema_version": 1,
            "claim": "historical_st_evidence",
            "scope": scope,
            "status": "complete",
            "completed_symbols": sorted(SYMBOLS),
            "evidence_rows_by_symbol": dict(counts),
            "unresolved_symbols": [],
            "evidence_covered_through": old_end.isoformat(),
            "completion_run_id": "seed-run",
        },
    )


# ─── 前缀复用 ─────────────────────────────────────────────────────────


def test_prefix_reuse_requires_opt_in(cfg, monkeypatch):
    monkeypatch.setattr(reference, "current_st_universe", lambda config: SYMBOLS)
    _seed_checkpoint(cfg, date(2024, 6, 27), {"000732.SZ": 5, "600519.SH": 0})

    new_scope = build_st_scope(SYMBOLS, date(2016, 1, 1), date(2024, 6, 28), universe="all_a", source="tushare")

    strict = load_st_checkpoint(cfg, new_scope)
    assert strict["completed_symbols"] == []  # 默认不认旧 end 的 checkpoint

    extended = load_st_checkpoint(cfg, new_scope, allow_end_extension=True)
    assert extended["completed_symbols"] == sorted(SYMBOLS)
    assert extended["evidence_covered_through"] == "2024-06-27"
    assert extended["evidence_rows_by_symbol"]["000732.SZ"] == 5


def test_prefix_reuse_rejects_newer_or_other_source(cfg):
    _seed_checkpoint(cfg, date(2024, 6, 27), {"000732.SZ": 5, "600519.SH": 0})

    # 旧 end 比请求窗口还晚 → 不是前缀，不能复用
    older_scope = build_st_scope(SYMBOLS, date(2016, 1, 1), date(2024, 6, 20), universe="all_a", source="tushare")
    assert load_st_checkpoint(cfg, older_scope, allow_end_extension=True)["completed_symbols"] == []

    # 来源不同 → 不能复用
    baostock_scope = build_st_scope(
        SYMBOLS, date(2016, 1, 1), date(2024, 6, 28), universe="all_a", source="baostock"
    )
    assert load_st_checkpoint(cfg, baostock_scope, allow_end_extension=True)["completed_symbols"] == []


# ─── 每日续签 ─────────────────────────────────────────────────────────


def test_daily_step_extends_evidence_and_checkpoint(cfg, monkeypatch):
    monkeypatch.setattr(reference, "current_st_universe", lambda config: SYMBOLS)
    _write_stock_st(cfg, _stock_st_rows())
    _seed_checkpoint(cfg, date(2024, 6, 27), {"000732.SZ": 5, "600519.SH": 0})

    result = reference.step_trading_status_st(cfg, date(2024, 6, 28), "run-daily", {})

    assert result["rows_read"] == 1  # 只补 06-28 这一天
    assert result["rows_written"] == 1
    assert result["coverage_pending_compact"] is True
    assert result["evidence_covered_through"] == "2024-06-28"
    assert result["extended_from"] == "2024-06-27"

    ckpt = json.loads(open(result["checkpoint"], encoding="utf-8").read())
    assert ckpt["scope"]["end"] == "2024-06-28"
    assert ckpt["evidence_covered_through"] == "2024-06-28"
    assert ckpt["status"] == "complete"
    assert ckpt["completion_run_id"] == "run-daily"
    # 累计计数：旧 5 + 新增 1；名单里没有的标的记 0（窗口内从未 ST）
    assert ckpt["evidence_rows_by_symbol"] == {"000732.SZ": 6, "600519.SH": 0}


def test_daily_step_is_idempotent(cfg, monkeypatch):
    monkeypatch.setattr(reference, "current_st_universe", lambda config: SYMBOLS)
    _write_stock_st(cfg, _stock_st_rows())
    _seed_checkpoint(cfg, date(2024, 6, 27), {"000732.SZ": 5, "600519.SH": 0})

    first = reference.step_trading_status_st(cfg, date(2024, 6, 28), "run-a", {})
    second = reference.step_trading_status_st(cfg, date(2024, 6, 28), "run-b", {})

    assert first["rows_written"] == 1
    assert second["rows_read"] == 0 and second["rows_written"] == 0
    ckpt = json.loads(open(second["checkpoint"], encoding="utf-8").read())
    assert ckpt["evidence_rows_by_symbol"] == {"000732.SZ": 6, "600519.SH": 0}
    assert ckpt["completion_run_id"] == "run-b"


def test_daily_step_skips_without_stock_st(cfg, monkeypatch):
    monkeypatch.setattr(reference, "current_st_universe", lambda config: SYMBOLS)
    out = reference.step_trading_status_st(cfg, date(2024, 6, 28), "run", {})
    assert out["status"] == "warning"
    assert "stock_st" in out["error"]


def test_daily_step_noop_for_baostock_source(cfg):
    cfg.trading_status_st_backfill_source = "baostock"
    out = reference.step_trading_status_st(cfg, date(2024, 6, 28), "run", {})
    assert out["rows_written"] == 0
    assert "baostock" in out["note"]
