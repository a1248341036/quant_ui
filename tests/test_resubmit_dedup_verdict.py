# -*- coding: utf-8 -*-
"""回归：resubmit 重放脚本的同表达式去重必须兼容历史 verdict 字面量（OCR 2026-10-01 medium）。

历史库里同一表达式可能同时存在 `promising`（改名前的过线）与 `rejected` 两行；
去重应仍优先过线行，否则改名会让旧过线行被 rejected 顶掉，与"改名不改变历史口径"相悖。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import resubmit_promising_factors as R  # noqa: E402


def _make_db(path: Path, rows: list[tuple[str, str, str]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE memory_entries ("
        "factor_name TEXT, expression TEXT, verdict TEXT, metrics_json TEXT, "
        "conclusion TEXT, last_run_id TEXT, attempts INTEGER DEFAULT 1, "
        "updated_at TEXT DEFAULT '')"
    )
    for name, verdict, expr in rows:
        conn.execute(
            "INSERT INTO memory_entries (factor_name, expression, verdict, metrics_json, "
            "conclusion, last_run_id) VALUES (?,?,?,?,?,'r1')",
            (name, expr, verdict,
             json.dumps({"summary": {"ic": 0.03, "icir": 0.4}}), ""),
        )
    conn.commit()
    conn.close()


@pytest.mark.parametrize("legacy_name", ["promising", "train_passed"])
def test_dedup_prefers_passed_over_rejected_for_both_generations(tmp_path, monkeypatch, legacy_name):
    db = tmp_path / "m.db"
    expr = "RANK($close)"
    _make_db(db, [
        ("fac_old", legacy_name, expr),      # 过线行（可能是改名前的旧字面量）
        ("fac_rej", "rejected", expr),       # 同表达式的 rejected 行
    ])
    monkeypatch.setattr(R, "MEMORY_DB", db)

    rows = R._load_entries(None, rejected_patterns=None, min_ic=0.0)
    assert len(rows) == 1, f"同表达式应去重为一条: {rows}"
    assert rows[0]["factor_name"] == "fac_old", (
        f"{legacy_name} 行应优先于 rejected 行，实际取到 {rows[0]['factor_name']}"
    )
