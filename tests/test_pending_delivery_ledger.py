# -*- coding: utf-8 -*-
"""B-05 修复测试：达标未交付台账查询口径。

- verdict ∈ {train_passed, validated} 且 error/failure_code 为空才入台账；
- eval_error / weak 等其他 verdict 与带错条目排除；
- metrics_json 解析 best-effort（坏 JSON 不炸）。
"""
from __future__ import annotations

import sqlite3

from scripts.pending_delivery_ledger import fetch_pending


def _mk_db(tmp_path):
    db = tmp_path / "mem.db"
    con = sqlite3.connect(db)
    con.execute(
        """CREATE TABLE memory_entries (
            id TEXT PRIMARY KEY, factor_name TEXT, expression TEXT, verdict TEXT,
            attempts INTEGER, created_at TEXT, metrics_json TEXT, family TEXT,
            last_run_id TEXT, error TEXT, failure_code TEXT)"""
    )
    con.executemany(
        "INSERT INTO memory_entries VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("e1", "f_a", "EXPR", "train_passed", 2, "2026-10-20T10:00:00",
             '{"ic": 0.06, "icir": 0.9}', "rev", None, "", None),
            ("e2", "f_b", "EXPR", "validated", 1, "2026-10-20T11:00:00",
             None, None, None, None, None),
            ("e3", "f_c", "EXPR", "train_passed", 1, "2026-10-20T12:00:00",
             "not-json", None, None, "", None),
            ("e4", "f_x", "EXPR", "train_passed", 1, "2026-10-20T13:00:00",
             None, None, None, "boom", None),
            ("e5", "f_y", "EXPR", "train_passed", 1, "2026-10-20T14:00:00",
             None, None, None, None, "eval_failed"),
            ("e6", "f_z", "EXPR", "weak", 1, "2026-10-20T15:00:00",
             None, None, None, None, None),
        ],
    )
    con.commit()
    con.close()
    return db


def test_fetch_pending_filters(tmp_path):
    rows = fetch_pending(_mk_db(tmp_path))
    names = [r["factor_name"] for r in rows]
    assert names == ["f_a", "f_b", "f_c"]  # e4/e5 带错排除，e6 verdict 不符排除
    assert rows[0]["metrics"] == {"ic": 0.06, "icir": 0.9}
    assert rows[1]["metrics"] == {}  # 无 metrics_json
    assert rows[2]["metrics"] == {}  # 坏 JSON best-effort
    assert rows[0]["verdict"] == "train_passed"
    assert rows[1]["verdict"] == "validated"
