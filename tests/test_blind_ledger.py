# -*- coding: utf-8 -*-
"""盲测消费台账单测：追加/加载、重复消费守卫的计数与警告。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_btf():
    spec = importlib.util.spec_from_file_location(
        "blind_test_factors", SCRIPTS / "blind_test_factors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ledger_append_and_load_roundtrip(tmp_path: Path):
    btf = _load_btf()
    ledger = tmp_path / "ledger.jsonl"
    assert btf._load_blind_ledger(ledger) == []
    btf._append_blind_ledger({"ts": "20261020_120000", "n_factors": 3,
                              "factor_ids": ["f1", "f2", "f3"]}, path=ledger)
    rows = btf._load_blind_ledger(ledger)
    assert len(rows) == 1 and rows[0]["n_factors"] == 3


def test_warn_duplicate_counts_overlap(tmp_path: Path):
    btf = _load_btf()
    ledger = tmp_path / "ledger.jsonl"
    btf._append_blind_ledger({"ts": "20261001_000000", "generated_at": "2026-10-01T00:00:00",
                              "factor_ids": ["f1", "f2"]}, path=ledger)
    btf._append_blind_ledger({"ts": "20261010_000000", "generated_at": "2026-10-10T00:00:00",
                              "factor_ids": ["f3"]}, path=ledger)
    # f1 消费过 → 重叠 1；f9 全新 → 0
    assert btf._warn_duplicate_consumption(["f1", "f9"], path=ledger) == 1
    assert btf._warn_duplicate_consumption(["f9", "f10"], path=ledger) == 0
    # 全部重叠
    assert btf._warn_duplicate_consumption(["f1", "f2", "f3"], path=ledger) == 3


def test_warn_duplicate_empty_ledger(tmp_path: Path):
    btf = _load_btf()
    assert btf._warn_duplicate_consumption(["f1"], path=tmp_path / "none.jsonl") == 0
