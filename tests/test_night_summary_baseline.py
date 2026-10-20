# -*- coding: utf-8 -*-
"""night_summary 自动基线台账单测：同 since 幂等替换、跨夜基线滚动。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_nsr():
    spec = importlib.util.spec_from_file_location(
        "night_summary_report", SCRIPTS / "night_summary_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_nsr(tmp_path: Path):
    mod = _load_nsr()
    mod.LEDGER_PATH = tmp_path / "night_summary_ledger.jsonl"
    return mod


def test_append_ledger_same_since_replaces(tmp_path: Path):
    nsr = _make_nsr(tmp_path)
    nsr.append_ledger({"since": "A", "tot_eval": 5})
    nsr.append_ledger({"since": "A", "tot_eval": 7})  # 同一夜重跑 → 替换不追加
    rows = nsr._ledger_rows()
    assert len(rows) == 1 and rows[0]["tot_eval"] == 7


def test_load_prev_baseline_rolls_across_nights(tmp_path: Path):
    nsr = _make_nsr(tmp_path)
    nsr.append_ledger({"since": "A", "dual_gate_rate": 0.041, "tot_judge": 50, "tot_pass": 3})
    nsr.append_ledger({"since": "B", "dual_gate_rate": 0.060, "tot_judge": 60, "tot_pass": 6})
    prev = nsr.load_prev_baseline("B")
    assert prev["since"] == "A" and prev["dual_gate_rate"] == 0.041
    # 查 A 时返回最近一条不同 since 的记录（B）
    assert nsr.load_prev_baseline("A")["since"] == "B"


def test_load_prev_baseline_empty_ledger(tmp_path: Path):
    nsr = _make_nsr(tmp_path)
    assert nsr.load_prev_baseline("A") is None
