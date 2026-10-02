# -*- coding: utf-8 -*-
"""bench 身份可区分 research_mode（2026-10-03）。

缺陷：API/监控启动的 run 没有 `bench_meta.json` → `time_meta.config_hash = None` →
`compare.py` 的 `config_matched` 失去意义，report run 会被拿去和 technical 基线对比并
输出误导性结论（实测 c51f08948c14 得到 REGRESSED）。
修法：无 bench_meta 时用 run 自身 `(research_mode, research_spec_hash)` 合成身份。
"""
from __future__ import annotations

import json

from alphaagent.factor.mining.bench.extended_metrics import _extract_time_meta


def _make_run(tmp_path, name: str, mode: str, spec_hash: str):
    d = tmp_path / name
    d.mkdir()
    (d / "run_meta.json").write_text(
        json.dumps({"params": {"research_mode": mode}}, ensure_ascii=False), encoding="utf-8")
    (d / "run_manifest.json").write_text(
        json.dumps({"model": "m", "research_spec_hash": spec_hash}, ensure_ascii=False),
        encoding="utf-8")
    return d


def test_report_and_technical_get_different_identity(tmp_path):
    rep = _extract_time_meta(_make_run(tmp_path, "a", "report", "S1"), [])
    tec = _extract_time_meta(_make_run(tmp_path, "b", "technical", "S2"), [])
    assert rep["research_mode"] == "report"
    assert tec["research_mode"] == "technical"
    assert rep["config_hash"] and tec["config_hash"]
    assert rep["config_hash"] != tec["config_hash"]


def test_same_mode_and_spec_hash_are_comparable(tmp_path):
    a = _extract_time_meta(_make_run(tmp_path, "a", "report", "S1"), [])
    b = _extract_time_meta(_make_run(tmp_path, "b", "report", "S1"), [])
    assert a["config_hash"] == b["config_hash"]


def test_same_mode_but_different_spec_is_not_comparable(tmp_path):
    a = _extract_time_meta(_make_run(tmp_path, "a", "report", "S1"), [])
    b = _extract_time_meta(_make_run(tmp_path, "b", "report", "S2"), [])
    assert a["config_hash"] != b["config_hash"]


def test_bench_meta_hash_wins_when_present(tmp_path):
    """bench 自己启动的 run 有 bench_meta.json → 沿用其 frozen hash（不覆盖）。"""
    d = _make_run(tmp_path, "a", "report", "S1")
    (d / "bench_meta.json").write_text(
        json.dumps({"config_hash": "d4da7818f564"}, ensure_ascii=False), encoding="utf-8")
    meta = _extract_time_meta(d, [])
    assert meta["config_hash"] == "d4da7818f564"
