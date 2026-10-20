# -*- coding: utf-8 -*-
"""finalize_run 单测：幂等选择、终态过滤、单 run 收尾记录、整批幂等。"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_finalize():
    spec = importlib.util.spec_from_file_location("finalize_run", SCRIPTS / "finalize_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_run(tmp: Path, rid: str, *, age_s: float = 7200) -> Path:
    """合成一个 run 目录：patrol 可识别的最小 steps.log + run_summary.json，目录静默 age_s 秒。"""
    d = tmp / rid
    d.mkdir(parents=True)
    (d / "steps.log").write_text(
        "2026-10-19 01:00:00 | report_reproduce_judge | qid=RQ_a factor=f1 PASS [ic=0.05 icir=0.5]\n",
        encoding="utf-8")
    (d / "run_meta.json").write_text("{}", encoding="utf-8")
    (d / "run_summary.json").write_text(json.dumps({
        "outcome": "completed",
        "candidate_funnel": {"unique_train_evaluated": 5, "candidate_stored": 2,
                             "production_stored": 1},
    }, ensure_ascii=False), encoding="utf-8")
    old = time.time() - age_s
    os.utime(d / "steps.log", (old, old))
    return d


def test_select_runs_skips_finalized(tmp_path: Path):
    fr = _load_finalize()
    r1, r2 = _make_run(tmp_path, "r1"), _make_run(tmp_path, "r2")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"run_id": "r1"}) + "\n", encoding="utf-8")
    picked = fr.select_runs(tmp_path, ledger, statuses=None, idle_minutes=1)
    assert [p.name for p in picked] == ["r2"]


def test_select_runs_terminal_only_when_api_alive(tmp_path: Path):
    fr = _load_finalize()
    _make_run(tmp_path, "r1", age_s=0)      # API 说还在跑
    _make_run(tmp_path, "r2", age_s=0)      # API 说已终态
    statuses = {"r1": "running", "r2": "completed"}
    picked = fr.select_runs(tmp_path, tmp_path / "no_ledger.jsonl",
                            statuses=statuses, idle_minutes=60)
    assert [p.name for p in picked] == ["r2"]


def test_select_runs_offline_idle_guard(tmp_path: Path):
    fr = _load_finalize()
    _make_run(tmp_path, "fresh", age_s=60)   # 静默 1 分钟 < idle 30 → 保守跳过
    _make_run(tmp_path, "stale", age_s=7200)
    picked = fr.select_runs(tmp_path, tmp_path / "no_ledger.jsonl",
                            statuses=None, idle_minutes=30)
    assert [p.name for p in picked] == ["stale"]


def test_select_runs_explicit_only_bypasses_idle_guard(tmp_path: Path):
    """--runs 显式指定 = 跳过 API/空闲判定（monitor 刚结束就调用的场景）。"""
    fr = _load_finalize()
    _make_run(tmp_path, "fresh", age_s=0)    # 静默不足 → 扫描模式会跳过
    picked = fr.select_runs(tmp_path, tmp_path / "no_ledger.jsonl",
                            statuses=None, idle_minutes=30, only="fresh")
    assert [p.name for p in picked] == ["fresh"]


def test_select_runs_only_missing_dir_skipped(tmp_path: Path):
    """--runs 指向不存在目录 → 警告跳过，不炸整批。"""
    fr = _load_finalize()
    picked = fr.select_runs(tmp_path, tmp_path / "no_ledger.jsonl",
                            statuses=None, only="ghost")
    assert picked == []


def test_finalize_one_record_from_patrol(tmp_path: Path):
    fr = _load_finalize()
    d = _make_run(tmp_path, "r1")
    rec = fr.finalize_one(d)
    assert rec["run_id"] == "r1"
    assert rec["outcome"] == "completed"
    assert rec["cand"] == 2 and rec["prod"] == 1
    assert rec["judged"] == 1 and rec["pass"] == 1
    # bench 子集：真实跑 generate_scorecard 或失败降级为 bench_error，二者必有其一
    assert ("bench" in rec) != ("bench_error" in rec)


def test_run_finalize_end_to_end_idempotent(tmp_path: Path):
    fr = _load_finalize()
    for rid in ("r1", "r2"):
        _make_run(tmp_path, rid)
    ledger = tmp_path / "ledger.jsonl"
    out_md = tmp_path / "finalize_report.md"
    kw = dict(ui_root=tmp_path, ledger_path=ledger, backend="", idle_minutes=30,
              out_path=out_md, notify=False)
    recs1 = fr.run_finalize(**kw)
    assert {r["run_id"] for r in recs1} == {"r1", "r2"}
    assert len(ledger.read_text(encoding="utf-8").strip().splitlines()) == 2
    md1 = out_md.read_text(encoding="utf-8")
    assert "`r1`" in md1 and "`r2`" in md1
    # 第二遍：全部已在台账 → 零新收尾，台账不变（幂等）
    recs2 = fr.run_finalize(**kw)
    assert recs2 == []
    assert len(ledger.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_run_finalize_dry_run_writes_nothing(tmp_path: Path):
    fr = _load_finalize()
    _make_run(tmp_path, "r1")
    ledger = tmp_path / "ledger.jsonl"
    recs = fr.run_finalize(ui_root=tmp_path, ledger_path=ledger, backend="",
                           idle_minutes=30, notify=False, dry_run=True)
    assert recs == []
    assert not ledger.exists()
