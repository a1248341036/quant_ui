"""bench trend 跳过进行中 run 的回归（2026-10-01 实测的读数陷阱）。

进行中的 run 只有部分 scorecard（甚至完全没有），trend 的"即时重算"兜底会把
未完成值算进趋势末行 → 读表误判为回归（实测 run 跑 5 分钟即出现 family_coverage=2 的假[-]）。
完成态判据：agentscope_run 收尾时写 `<log>.messages.json.gz`。
"""
from __future__ import annotations

import json

import alphaagent.factor.mining.bench.trend as trend


def _mk_run(root, name: str, value: float, *, finished: bool) -> None:
    d = root / name
    d.mkdir()
    (d / "scorecard.json").write_text(
        json.dumps({
            "run_id": name,
            "time_meta": {"created_at": f"2026-10-01T0{int(value)}:00:00", "config_hash": "cfg"},
            "v": value,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    if finished:
        (d / "run_20261001_000000.messages.json.gz").write_bytes(b"")


def test_inflight_run_skipped(tmp_path, monkeypatch):
    _mk_run(tmp_path, "run_finished", 1.0, finished=True)
    _mk_run(tmp_path, "run_inflight", 2.0, finished=False)

    monkeypatch.setattr(trend, "UI_ROOT", tmp_path)
    monkeypatch.setattr(trend, "_extract_metric", lambda sc, k: sc.get("v"))

    ids = [r["run_id"] for r in trend.get_metric_trend("effective_novelty_rate", last=10)]
    assert ids == ["run_finished"]


def test_finished_runs_ordered_by_time(tmp_path, monkeypatch):
    _mk_run(tmp_path, "a_first", 1.0, finished=True)
    _mk_run(tmp_path, "b_second", 2.0, finished=True)

    monkeypatch.setattr(trend, "UI_ROOT", tmp_path)
    monkeypatch.setattr(trend, "_extract_metric", lambda sc, k: sc.get("v"))

    items = trend.get_metric_trend("effective_novelty_rate", last=10)
    assert [r["run_id"] for r in items] == ["a_first", "b_second"]
    assert items[1]["delta"] == 1.0
    assert items[1]["verdict"] == "[+]"
