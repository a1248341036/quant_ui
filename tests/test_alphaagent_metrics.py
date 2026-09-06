"""alphaagent_metrics 的解析与汇总计算（合成轨迹，不依赖真实日志）。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.alphaagent_metrics import compute_run_metrics

T0, T1 = "2026-09-06T00:00:00", "2026-09-06T01:00:00"


def _make_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runabc123"
    d.mkdir()
    events = [
        {"ts": T0, "event": "session_start"},
        {"ts": T0, "event": "research_memory_retrieved"},
        {"ts": T0, "event": "agent_thinking", "content": "思考" * 100},  # 200 字符
        {"ts": T0, "event": "assistant_tool_call", "name": "evaluate_factor"},
        {"ts": T0, "event": "tool_results", "results": [
            {"name": "evaluate_factor", "elapsed_seconds": 60, "result": {"ok": True}},
            {"name": "submit_factor", "elapsed_seconds": 30,
             "result": {"ok": True, "candidate_stored": True, "stored": False}},
        ]},
        {"ts": T1, "event": "usage_total", "calls": 5, "input_tokens": 100_000,
         "output_tokens": 11_000, "cache_input_tokens": 20_000},
    ]
    (d / "run_20260906_000000.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8")
    steps = "\n".join([
        "2026-09-06 00:01:00 |INFO| turn=0 | submit.stage_one | f1 | passed=True | ms=1",
        "2026-09-06 00:02:00 |INFO| turn=0 | submit.stage_two | f1 | passed=False | ms=1",
        "2026-09-06 00:03:00 |INFO| turn=0 | submit.engine_gate | f2 | passed=False | fail=[excess_annual,tail_stability] | ms=1",
        "2026-09-06 00:04:00 |INFO| turn=0 | submit.promoted | f3 | passed=True | ms=1",
        "2026-09-06 00:05:00 |INFO| turn=0 | submit_result | f4 | skipped=blind_test_failed:blind_ic | error=BlindTestError",
    ])
    (d / "steps.log").write_text(steps, encoding="utf-8")
    return d


def test_compute_run_metrics(tmp_path) -> None:
    m = compute_run_metrics("runabc123", _make_run_dir(tmp_path))
    assert m["wall_minutes"] == 60.0
    assert m["llm_calls"] == 5
    assert m["input_k_tokens"] == 100.0
    assert m["output_k_tokens"] == 11.0
    assert m["cache_hit_rate"] == 0.2
    assert m["thinking_k_chars"] == 0.2
    assert m["tool_minutes"] == 1.5
    assert m["n_eval"] == 1
    assert m["n_submit"] == 1
    assert m["stored_candidate"] == 1 and m["stored_production"] == 0
    f = m["funnel"]
    assert f["stage_one_pass"] == 1 and f["stage_two_fail"] == 1
    assert f["gate_fail"] == 1 and f["promoted"] == 1 and f["blind_fail"] == 1
    assert m["gate_fail_reasons"] == {"excess_annual": 1, "tail_stability": 1}
    assert m["minutes_per_delivered"] == 60.0
    assert m["output_tokens_per_delivered_k"] == 11.0


def test_live_state_accumulator_matches_offline() -> None:
    """实时累计器与离线解析对同一事件流应给出一致口径。"""
    import warnings
    from datetime import datetime, timezone

    from alphaagent.factor.mining.run_metrics import (
        build_metrics_snapshot, new_live_state, observe_event)

    state = new_live_state(datetime.now(timezone.utc))
    stream = [
        ("agent_thinking", {"content": "ab" * 300}),                    # 600 字符
        ("usage", {"input_tokens": 10_000, "output_tokens": 2_000, "cache_input_tokens": 4_000}),
        ("tool_results", {"results": [
            {"name": "evaluate_factor", "elapsed_seconds": 30, "result": {"ok": True}},
            {"name": "eval_on_val_set", "elapsed_seconds": 60, "result": {"ok": True}},
            {"name": "submit_factor", "elapsed_seconds": 20,
             "result": {"ok": True, "candidate_stored": True, "stored": False,
                        "factor_name": "f1", "verdict": "candidate_approved"}},
        ]}),
    ]
    for ev, payload in stream:
        observe_event(state, ev, payload)
        if ev != "agent_thinking":
            with warnings.catch_warnings():
                pass
    snap = build_metrics_snapshot(state)
    assert snap["thinking_k_chars"] == 0.6
    assert snap["llm_calls"] == 1
    assert snap["input_k_tokens"] == 10.0 and snap["output_k_tokens"] == 2.0
    assert snap["n_eval"] == 1 and snap["n_eval_val"] == 1 and snap["n_submit"] == 1
    assert snap["stored_candidate"] == 1 and snap["stored_production"] == 0
    assert snap["tool_minutes"] == round(110 / 60, 1)
    assert snap["last_submit"]["factor"] == "f1"
    assert snap["wall_minutes"] >= 0
