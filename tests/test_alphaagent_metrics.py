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
