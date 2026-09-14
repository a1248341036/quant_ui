"""测试 Run Scorecard 评分卡与基准指标统计正确性。"""
from __future__ import annotations

import json
from pathlib import Path
from alphaagent.factor.mining.run_metrics import (
    generate_scorecard,
    new_live_state,
    observe_event,
    build_metrics_snapshot,
)
from datetime import datetime


def test_live_metrics_snapshot():
    state = new_live_state(datetime.utcnow())

    # 1. 模拟 usage
    observe_event(state, "usage", {"input_tokens": 1000, "output_tokens": 200, "cache_input_tokens": 400})
    # 2. 模拟 thinking
    observe_event(state, "agent_thinking", {"content": "thinking some economic logic..."})
    # 3. 模拟 tool_results (含 prediction 与 ablation)
    observe_event(
        state,
        "tool_results",
        {
            "results": [
                {
                    "name": "evaluate_factor",
                    "elapsed_seconds": 1.2,
                    "result": {
                        "ok": True,
                        "prediction_check": {"verdict": "confirmed"},
                        "ablation_check": {"verdict": "conditioning_added_value"},
                    },
                },
                {
                    "name": "evaluate_factor",
                    "elapsed_seconds": 0.8,
                    "result": {
                        "ok": True,
                        "prediction_check": {"verdict": "contradicted"},
                        "near_miss_hint": "IC 达标 80%",
                    },
                },
            ]
        },
    )

    snap = build_metrics_snapshot(state)
    assert snap["llm_calls"] == 1
    assert snap["n_eval"] == 2
    assert snap["prediction_confirmed"] == 1
    assert snap["prediction_contradicted"] == 1
    assert snap["prediction_confirmed_ratio"] == 0.5
    assert snap["ablation_added_value"] == 1
    assert snap["near_miss_count"] == 1


def test_generate_scorecard_file(tmp_path: Path):
    run_id = "test_run_123"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 构造虚构 run_*.jsonl
    jsonl_file = run_dir / "run_20260914.jsonl"
    events = [
        {"ts": "2026-09-14T10:00:00Z", "event": "session_start"},
        {
            "ts": "2026-09-14T10:05:00Z",
            "event": "tool_results",
            "results": [
                {
                    "name": "evaluate_factor",
                    "elapsed_seconds": 2.5,
                    "arguments_raw": "RANK($close)",
                    "result": {
                        "ok": True,
                        "prediction_check": {"verdict": "confirmed"},
                    },
                },
                {
                    "name": "submit_factor",
                    "elapsed_seconds": 5.0,
                    "result": {"ok": True, "stored": True},
                },
            ],
        },
        {"ts": "2026-09-14T10:10:00Z", "event": "usage_total", "input_tokens": 50000, "output_tokens": 10000, "calls": 5},
    ]
    with jsonl_file.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    summary_mock = {
        "candidate_funnel": {
            "unique_train_evaluated": 1,
            "candidate_stored": 1,
            "production_stored": 1,
        }
    }

    sc = generate_scorecard(run_id, run_dir, summary_dict=summary_mock)
    assert sc["run_id"] == run_id
    assert sc["schema_version"] == 3
    assert sc["funnel"]["candidate_stored"] == 1
    assert sc["funnel"]["production_stored"] == 1
    assert sc["funnel"]["gate_survival_pct"] == 100.0
    assert sc["cognition"]["prediction_confirmed"] == 1

    # 验证 scorecard.json 物理存在且能正常读取
    sc_file = run_dir / "scorecard.json"
    assert sc_file.is_file()
    loaded = json.loads(sc_file.read_text(encoding="utf-8"))
    assert loaded["run_id"] == run_id
