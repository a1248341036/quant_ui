# -*- coding: utf-8 -*-
"""AlphaAgent 挖掘评测 Bench 单元测试。

覆盖：
1. 冻结配置（load/save/set/hash）
2. 扩展指标计算（A-F 分组、时间元数据、头条汇总）
3. 基线与对比判定（可判定分栏、容差带、IMPROVED/REGRESSED/NO_CLEAR_CHANGE/INVALID）
4. 台账记录与趋势追踪
5. 一页报告渲染
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alphaagent.factor.mining.bench.baseline import load_baseline, set_baseline
from alphaagent.factor.mining.bench.compare import compare_scorecards, format_one_page_report
from alphaagent.factor.mining.bench.config import (
    DEFAULT_BENCH_CONFIG,
    compute_config_hash,
    load_bench_config,
    save_bench_config,
    set_config_value,
)
from alphaagent.factor.mining.bench.extended_metrics import compute_extended_metrics
from alphaagent.factor.mining.bench.ledger import load_ledger, record_eval
from alphaagent.factor.mining.bench.trend import format_trend_table, get_metric_trend


def test_bench_config_load_and_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_config_file = tmp_path / "config.json"
    monkeypatch.setattr("alphaagent.factor.mining.bench.config.CONFIG_FILE", fake_config_file)
    monkeypatch.setattr("alphaagent.factor.mining.bench.config.BENCH_DIR", tmp_path)

    # 首次加载自动初始化
    cfg = load_bench_config()
    assert cfg["panel"] == "cne://"
    assert cfg["max_turns"] == 5

    # 计算哈希
    h1 = compute_config_hash(cfg)
    assert len(h1) == 12

    # 修改配置
    set_config_value("max_turns", "10")
    cfg2 = load_bench_config()
    assert cfg2["max_turns"] == 10
    h2 = compute_config_hash(cfg2)
    assert h1 != h2


def test_extended_metrics_synthetic(tmp_path: Path):
    run_id = "test_synth_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 写入 run_meta.json
    (run_dir / "run_meta.json").write_text(json.dumps({
        "run_id": run_id,
        "created_at": "2026-09-25T10:00:00+00:00",
        "title": "测试 run",
    }, ensure_ascii=False), encoding="utf-8")

    # 构造事件
    events = [
        {"ts": "2026-09-25T10:00:00+00:00", "event": "session_start", "turn": 0},
        {"ts": "2026-09-25T10:01:00+00:00", "event": "usage", "input_tokens": 1000, "output_tokens": 200, "turn": 0},
        {
            "ts": "2026-09-25T10:02:00+00:00",
            "event": "tool_results",
            "turn": 0,
            "results": [
                {
                    "name": "evaluate_factor",
                    "elapsed_seconds": 2.5,
                    "arguments_raw": json.dumps({
                        "factor_name": "f1",
                        "multi_line_expr": "a = TS_MEAN($close, 10)\nCS_ZSCORE(a)",
                        "prediction": {"expected_shape": "monotonic_increasing"},
                        "interaction": {"interaction_type": "gated_signal"},
                    }),
                    "result": {
                        "ok": True,
                        "summary": {"ic": 0.025, "rank_ic": 0.03, "factor_coverage": 0.95},
                        "screen_rules": [
                            {"metric": "ic", "op": "abs_gte", "expected": 0.02, "actual": 0.025, "passed": True},
                            {"metric": "coverage", "op": "gte", "expected": 0.85, "actual": 0.95, "passed": True},
                        ],
                        "prediction_check": {"verdict": "confirmed"},
                    },
                },
                {
                    "name": "evaluate_factor",
                    "elapsed_seconds": 1.5,
                    "arguments_raw": json.dumps({
                        "factor_name": "f2",
                        "multi_line_expr": "b = TS_STD($volume, 20)\nCS_ZSCORE(b)",
                        "prediction": {"expected_shape": "monotonic_increasing"},
                    }),
                    "result": {
                        "ok": True,
                        "summary": {"ic": 0.015, "rank_ic": 0.018, "factor_coverage": 0.92},
                        "screen_rules": [
                            {"metric": "ic", "op": "abs_gte", "expected": 0.02, "actual": 0.015, "passed": False},
                        ],
                        "prediction_check": {"verdict": "contradicted"},
                    },
                },
            ],
        },
        {"ts": "2026-09-25T10:05:00+00:00", "event": "usage", "input_tokens": 1500, "output_tokens": 300, "turn": 1},
        {
            "ts": "2026-09-25T10:06:00+00:00",
            "event": "tool_results",
            "turn": 1,
            "results": [
                {
                    "name": "submit_factor",
                    "arguments_raw": json.dumps({"factor_name": "f1", "multi_line_expr": "a = TS_MEAN($close, 10)\nCS_ZSCORE(a)"}),
                    "result": {"ok": True, "candidate_stored": True, "stored": False, "production_similarity": {"max_abs_corr": 0.28}},
                }
            ],
        },
        {"ts": "2026-09-25T10:10:00+00:00", "event": "session_end", "turn": 1},
    ]

    base_metrics = {
        "run_id": run_id,
        "wall_minutes": 10.0,
        "input_k_tokens": 2.5,
        "output_k_tokens": 0.5,
        "stored_candidate": 1,
        "stored_production": 0,
        "funnel": {"stage_one_yield_pct": 50.0, "gate_survival_pct": 0.0},
    }

    ext = compute_extended_metrics(run_id, run_dir, base_metrics=base_metrics, events=events)

    assert ext["schema_version"] == 4
    assert ext["time_meta"]["created_at"] == "2026-09-25T10:00:00+00:00"
    assert ext["time_meta"]["wall_minutes"] == 10.0

    # 探索组 A
    exp = ext["exploration"]
    assert exp["effective_novelty_rate"] == 1.0  # 2 个尝试均不重复
    assert exp["repeat_attempt_rate"] == 0.0
    assert exp["facet_coverage"] >= 1

    # 过程组 E
    proc = ext["process"]
    assert proc["prediction_coverage"] == 1.0  # 两个 evaluate 都有 prediction
    assert proc["near_miss_progress"]["min_gap"] is not None

    # 成本组 D
    cost = ext["cost"]
    assert cost["cost_per_candidate_k_tokens"] == 3.0  # (2.5 + 0.5) / 1
    assert cost["first_pass_turn"] == 0

    # 轨迹组 B
    dyn = ext["dynamics"]
    assert len(dyn["best_so_far_curve"]) >= 2
    assert dyn["best_so_far_curve"][-1]["max_abs_ic"] == 0.025

    # 诚实组 F
    integ = ext["integrity"]
    assert integ["total_passing_unique_count"] == 1
    assert integ["unsubmitted_passing_rate"] == 0.0  # f1 被提交了，无遗漏


def test_compare_scorecards_improved_and_regressed():
    base_sc = {
        "run_id": "run_base",
        "time_meta": {"config_hash": "cfg_same"},
        "headline": {
            "stage_one_yield_pct": 10.0,
            "effective_novelty_rate": 0.70,
            "unsubmitted_passing_rate": 0.20,
        },
        "exploration": {"facet_coverage": 4},
        "process": {"prediction_coverage": 80.0, "tool_error_rate": 0.25},
        "cost": {"cost_per_candidate_k_tokens": 500.0},
        "funnel": {"candidate_stored": 1, "production_stored": 0},
    }

    # 改善情况：海选过线率升、新颖率升、未提交率降、成本降
    curr_improved = {
        "run_id": "run_imp",
        "time_meta": {"config_hash": "cfg_same"},
        "headline": {
            "stage_one_yield_pct": 15.0,  # +5.0pp (好)
            "effective_novelty_rate": 0.85,  # +0.15 (好)
            "unsubmitted_passing_rate": 0.05,  # -0.15 (好)
        },
        "exploration": {"facet_coverage": 6},  # +2 (好)
        "process": {"prediction_coverage": 95.0, "tool_error_rate": 0.10},  # 好
        "cost": {"cost_per_candidate_k_tokens": 300.0},  # -200 (好)
        "funnel": {"candidate_stored": 2, "production_stored": 1},
    }

    diff_imp = compare_scorecards(curr_improved, base_sc)
    assert diff_imp["overall"] == "IMPROVED"
    assert diff_imp["improved_count"] >= 4
    assert diff_imp["regressed_count"] == 0

    # 验证报告文本格式化
    rep = format_one_page_report(curr_improved, base_sc, diff_imp)
    assert "总评: IMPROVED" in rep
    assert "[一致]" in rep

    # 回归情况：过线率大降、错误率升高
    curr_regressed = {
        "run_id": "run_reg",
        "time_meta": {"config_hash": "cfg_same"},
        "headline": {
            "stage_one_yield_pct": 2.0,  # -8pp (差)
            "effective_novelty_rate": 0.50,  # -0.20 (差)
            "unsubmitted_passing_rate": 0.50,  # +0.30 (差)
        },
        "exploration": {"facet_coverage": 2},
        "process": {"prediction_coverage": 60.0, "tool_error_rate": 0.40},
        "cost": {"cost_per_candidate_k_tokens": 800.0},
        "funnel": {"candidate_stored": 0, "production_stored": 0},
    }

    diff_reg = compare_scorecards(curr_regressed, base_sc)
    assert diff_reg["overall"] == "REGRESSED"
    assert diff_reg["regressed_count"] >= 4


def test_compare_config_mismatch():
    base_sc = {
        "run_id": "run_base",
        "time_meta": {"config_hash": "cfg_1111"},
        "headline": {"stage_one_yield_pct": 10.0},
    }
    curr_sc = {
        "run_id": "run_curr",
        "time_meta": {"config_hash": "cfg_2222"},
        "headline": {"stage_one_yield_pct": 20.0},
    }
    diff = compare_scorecards(curr_sc, base_sc)
    assert diff["overall"] == "INVALID_CONFIG_MISMATCH"
    assert diff["config_matched"] is False


def test_ledger_record_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_bench_dir = tmp_path / "bench"
    monkeypatch.setattr("alphaagent.factor.mining.bench.ledger.BENCH_DIR", fake_bench_dir)
    monkeypatch.setattr("alphaagent.factor.mining.bench.ledger.RECORDS_DIR", fake_bench_dir / "eval_records")
    monkeypatch.setattr("alphaagent.factor.mining.bench.ledger.LEDGER_CSV", fake_bench_dir / "ledger.csv")

    sc = {
        "run_id": "test_run_led",
        "time_meta": {"created_at": "2026-09-25T12:00:00+00:00", "wall_minutes": 25.0, "config_hash": "abc123"},
        "headline": {"stage_one_yield_pct": 8.5, "effective_novelty_rate": 0.88},
        "summary": {"total_tokens": 50000},
    }

    diff_res = {"overall": "IMPROVED", "improved_count": 3, "regressed_count": 0}
    rec_path = record_eval(sc, diff_res=diff_res, note="测试记账", branch="main", commit="12345678", dirty=False, exit_code=0)

    assert rec_path.is_file()
    ledger_rows = load_ledger()
    assert len(ledger_rows) == 1
    assert ledger_rows[0]["run_id"] == "test_run_led"
    assert ledger_rows[0]["verdict"] == "IMPROVED"
    assert ledger_rows[0]["note"] == "测试记账"


def test_trend_format():
    trend_items = [
        {
            "run_id": "run_1",
            "created_at": "2026-09-25T10:00:00+00:00",
            "wall_minutes": 15.0,
            "commit": "a1b2c3d4",
            "note": "初次运行",
            "config_hash": "cfg_same",
            "value": 0.75,
            "delta": None,
            "verdict": "-",
        },
        {
            "run_id": "run_2",
            "created_at": "2026-09-25T12:00:00+00:00",
            "wall_minutes": 18.0,
            "commit": "e5f6g7h8",
            "note": "修复Prompt",
            "config_hash": "cfg_same",
            "value": 0.88,
            "delta": 0.13,
            "verdict": "[+]",
        },
    ]

    tbl = format_trend_table("effective_novelty_rate", trend_items)
    assert "有效新颖率" in tbl
    assert "09-25 10:00" in tbl
    assert "09-25 12:00" in tbl
    assert "+0.13" in tbl
    assert "[+]" in tbl
