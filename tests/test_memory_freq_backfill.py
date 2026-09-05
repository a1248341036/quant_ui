# -*- coding: utf-8 -*-
"""调仓频率随评估落库 + 存量回填（run spec 快照口径）。"""
import json

from alphaagent.factor.mining.research_memory import ResearchMemoryStore


def _eval_row(name="eval_on_train_set", expr="RANK(TS_MEAN($adj_close, 10))", factor_name="demo_f", ic=0.03):
    args = {"multi_line_expr": expr, "factor_name": factor_name}
    result = {"ok": True, "split": "train", "metrics": {"ic": ic, "icir": 0.4, "factor_coverage": 0.9}}
    return {"name": name, "arguments_raw": json.dumps(args), "result": result}


def test_record_with_run_freq_context(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    entry = store.record_tool_result(
        run_id="r1",
        row=_eval_row(),
        run_freq_context={"rebalance_freq": "weekly", "research_mode": "technical", "freq_source": "run_spec"},
    )
    assert entry["metrics"]["rebalance_freq"] == "weekly"
    assert entry["metrics"]["research_mode"] == "technical"
    assert entry["metrics"]["freq_source"] == "run_spec"

    # 不带 context 的调用不受影响
    entry2 = store.record_tool_result(
        run_id="r2",
        row=_eval_row(expr="RANK(TS_MEAN($adj_close, 22))", factor_name="demo_g"),
    )
    assert "rebalance_freq" not in entry2["metrics"]


def test_backfill_freq_from_run_specs(tmp_path):
    runs_root = tmp_path / "ui"
    # run A：spec 带 weekly/technical；run B：spec 无 engine_gate.freq；run C：目录缺失
    (runs_root / "ra").mkdir(parents=True)
    (runs_root / "ra" / "research_spec.json").write_text(json.dumps({
        "research_mode": "technical",
        "delivery_policy": {"production": {"engine_gate": {"freq": "weekly", "allowed_freqs": ["daily", "weekly"]}}},
    }), encoding="utf-8")
    (runs_root / "rb").mkdir(parents=True)
    (runs_root / "rb" / "research_spec.json").write_text(json.dumps({"research_mode": "technical"}), encoding="utf-8")

    store = ResearchMemoryStore(tmp_path / "m.db")
    # e1：记录时即带 run_spec 频率（新链路口径）→ 回填必须幂等跳过
    e1 = store.record_tool_result(
        run_id="ra",
        row=_eval_row(expr="RANK(TS_MEAN($adj_close, 10))", factor_name="f1"),
        run_freq_context={"rebalance_freq": "weekly", "research_mode": "technical", "freq_source": "run_spec"},
    )
    e2 = store.record_tool_result(run_id="rb", row=_eval_row(expr="RANK(TS_MEAN($adj_close, 11))", factor_name="f2"))
    e3 = store.record_tool_result(run_id="rc", row=_eval_row(expr="RANK(TS_MEAN($adj_close, 12))", factor_name="f3"))
    e4 = store.record_tool_result(run_id="ra", row=_eval_row(expr="RANK(TS_MEAN($adj_close, 13))", factor_name="f4"))

    summary = store.backfill_freq_from_run_specs(runs_root)
    assert summary["scanned"] == 4
    assert summary["updated"] == 2  # e2 只补 mode；e4 补 freq+mode
    assert summary["skipped_present"] == 1  # e1 已带 run_spec 记录，幂等跳过
    assert summary["unresolvable"] == 1  # e3 的 run 目录缺失

    by_id = {e["id"]: e for e in store.recent(limit=10, offset=0, order="recent")[0]}
    assert by_id[e1["id"]]["metrics"]["rebalance_freq"] == "weekly"  # 幂等未覆盖
    assert by_id[e1["id"]]["metrics"]["freq_source"] == "run_spec"
    assert by_id[e2["id"]]["metrics"].get("rebalance_freq") is None  # spec 无 freq → freq 不动
    assert by_id[e2["id"]]["metrics"]["research_mode"] == "technical"  # mode 照常回填
    assert by_id[e2["id"]]["metrics"]["freq_source"] == "derived_run_spec"
    assert "rebalance_freq" not in by_id[e3["id"]]["metrics"]  # run 目录缺失 → 不动
    assert by_id[e4["id"]]["metrics"]["rebalance_freq"] == "weekly"
    assert by_id[e4["id"]]["metrics"]["freq_source"] == "derived_run_spec"
    assert by_id[e4["id"]]["metrics"]["research_mode"] == "technical"

    # 再次运行：全部跳过（幂等）；e3 依旧 unresolvable
    summary2 = store.backfill_freq_from_run_specs(runs_root)
    assert summary2["updated"] == 0
    assert summary2["skipped_present"] == 3
    assert summary2["unresolvable"] == 1
