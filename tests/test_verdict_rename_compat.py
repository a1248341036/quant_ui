# -*- coding: utf-8 -*-
"""verdict 改名（promising → train_passed）的兼容性测试（2026-10-01）。

背景：``promising`` 一词在评估层表示"只过训练集"，却被记忆/检索层当作正向结果用，
实测在多次复盘里被误读成"已成候选/已达标"。故改名为 ``train_passed``。
历史记忆库（``artifacts/alphaagent/research_memory.db``）里仍是旧字面量，
本测试固定"改名不改变历史数据口径"这一约束。
"""
from __future__ import annotations

import json

from alphaagent.factor.mining.memory import constants as C
from alphaagent.factor.mining.research_memory import ResearchMemoryStore


def test_constants_expose_both_generations():
    assert C.VERDICT_TRAIN_PASS == "train_passed"
    assert C.LEGACY_VERDICT_TRAIN_PASS == "promising"
    assert C.VERDICT_TRAIN_PASS in C.POSITIVE_VERDICTS
    # 读侧集合必须同时含新旧字面量，否则历史统计会漂
    assert C.VERDICT_TRAIN_PASS in C.POSITIVE_VERDICTS_READ
    assert C.LEGACY_VERDICT_TRAIN_PASS in C.POSITIVE_VERDICTS_READ
    assert C.TRAIN_PASS_VERDICTS == ("train_passed", "promising")


def test_normalize_verdict_maps_legacy_only():
    assert C.normalize_verdict("promising") == "train_passed"
    assert C.normalize_verdict("train_passed") == "train_passed"
    # 其它 verdict 原样返回，不受影响
    for other in ("weak", "rejected", "candidate_approved", "", None):
        assert C.normalize_verdict(other) == str(other or "")


def _row(factor_name: str, verdict: str) -> dict:
    args = {"multi_line_expr": f"RANK(${factor_name}_src)", "factor_name": factor_name}
    return {
        "name": "eval_on_train_set",
        "arguments_raw": json.dumps(args),
        "result": {"ok": True, "split": "train", "verdict": verdict,
                   "metrics": {"ic": 0.03, "icir": 0.4, "factor_coverage": 0.9}},
    }


def test_legacy_promising_rows_still_counted_as_train_passed(tmp_path):
    """历史行（verdict='promising'）与新行（train_passed）在饱和度统计里等价计入。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(run_id="r_legacy", row=_row("legacy_fac", "promising"))
    store.record_tool_result(run_id="r_new", row=_row("new_fac", "train_passed"))

    sat = store.compute_saturation()
    # 两个因子同属某个族，族内 n_promising 至少为 1（两条都应被计入）
    total = sum(int(v.get("n_promising") or 0) for v in sat.values()) if isinstance(sat, dict) else 0
    assert total >= 2, f"历史 promising 行未被计入: {sat}"
