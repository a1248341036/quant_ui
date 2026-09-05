# -*- coding: utf-8 -*-
"""P0/P1 挖掘产出率改进的回归测试（2026-09-05）。

覆盖：
- P0-1 _attach_yield_hints：训练过线 → submit_decision_required；未过线不加
- P0-2 near_miss：near-miss 判定（dispatch 宽口径 + memory._classify 档位口径）
  与 verdict=near_miss 的 conclusion 文案
- P1-4 PIT 警戒：train |IC| ≥ 0.045 → pit_warning
- P1-5 _yield_block：family 过线率统计块（零产出族 + 高产出族）
"""
from __future__ import annotations

import pytest

from alphaagent.factor.mining.tools._dispatch import (
    _attach_yield_hints,
    _near_miss_verdict,
)


def _result(ic=0.024, icir=0.3, cov=0.99, passed=True, split="train"):
    return {
        "ok": True,
        "passed": passed,
        "split": split,
        "metrics": {
            "cross_sectional_core": {"ic": ic, "icir": icir, "factor_coverage": cov},
        },
    }


class TestSubmitDecisionRequired:
    def test_passed_train_gets_decision_hint(self):
        r = _result()
        _attach_yield_hints(r, "expr", {})
        assert "submit_decision_required" in r
        assert "submit_factor" in r["submit_decision_required"]

    def test_failed_train_no_decision_hint(self):
        r = _result(ic=0.005, passed=False)
        _attach_yield_hints(r, "expr", {})
        assert "submit_decision_required" not in r

    def test_val_split_no_decision_hint(self):
        r = _result(split="val")
        _attach_yield_hints(r, "expr", {})
        assert "submit_decision_required" not in r


class TestNearMissHint:
    def test_near_miss_hint_injected(self):
        # |IC|=0.017 ∈ [0.012, 0.02)，ICIR>0.2，coverage>0.85 → near_miss 提示
        r = _result(ic=0.017, icir=0.35, cov=0.99, passed=False)
        _attach_yield_hints(r, "expr", {})
        assert "near_miss_hint" in r
        assert "窗口微调" in r["near_miss_hint"]

    def test_far_below_no_hint(self):
        r = _result(ic=0.005, icir=0.1, cov=0.99, passed=False)
        _attach_yield_hints(r, "expr", {})
        assert "near_miss_hint" not in r

    def test_pit_warning_overrides_near_miss(self):
        # |IC|=0.08 同时满足 PIT 高线 → 只给 PIT 警戒（return 提前）
        r = _result(ic=0.08, icir=0.9, cov=0.99, passed=False)
        _attach_yield_hints(r, "expr", {})
        assert "pit_warning" in r
        assert "near_miss_hint" not in r

    def test_pit_warning_threshold(self):
        r = _result(ic=0.046, icir=0.9, cov=0.99, passed=False)
        _attach_yield_hints(r, "expr", {})
        assert "pit_warning" in r
        r2 = _result(ic=0.043, icir=0.9, cov=0.99, passed=False)
        _attach_yield_hints(r2, "expr", {})
        assert "pit_warning" not in r2


class TestNearMissVerdict:
    def test_dispatch_helper_technical(self):
        assert _near_miss_verdict({"ic": 0.017, "icir": 0.3, "factor_coverage": 0.99}) is True
        assert _near_miss_verdict({"ic": 0.010, "icir": 0.3, "factor_coverage": 0.99}) is False
        assert _near_miss_verdict({"ic": 0.017, "icir": 0.1, "factor_coverage": 0.99}) is False
        assert _near_miss_verdict({"ic": 0.017, "icir": 0.3, "factor_coverage": 0.5}) is False
        # 过线的不属于 near_miss
        assert _near_miss_verdict({"ic": 0.025, "icir": 0.3, "factor_coverage": 0.99}) is False

    def test_dispatch_helper_fundamental_mode(self):
        assert _near_miss_verdict(
            {"ic": 0.013, "icir": 0.3, "factor_coverage": 0.99, "research_mode": "fundamental"}
        ) is True
        assert _near_miss_verdict(
            {"ic": 0.013, "icir": 0.3, "factor_coverage": 0.99, "research_mode": "technical"}
        ) is False  # technical 档 0.013 < 0.8×0.02=0.016

    def test_memory_classify_near_miss(self):
        from alphaagent.factor.mining.memory.schema import SchemaMixin

        result = {"ok": True, "split": "train"}
        metrics = {"ic": 0.017, "icir": 0.35, "factor_coverage": 0.99}
        verdict, conclusion = SchemaMixin._classify(
            "evaluate_factor", result, metrics, error=""
        )
        assert verdict == "near_miss"
        assert "接近海选线" in conclusion
        # fundamental 档：IC 0.013 应为 near_miss 而非 weak
        metrics_f = {"ic": 0.013, "icir": 0.30, "factor_coverage": 0.99,
                     "research_mode": "fundamental"}
        verdict_f, _ = SchemaMixin._classify(
            "evaluate_factor", result, metrics_f, error=""
        )
        assert verdict_f == "near_miss"

    def test_memory_classify_promising_unchanged(self):
        from alphaagent.factor.mining.memory.schema import SchemaMixin

        result = {"ok": True, "split": "train"}
        metrics = {"ic": 0.025, "icir": 0.35, "factor_coverage": 0.99}
        verdict, _ = SchemaMixin._classify(
            "evaluate_factor", result, metrics, error=""
        )
        assert verdict == "promising"


class TestConstants:
    def test_near_miss_not_positive(self):
        """near_miss 不得触发重复提交拦截（不在 POSITIVE_VERDICTS）。"""
        from alphaagent.factor.mining.memory.constants import (
            NEGATIVE_VERDICTS,
            POSITIVE_VERDICTS,
            VERDICT_WEIGHT,
        )

        assert "near_miss" in NEGATIVE_VERDICTS
        assert "near_miss" not in POSITIVE_VERDICTS
        assert VERDICT_WEIGHT["near_miss"] == -0.2


class TestYieldBlock:
    def test_yield_block_lists_zero_and_productive(self, tmp_path):
        from alphaagent.factor.mining.memory.store import ResearchMemoryStore

        store = ResearchMemoryStore(str(tmp_path / "mem.db"))
        with store._open() as conn:
            rows = []
            # volume：30 次 weak（零产出）
            for i in range(30):
                rows.append((f"fp_vol_{i}", f"v_{i}", "f = TS_MEAN($volume,{i})",
                             "weak", "volume", "2026-09-05"))
            # gap_overnight：20 次，5 个 promising
            for i in range(20):
                rows.append((f"fp_gap_{i}", f"g_{i}", "f = TS_MEAN($open,{i})",
                             "promising" if i < 5 else "weak", "gap_overnight",
                             "2026-09-05"))
            conn.executemany(
                "INSERT INTO memory_entries (id, factor_name, expression, verdict, family, "
                "created_at, updated_at, attempts, structure_fingerprint) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, 1, ?)",
                [(r[0], r[1], r[2], r[3], r[4], r[5], r[5], r[0]) for r in rows],
            )
            conn.commit()
        block = store._yield_block(min_attempts=20)
        assert "信号族产出率" in block
        assert "gap_overnight" in block and "25.0%" in block
        assert "volume" in block and "0 过线" in block
