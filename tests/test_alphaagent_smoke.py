"""AlphaAgent 冒烟测试：DSL 解析求值、研究记忆、研究策略、提交门槛。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphaagent.dsl.eval import eval_multi_line_factor
from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from alphaagent.factor.mining.research_spec import (
    default_research_spec,
    normalize_research_spec,
)
from alphaagent.factor.mining.submit import _check_stage_one, _check_stage_two


# ── DSL parser + eval ──────────────────────────────────────────────


@pytest.fixture()
def mini_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 60
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.02, (n, 3)), axis=0)
    volume = rng.uniform(1e6, 5e6, (n, 3))
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["datetime", "instrument"])
    return pd.DataFrame({"close": close.ravel(), "volume": volume.ravel()}, index=idx)


class TestDslEval:
    def test_single_line_ts_mean(self, mini_panel: pd.DataFrame) -> None:
        result = eval_multi_line_factor("TS_MEAN($close, 5)", mini_panel)
        assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_multi_line_with_assignment(self, mini_panel: pd.DataFrame) -> None:
        expr = "x = DELTA($close, 1)\nTS_MEAN(x, 10)"
        result = eval_multi_line_factor(expr, mini_panel)
        assert result is not None
        assert len(result) > 0

    def test_cs_rank_operator(self, mini_panel: pd.DataFrame) -> None:
        result = eval_multi_line_factor("RANK(TS_PCTCHANGE($close, 5))", mini_panel)
        assert result is not None

    def test_invalid_syntax_raises(self, mini_panel: pd.DataFrame) -> None:
        with pytest.raises(MultiLineFactorEvalError):
            eval_multi_line_factor("TS_MEAN(", mini_panel)

    def test_unknown_column_raises(self, mini_panel: pd.DataFrame) -> None:
        with pytest.raises(MultiLineFactorEvalError):
            eval_multi_line_factor("TS_MEAN($nonexistent_col, 5)", mini_panel)

    def test_volume_based_factor(self, mini_panel: pd.DataFrame) -> None:
        expr = "v = TS_STD($volume, 20)\nRANK(v)"
        result = eval_multi_line_factor(expr, mini_panel)
        assert result is not None


# ── Research spec ──────────────────────────────────────────────────


class TestResearchSpec:
    def test_default_is_valid(self) -> None:
        spec = normalize_research_spec(default_research_spec())
        assert spec["version"] == 1
        assert isinstance(spec["search_policy"]["allowed_signal_families"], list)
        assert spec["evaluation_policy"]["min_train_abs_ic"] > 0

    def test_merge_override(self) -> None:
        override = {"evaluation_policy": {"min_train_abs_ic": 0.05}}
        spec = normalize_research_spec(override)
        assert spec["evaluation_policy"]["min_train_abs_ic"] == 0.05
        # other defaults preserved
        assert spec["search_policy"]["max_candidates_per_round"] == default_research_spec()["search_policy"]["max_candidates_per_round"]

    def test_invalid_version_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"version": 99})

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec("not_a_dict")

    def test_bad_ic_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"evaluation_policy": {"min_train_abs_ic": -1}})

    def test_fundamental_mode_defaults(self) -> None:
        spec = normalize_research_spec(default_research_spec("fundamental"))
        assert spec["research_mode"] == "fundamental"
        assert spec["recommended_label_col"] == "label_10d_close_to_close"
        assert any(f.startswith("fundamental_") for f in spec["search_policy"]["allowed_signal_families"])

    def test_fundamental_mode_via_override(self) -> None:
        spec = normalize_research_spec({"research_mode": "fundamental"})
        assert spec["research_mode"] == "fundamental"
        assert spec["recommended_label_col"] == "label_10d_close_to_close"

    def test_fundamental_thresholds_looser_than_technical(self) -> None:
        """基本面模式门槛应比 technical 放宽（慢因子弱信号），但可交易性门禁保留。"""
        tech = normalize_research_spec(default_research_spec("technical"))
        fund = normalize_research_spec(default_research_spec("fundamental"))

        # 统计门槛：基本面必须 ≤ technical（宽松或持平），不允许更严
        assert fund["evaluation_policy"]["min_train_abs_ic"] <= tech["evaluation_policy"]["min_train_abs_ic"]
        assert fund["evaluation_policy"]["min_train_icir"] <= tech["evaluation_policy"]["min_train_icir"]
        assert fund["evaluation_policy"]["min_val_abs_ic"] <= tech["evaluation_policy"]["min_val_abs_ic"]
        assert fund["delivery_policy"]["candidate"]["min_abs_ic"] <= tech["delivery_policy"]["candidate"]["min_abs_ic"]
        assert fund["delivery_policy"]["candidate"]["min_icir"] <= tech["delivery_policy"]["candidate"]["min_icir"]
        assert fund["delivery_policy"]["production"]["min_train_abs_ic"] <= tech["delivery_policy"]["production"]["min_train_abs_ic"]
        assert fund["delivery_policy"]["production"]["min_train_icir"] <= tech["delivery_policy"]["production"]["min_train_icir"]
        assert fund["delivery_policy"]["production"]["min_val_abs_ic"] <= tech["delivery_policy"]["production"]["min_val_abs_ic"]

        # 换手性硬门：基本面保留（防排名日度剧变）
        assert fund["delivery_policy"]["candidate"]["min_cs_autocorr"] == 0.18
        # engine_gate：月频、超额与夏普门槛不低于 technical 的 2/3 力度
        assert fund["delivery_policy"]["production"]["engine_gate"]["freq"] == "monthly"
        assert fund["delivery_policy"]["production"]["engine_gate"]["min_excess_annual"] < tech["delivery_policy"]["production"]["engine_gate"]["min_excess_annual"]
        assert fund["delivery_policy"]["production"]["engine_gate"]["min_excess_sharpe"] < tech["delivery_policy"]["production"]["engine_gate"]["min_excess_sharpe"]

    def test_fundamental_override_preserves_defaults(self) -> None:
        """用户显式覆盖门槛时，基本面默认值不应污染 technical 默认。"""
        fund = normalize_research_spec({"research_mode": "fundamental"})
        assert fund["delivery_policy"]["candidate"]["min_abs_ic"] == 0.012
        assert fund["delivery_policy"]["production"]["min_train_abs_ic"] == 0.020
        assert fund["delivery_policy"]["production"]["engine_gate"]["min_excess_annual"] == 0.02

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"research_mode": "macro"})


# ── Submit gating ─────────────────────────────────────────────────


class TestSubmitGating:
    GOOD_METRICS = {
        "ic": 0.06,
        "icir": 0.8,
        "coverage": 0.95,
        "long_group_annual_excess_return": 0.08,
        "winsorized_abs_ic_decay": 0.03,
        "mls_fmb": {"nw_t_ls": 3.2},
    }

    def test_stage_one_pass(self) -> None:
        ok, reasons = _check_stage_one(
            {"ic": 0.04, "icir": 0.4, "coverage": 0.90},
            {"max_abs_corr": 0.3},
        )
        assert ok and not reasons

    def test_stage_one_fail_low_ic(self) -> None:
        ok, reasons = _check_stage_one(
            {"ic": 0.005, "icir": 0.4, "coverage": 0.90},
            {"max_abs_corr": 0.3},
        )
        assert not ok and "ic" in reasons

    def test_stage_one_fail_high_corr(self) -> None:
        ok, reasons = _check_stage_one(
            {"ic": 0.04, "icir": 0.4, "coverage": 0.90},
            {"max_abs_corr": 0.75},
        )
        assert not ok and "max_cs_corr" in reasons

    def test_stage_two_pass(self) -> None:
        ok, reasons = _check_stage_two(
            self.GOOD_METRICS,
            {"ic": 0.05, "val_long_excess": 0.02},
            {"max_abs_corr": 0.25},
        )
        assert ok and not reasons

    def test_stage_two_fail_val_ic(self) -> None:
        ok, reasons = _check_stage_two(
            self.GOOD_METRICS,
            {"ic": 0.005},
            {"max_abs_corr": 0.25},
        )
        assert not ok and "val_ic" in reasons

    def test_stage_two_fail_decay(self) -> None:
        metrics = {**self.GOOD_METRICS, "winsorized_abs_ic_decay": 0.15}
        ok, reasons = _check_stage_two(
            metrics,
            {"ic": 0.05, "val_long_excess": 0.02},
            {"max_abs_corr": 0.25},
        )
        assert not ok and "winsorized_abs_ic_decay" in reasons


# ── Research memory ────────────────────────────────────────────────


def _make_tool_row(name: str, expr: str, metrics: dict, verdict_hint: str = "") -> dict:
    return {
        "name": name,
        "arguments_raw": json.dumps({"multi_line_expr": expr, "factor_name": "test_factor"}),
        "result": {"ok": True, "metrics": metrics},
    }


class TestResearchMemory:
    @pytest.fixture()
    def store(self, tmp_path):
        return ResearchMemoryStore(tmp_path / "memory.json")

    def test_empty_store(self, store: ResearchMemoryStore) -> None:
        assert store.recent() == []
        stats = store.statistics()
        assert stats["entries"] == 0

    def test_record_and_retrieve(self, store: ResearchMemoryStore) -> None:
        row = _make_tool_row("evaluate_factor", "TS_MEAN($close, 5)", {"ic": 0.03, "icir": 0.4})
        entry = store.record_tool_result(run_id="r1", row=row)
        assert entry is not None
        recent = store.recent(limit=10)
        assert len(recent) >= 1
        assert recent[0]["factor_name"] == "test_factor"

    def test_ignores_unknown_tool(self, store: ResearchMemoryStore) -> None:
        row = _make_tool_row("some_random_tool", "expr", {})
        assert store.record_tool_result(run_id="r1", row=row) is None

    def test_dedup_by_expression_hash(self, store: ResearchMemoryStore) -> None:
        row = _make_tool_row("evaluate_factor", "TS_MEAN($close, 5)", {"ic": 0.03})
        e1 = store.record_tool_result(run_id="r1", row=row)
        e2 = store.record_tool_result(run_id="r2", row=dict(row))
        assert e1["id"] == e2["id"]

    def test_persist_roundtrip(self, store: ResearchMemoryStore) -> None:
        row = _make_tool_row("evaluate_factor", "DELTA($close, 1)", {"ic": 0.02})
        store.record_tool_result(run_id="r1", row=row)
        fresh = ResearchMemoryStore(store.path)
        assert len(fresh.recent()) == 1
