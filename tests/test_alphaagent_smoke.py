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
from alphaagent.factor.mining.delivery_checker import DeliveryChecker
from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria


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

    def test_engine_gate_validation_restored(self) -> None:
        """2026-09-12 回归：35c8b30 误删 engine_gate 校验，freq/数值边界
        不再被拒（非法值可落盘、运行时才炸）——恢复校验并锁定。"""
        # 非法 freq 白名单
        with pytest.raises(ValueError) as e1:
            normalize_research_spec({
                "delivery_policy": {"production": {"engine_gate": {
                    "freq": "hourly", "allowed_freqs": ["daily", "hourly"],
                }}},
            })
        assert "engine_gate" in str(e1.value)
        # freq 不在 allowed_freqs 内
        with pytest.raises(ValueError):
            normalize_research_spec({
                "delivery_policy": {"production": {"engine_gate": {
                    "freq": "monthly", "allowed_freqs": ["daily", "weekly"],
                }}},
            })
        # 数值边界
        for patch in ({"max_drawdown": -5}, {"min_excess_annual": 99}, {"min_invested_ratio": 2}):
            with pytest.raises(ValueError):
                normalize_research_spec({
                    "delivery_policy": {"production": {"engine_gate": patch}},
                })

    def test_fundamental_mode_defaults(self) -> None:
        spec = normalize_research_spec(default_research_spec("fundamental"))
        assert spec["research_mode"] == "fundamental"
        assert spec["recommended_label_col"] == "label_20d_close_to_close"
        assert any(f.startswith("fundamental_") for f in spec["search_policy"]["allowed_signal_families"])

    def test_fundamental_mode_via_override(self) -> None:
        spec = normalize_research_spec({"research_mode": "fundamental"})
        assert spec["research_mode"] == "fundamental"
        assert spec["recommended_label_col"] == "label_20d_close_to_close"

    def test_fundamental_thresholds_anchored_to_20d_scale(self) -> None:
        """fundamental 是 label_20d 档：统计门槛按 **20d 尺度**锚定，与 technical_monthly
        **完全一致**（门槛只由 label 持有期唯一决定，与数据面无关）。

        2026-10-08 定调：此前 fundamental 曾单独标定 0.035/0.45/0.021（"文献月度基本面量级"），
        造成同为 label_20d 却门槛不同（技术月频 0.053/0.65 vs 基本面松 0.035/0.45）。已统一：
        fundamental 与 technical_monthly 共用 _MONTHLY_20D_OVERRIDES（见 core/research_modes.py）。
        """
        tech = normalize_research_spec(default_research_spec("technical"))
        fund = normalize_research_spec(default_research_spec("fundamental"))
        m20 = normalize_research_spec(default_research_spec("technical_monthly"))

        # label 尺度不同（20d vs 1d），绝对值必须更严
        assert fund["evaluation_policy"]["min_train_abs_ic"] > tech["evaluation_policy"]["min_train_abs_ic"]
        assert fund["evaluation_policy"]["min_train_icir"] > tech["evaluation_policy"]["min_train_icir"]
        assert fund["evaluation_policy"]["min_val_abs_ic"] > tech["evaluation_policy"]["min_val_abs_ic"]

        # ★ 与 technical_monthly 同档（同为 label_20d）：全字段一致
        for path, key in [
            ("evaluation_policy", "min_train_abs_ic"),
            ("evaluation_policy", "min_train_icir"),
            ("evaluation_policy", "min_val_abs_ic"),
            ("delivery_policy.candidate", "min_abs_ic"),
            ("delivery_policy.candidate", "min_icir"),
            ("delivery_policy.candidate", "min_val_abs_ic"),
            ("delivery_policy.candidate", "min_val_ic_retention"),
            ("delivery_policy.production", "min_train_abs_ic"),
            ("delivery_policy.production", "min_train_icir"),
            ("delivery_policy.production", "min_val_abs_ic"),
            ("delivery_policy.production", "min_val_ic_retention"),
        ]:
            seg_f, seg_m = fund, m20
            for part in path.split("."):
                seg_f, seg_m = seg_f[part], seg_m[part]
            assert seg_f[key] == seg_m[key], f"{path}.{key} fundamental vs monthly 不一致"

        # 换手性硬门：与 technical_monthly 同值
        assert fund["delivery_policy"]["candidate"]["min_cs_autocorr"] == m20["delivery_policy"]["candidate"]["min_cs_autocorr"] == 0.18
        # engine_gate：月频、年化 0.03/夏普 0.5，与 technical_monthly（及 technical）一致
        assert fund["delivery_policy"]["production"]["engine_gate"]["freq"] == "monthly"
        assert fund["delivery_policy"]["production"]["engine_gate"]["min_excess_annual"] == 0.03
        assert fund["delivery_policy"]["production"]["engine_gate"]["min_excess_sharpe"] == 0.5
        assert fund["delivery_policy"]["production"]["max_winsorized_abs_ic_decay"] == 0.10

    def test_fundamental_override_preserves_defaults(self) -> None:
        """门槛只由 label 决定：fundamental（label_20d）与 technical_monthly 同门槛，且同名
        显式覆盖不影响它模式。"""
        fund = normalize_research_spec({"research_mode": "fundamental"})
        assert fund["delivery_policy"]["candidate"]["min_abs_ic"] == 0.053
        assert fund["delivery_policy"]["production"]["min_train_abs_ic"] == 0.0663
        assert fund["delivery_policy"]["production"]["engine_gate"]["min_excess_annual"] == 0.03

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

    @pytest.fixture()
    def checker(self) -> DeliveryChecker:
        return DeliveryChecker(DeliveryCriteria.defaults())

    @staticmethod
    def _m(**kw):
        return {
            "ic": 0.04, "icir": 0.4, "coverage": 0.90,
            "cs_pearson_autocorr": 0.6,
            **kw,
        }

    def test_stage_one_pass(self, checker: DeliveryChecker) -> None:
        stats = checker.stage_one_stats(self._m())
        corr = checker.stage_one_correlation({"max_abs_corr": 0.3})
        assert stats.passed and not stats.fail_reasons
        assert corr.passed and not corr.fail_reasons

    def test_stage_one_fail_low_ic(self, checker: DeliveryChecker) -> None:
        stats = checker.stage_one_stats(self._m(ic=0.005))
        assert not stats.passed and "ic" in stats.fail_reasons

    def test_stage_one_fail_high_corr(self, checker: DeliveryChecker) -> None:
        stats = checker.stage_one_stats(self._m())
        corr = checker.stage_one_correlation({"max_abs_corr": 0.75})
        assert stats.passed
        assert not corr.passed and "max_cs_corr" in corr.fail_reasons

    def test_stage_two_pass(self, checker: DeliveryChecker) -> None:
        result = checker.stage_two(
            self.GOOD_METRICS,
            {"ic": 0.05, "val_long_excess": 0.02},
            {"max_abs_corr": 0.25},
        )
        assert result.passed and not result.fail_reasons

    def test_stage_two_fail_val_ic(self, checker: DeliveryChecker) -> None:
        result = checker.stage_two(
            self.GOOD_METRICS,
            {"ic": 0.005},
            {"max_abs_corr": 0.25},
        )
        assert not result.passed and "val_ic" in result.fail_reasons

    def test_stage_two_fail_decay(self, checker: DeliveryChecker) -> None:
        metrics = {**self.GOOD_METRICS, "winsorized_abs_ic_decay": 0.15}
        result = checker.stage_two(
            metrics,
            {"ic": 0.05, "val_long_excess": 0.02},
            {"max_abs_corr": 0.25},
        )
        assert not result.passed and "winsorized_abs_ic_decay" in result.fail_reasons


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
        assert store.recent()[0] == []
        stats = store.statistics()
        assert stats["entries"] == 0

    def test_record_and_retrieve(self, store: ResearchMemoryStore) -> None:
        row = _make_tool_row("evaluate_factor", "TS_MEAN($close, 5)", {"ic": 0.03, "icir": 0.4})
        entry = store.record_tool_result(run_id="r1", row=row)
        assert entry is not None
        recent, _total = store.recent(limit=10)
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
        assert len(fresh.recent()[0]) == 1

    # ── Phase 5: 结构解析器与编辑模式单元测试 ──────────────────────────

    def test_structure_fingerprint_stable(self, store: ResearchMemoryStore) -> None:
        """相同结构不同变量应产生相同指纹；不同算子结构不同指纹。"""
        from alphaagent.factor.mining.research_memory import _structure_fingerprint
        fp1 = _structure_fingerprint("rank(ts_mean($close, 5))")
        fp2 = _structure_fingerprint("rank(ts_mean($vwap, 5))")
        fp3 = _structure_fingerprint("rank(ts_std($close, 5))")
        assert fp1 == fp2  # 变量替换不影响指纹
        assert fp1 != fp3  # 算子不同影响指纹

