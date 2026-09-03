# -*- coding: utf-8 -*-
"""A) 预测-对账 与 E) 门控消融 的单元测试。

覆盖：
- 十分位形态分类（单调 / 倒U / U型 / 不规则）——倒U 用例取自 2026-09 实测反转因子数据
- prediction_check 的 confirmed / partial / contradicted / unverifiable 判定
- dispatch 层 prediction 必填与结果注入
- 门控表达式的 base-only 自动消融与 ablation_hint
- 记忆层对账摘要持久化 + 被证伪结论追加
"""
from __future__ import annotations

import json

from alphaagent.factor.mining.eval.prediction import (
    build_ablation_check,
    build_prediction_check,
    classify_decile_shape,
    normalize_prediction,
)
from alphaagent.factor.mining.tools import FactorEvalTools

# 实测（2026-09）：20d 反转因子 train 段十分位——倒 U、峰值 D7、D10 回落、中间组最强
_INVERTED_U = [
    {"decile": 1, "mean_label": -0.00545},
    {"decile": 2, "mean_label": 0.003216},
    {"decile": 3, "mean_label": 0.007542},
    {"decile": 4, "mean_label": 0.010319},
    {"decile": 5, "mean_label": 0.013679},
    {"decile": 6, "mean_label": 0.015610},
    {"decile": 7, "mean_label": 0.015966},
    {"decile": 8, "mean_label": 0.014903},
    {"decile": 9, "mean_label": 0.011078},
    {"decile": 10, "mean_label": 0.006927},
]

# 同因子 val 段：峰值 D8、尾部轻度回落—— legitimately 单调爬升（边界用例）
_SOFTENED_TAIL = [
    {"decile": 1, "mean_label": -0.008037},
    {"decile": 2, "mean_label": 0.001607},
    {"decile": 3, "mean_label": 0.003618},
    {"decile": 4, "mean_label": 0.004964},
    {"decile": 5, "mean_label": 0.005317},
    {"decile": 6, "mean_label": 0.006106},
    {"decile": 7, "mean_label": 0.007985},
    {"decile": 8, "mean_label": 0.010076},
    {"decile": 9, "mean_label": 0.008877},
    {"decile": 10, "mean_label": 0.005743},
]


def _monotonic() -> list[dict]:
    return [{"decile": i, "mean_label": 0.001 * i} for i in range(1, 11)]


class TestClassifyDecileShape:
    def test_monotonic_increasing(self):
        actual = classify_decile_shape(_monotonic())
        assert actual["shape"] == "monotonic_increasing"
        assert actual["strong_side"] == "high_factor"
        assert actual["spearman"] > 0.99

    def test_inverted_u_real_case(self):
        actual = classify_decile_shape(_INVERTED_U)
        assert actual["shape"] == "inverted_u"
        assert actual["strong_side"] == "middle"
        assert actual["peak_decile"] == 7
        assert actual["d1"] < 0 < actual["d10"]

    def test_softened_tail_is_monotonic_boundary(self):
        """峰值 D8、仅尾部轻度回落：判单调（不是倒U）——分类边界锁定。"""
        actual = classify_decile_shape(_SOFTENED_TAIL)
        assert actual["shape"] == "monotonic_increasing"
        assert actual["strong_side"] == "high_factor"

    def test_monotonic_decreasing(self):
        rows = [{"decile": i, "mean_label": 0.011 - 0.001 * i} for i in range(1, 11)]
        actual = classify_decile_shape(rows)
        assert actual["shape"] == "monotonic_decreasing"
        assert actual["strong_side"] == "low_factor"

    def test_insufficient_deciles_returns_none(self):
        assert classify_decile_shape(None) is None
        assert classify_decile_shape([{"decile": 1, "mean_label": 0.01}]) is None
        assert classify_decile_shape([{"decile": i, "mean_label": None} for i in range(1, 11)]) is None


class TestBuildPredictionCheck:
    _PRED = {"expected_shape": "monotonic_increasing", "expected_strong_side": "high_factor", "expected_sign": 1}

    def test_confirmed(self):
        check = build_prediction_check(self._PRED, ic=0.03, decile_rows=_monotonic())
        assert check["verdict"] == "confirmed"
        assert "预测对账通过" in check["message"]

    def test_contradicted_side_mismatch(self):
        pred = {**self._PRED, "expected_strong_side": "high_factor", "falsifier": "若中间组最强则压力位假设不成立"}
        check = build_prediction_check(pred, ic=0.03, decile_rows=_INVERTED_U)
        assert check["verdict"] == "contradicted"
        assert "证伪条件触发" in check["message"]
        # 多头端可交易性提示（预期高因子端但实际不在高因子端）
        assert "纯多头" in check["message"]

    def test_sign_mismatch_downgrades(self):
        check = build_prediction_check(self._PRED, ic=-0.03, decile_rows=_monotonic())
        assert check["verdict"] == "contradicted"
        assert "IC 符号与预期相反" in check["message"]

    def test_partial_shape_only(self):
        pred = {**self._PRED, "expected_shape": "inverted_u", "expected_strong_side": "high_factor"}
        check = build_prediction_check(pred, ic=0.03, decile_rows=_monotonic())
        assert check["verdict"] == "partial"

    def test_unverifiable_without_deciles(self):
        check = build_prediction_check(self._PRED, ic=None, decile_rows=None)
        assert check["verdict"] == "unverifiable"

    def test_none_prediction_returns_none(self):
        assert build_prediction_check(None, ic=0.03, decile_rows=_monotonic()) is None
        assert normalize_prediction({"expected_shape": "bogus"}) is None


class _ProfileServiceWithDeciles:
    """模拟 eval_profile：返回带 decile 的引擎原生 shape；记录全部请求。"""

    def __init__(self, decile_rows=None):
        self.requests: list = []
        self.decile_rows = decile_rows if decile_rows is not None else _monotonic()

    def eval_profile(self, request):
        self.requests.append(request)
        return {
            "ok": True,
            "split": "train",
            "profile": {"profile_id": request.profile_id},
            "profile_hash": "h",
            "candidate": {"candidate_id": "c1", "factor_name": request.factor_name},
            "metrics": {
                "cross_sectional_core": {
                    "ic": 0.03,
                    "icir": 0.4,
                    "decile_mean_label": self.decile_rows,
                },
            },
        }

    class _Sessions:
        def get(self, _sid):
            return object()

    sessions = _Sessions()

    class _Engine:
        def evaluate(self, _session, **kwargs):
            # base-only 消融走引擎直调：返回弱一半的 IC
            return {
                "ok": True,
                "metrics": {"cross_sectional_core": {"ic": 0.055, "icir": 0.5}},
            }

    evaluation_engine = _Engine()


_PRED_OK = {"expected_shape": "monotonic_increasing", "expected_strong_side": "high_factor", "expected_sign": 1}


class TestDispatchPredictionAndAblation:
    def test_prediction_check_attached(self):
        service = _ProfileServiceWithDeciles()
        tools = FactorEvalTools(service, "session")
        result = tools.dispatch(
            "evaluate_factor",
            {
                "profile_id": "train_screen",
                "multi_line_expr": "TS_MEAN($adj_close, 20)",
                "prediction": _PRED_OK,
            },
        )
        assert result["ok"]
        assert result["prediction_check"]["verdict"] == "confirmed"

    def test_gated_expression_ablation_with_base_expr(self):
        service = _ProfileServiceWithDeciles()
        tools = FactorEvalTools(service, "session")
        result = tools.dispatch(
            "evaluate_factor",
            {
                "profile_id": "train_screen",
                "multi_line_expr": "GATED_SIGNAL(TS_MEAN($adj_close, 20), RANK($volume), 0.8, true, 0)",
                "prediction": _PRED_OK,
                "interaction": {
                    "interaction_type": "gated_signal",
                    "base_signal": "趋势",
                    "condition_signal": "量",
                    "economic_mechanism": "放量趋势延续性更强",
                    "base_expr": "TS_MEAN($adj_close, 20)",
                },
            },
        )
        assert result["ok"]
        check = result["ablation_check"]
        assert check["base_ic"] == 0.055
        assert check["full_ic"] == 0.03
        assert check["verdict"] == "conditioning_destroyed_value"
        assert len(service.requests) == 1  # base-only 走引擎直调，不重复占用 eval_profile

    def test_gated_expression_without_base_expr_gets_hint(self):
        service = _ProfileServiceWithDeciles()
        tools = FactorEvalTools(service, "session")
        result = tools.dispatch(
            "evaluate_factor",
            {
                "profile_id": "train_screen",
                "multi_line_expr": "GATED_SIGNAL(TS_MEAN($adj_close, 20), RANK($volume), 0.8, true, 0)",
                "prediction": _PRED_OK,
            },
        )
        assert result["ok"]
        assert "base_expr" in result["ablation_hint"]

    def test_nongated_expression_no_ablation(self):
        service = _ProfileServiceWithDeciles()
        tools = FactorEvalTools(service, "session")
        result = tools.dispatch(
            "evaluate_factor",
            {
                "profile_id": "train_screen",
                "multi_line_expr": "TS_MEAN($adj_close, 20)",
                "prediction": _PRED_OK,
            },
        )
        assert result["ok"]
        assert "ablation_check" not in result
        assert "ablation_hint" not in result


class TestMemoryIngestionPrediction:
    def _record(self, tmp_path, result):
        from alphaagent.factor.mining.research_memory import ResearchMemoryStore

        store = ResearchMemoryStore(tmp_path / "rm.json")
        entry = store.record_tool_result(
            run_id="r1",
            row={
                "name": "evaluate_factor",
                "arguments_raw": json.dumps({
                    "multi_line_expr": "TS_MEAN($adj_close, 20)",
                    "prediction": _PRED_OK,
                }),
                "result": result,
            },
        )
        return store, entry

    def test_prediction_check_persisted_in_observation(self, tmp_path):
        result = {
            "ok": True,
            "split": "train",
            "profile": {"profile_id": "train_screen"},
            "candidate": {"candidate_id": "c1"},
            "metrics": {"cross_sectional_core": {"ic": 0.03, "icir": 0.4, "decile_mean_label": _INVERTED_U}},
            "prediction_check": {
                "verdict": "contradicted",
                "expected": _PRED_OK,
                "actual": {"shape": "inverted_u", "strong_side": "middle"},
                "message": "预测被证伪：预期单调递增/高因子端，实际倒U型/中间组。",
            },
        }
        _store, entry = self._record(tmp_path, result)
        assert entry is not None
        assert "预测对账:" in entry["conclusion"]
        assert entry["metrics"]["prediction_check"]["verdict"] == "contradicted"

    def test_confirmed_check_not_appended_to_conclusion(self, tmp_path):
        result = {
            "ok": True,
            "split": "train",
            "profile": {"profile_id": "train_screen"},
            "candidate": {"candidate_id": "c1"},
            "metrics": {"cross_sectional_core": {"ic": 0.03, "icir": 0.4, "decile_mean_label": _monotonic()}},
            "prediction_check": {
                "verdict": "confirmed",
                "expected": _PRED_OK,
                "actual": {"shape": "monotonic_increasing", "strong_side": "high_factor"},
                "message": "预测对账通过。",
            },
        }
        _store, entry = self._record(tmp_path, result)
        assert entry is not None
        assert "预测对账:" not in entry["conclusion"]
        assert entry["metrics"]["prediction_check"]["verdict"] == "confirmed"


class TestBuildAblationCheck:
    def test_flipped_signal(self):
        check = build_ablation_check({"ic": 0.039}, {"ic": -0.005}, base_expr="x")
        assert check["verdict"] == "conditioning_flipped_signal"

    def test_destroyed_value(self):
        check = build_ablation_check({"ic": 0.05}, {"ic": 0.02}, base_expr="x")
        assert check["verdict"] == "conditioning_destroyed_value"

    def test_added_value(self):
        check = build_ablation_check({"ic": 0.02}, {"ic": 0.04}, base_expr="x")
        assert check["verdict"] == "conditioning_added_value"

    def test_neutral(self):
        check = build_ablation_check({"ic": 0.03}, {"ic": 0.031}, base_expr="x")
        assert check["verdict"] == "neutral"

    def test_missing_ic(self):
        check = build_ablation_check({}, {"ic": 0.03}, base_expr="x")
        assert check["verdict"] == "unverifiable"
