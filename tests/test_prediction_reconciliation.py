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


class TestPredictionAliasAndErrorDetail:
    """2026-09-05：别名归一（"D10"→high_factor）+ 错误信息逐字段回显。

    实测一次 run 7 次调用因 expected_strong_side="D10" 全部被拒，错误信息
    不带实际值导致 LLM 盲猜重试。
    """

    def test_side_alias_decile(self):
        from alphaagent.factor.mining.eval.prediction import normalize_prediction

        pred = normalize_prediction({
            "expected_shape": "monotonic_increasing",
            "expected_strong_side": "D10",
            "expected_sign": 1,
        })
        assert pred is not None
        assert pred["expected_strong_side"] == "high_factor"
        # D1/D4/Q 别名
        assert normalize_prediction({
            "expected_shape": "u_shape", "expected_strong_side": "d1", "expected_sign": -1,
        })["expected_strong_side"] == "low_factor"
        assert normalize_prediction({
            "expected_shape": "inverted_u", "expected_strong_side": "Q5", "expected_sign": 1,
        })["expected_strong_side"] == "middle"

    def test_shape_alias_variants(self):
        from alphaagent.factor.mining.eval.prediction import normalize_prediction

        for raw in ("u-shape", "u shape", "u_shape"):
            pred = normalize_prediction({
                "expected_shape": raw, "expected_strong_side": "high_factor", "expected_sign": 1,
            })
            assert pred is not None and pred["expected_shape"] == "u_shape"

    def test_sign_alias(self):
        from alphaagent.factor.mining.eval.prediction import normalize_prediction

        for raw in ("1", "+1", "positive", 1.0):
            pred = normalize_prediction({
                "expected_shape": "irregular", "expected_strong_side": "high", "expected_sign": raw,
            })
            assert pred is not None and pred["expected_sign"] == 1
        # bool 不是合法 sign
        assert normalize_prediction({
            "expected_shape": "irregular", "expected_strong_side": "high", "expected_sign": True,
        }) is None

    def test_truly_invalid_returns_none(self):
        from alphaagent.factor.mining.eval.prediction import normalize_prediction

        assert normalize_prediction({
            "expected_shape": "monotonic_increasing",
            "expected_strong_side": "strongest",  # 无法归一
            "expected_sign": 1,
        }) is None

    def test_dispatch_error_echoes_actual_value(self):
        """真正非法时错误信息必须包含实际收到的值。"""
        from alphaagent.factor.mining.tools._dispatch import _prediction_argument_error

        err = _prediction_argument_error({
            "multi_line_expr": "x",
            "prediction": {
                "expected_shape": "monotonic_increasing",
                "expected_strong_side": "strongest",
                "expected_sign": 1,
            },
        })
        assert err is not None
        assert "expected_strong_side" in err["error"]
        assert "'strongest'" in err["error"]
        assert "D8-D10" in err["error"]

    def test_dispatch_passes_alias(self):
        """别名纠正后直接放行（此前 "D10" 被拒）。"""
        from alphaagent.factor.mining.tools._dispatch import _prediction_argument_error

        err = _prediction_argument_error({
            "multi_line_expr": "x",
            "prediction": {
                "expected_shape": "monotonic_increasing",
                "expected_strong_side": "D10",
                "expected_sign": 1,
            },
        })
        assert err is None


class TestProsePrediction:
    """2026-09-06：散文→枚举归一。

    实测一次 run 并行 batch 11 条评估 8 条 expected_shape 写成整句机制
    （"供给收缩(D10)最优，供给扩张(D1)最差"）全部被拒——错误反馈要下一轮
    才到，一轮全灭 25 分钟。确定性关键词推断：梯度词全局计票 + 端点评级词
    挂靠前方最近端点。
    """

    # (expected_shape 散文, 期望 shape, 期望 side)——全部取自实测失败样例
    _PROSE_CASES = [
        ("D1→D10 递增，行业内集中度越高未来收益越高", "monotonic_increasing", "high_factor"),
        ("启用组(低换手)内集中度有效，D10最优；未启用组为0导致分布退化", "monotonic_increasing", "high_factor"),
        ("供给收缩(D10)最优，供给扩张(D1)最差，近似单调", "monotonic_increasing", "high_factor"),
        ("D1(规模调整后户数最少=机构型筹码)最优，D10(散户拥挤)最差", "monotonic_decreasing", "low_factor"),
        ("D1→D10递增，顶部集中(顶住供给扩容的吸筹)", "monotonic_increasing", "high_factor"),
        ("供给收缩桶内集中度秩高的组未来收益最高", "monotonic_increasing", "high_factor"),
        ("散户基数高的桶内集中度信号更强，组间IC有梯度", "monotonic_increasing", "high_factor"),
        ("D10(潜伏吸筹)最优，D1(集中已兑现过热)最差", "monotonic_increasing", "high_factor"),
    ]

    def test_prose_shape_and_side_resolved(self):
        from alphaagent.factor.mining.eval.prediction import normalize_prediction

        for prose, want_shape, want_side in self._PROSE_CASES:
            pred = normalize_prediction({
                "expected_shape": prose,
                "expected_strong_side": prose,
                "expected_sign": 1,
            })
            assert pred is not None, f"散文应归一成功: {prose}"
            assert pred["expected_shape"] == want_shape, prose
            assert pred["expected_strong_side"] == want_side, prose

    def test_prose_u_shape_and_middle(self):
        from alphaagent.factor.mining.eval.prediction import normalize_prediction

        pred = normalize_prediction({
            "expected_shape": "倒U型：中间组最强",
            "expected_strong_side": "中间",
            "expected_sign": 1,
        })
        assert pred is not None
        assert pred["expected_shape"] == "inverted_u"
        assert pred["expected_strong_side"] == "middle"

    def test_prose_sign(self):
        from alphaagent.factor.mining.eval.prediction import normalize_prediction

        assert normalize_prediction({
            "expected_shape": "irregular", "expected_strong_side": "high_factor",
            "expected_sign": "看多",
        })["expected_sign"] == 1
        assert normalize_prediction({
            "expected_shape": "irregular", "expected_strong_side": "high_factor",
            "expected_sign": "看空，预期负向",
        })["expected_sign"] == -1

    def test_prose_ambiguous_still_rejected(self):
        """无方向词/两端票数平的散文仍拒绝并回显合法值。"""
        from alphaagent.factor.mining.eval.prediction import (
            describe_prediction_issues,
            normalize_prediction,
        )

        assert normalize_prediction({
            "expected_shape": "随机描述没有方向词",
            "expected_strong_side": "随便",
            "expected_sign": 1,
        }) is None
        # 两端评级矛盾（D10 最优 + D1 最优）→ 平票拒绝
        assert normalize_prediction({
            "expected_shape": "D10最优，D1也最优，无法区分",
            "expected_strong_side": "D10",
            "expected_sign": 1,
        }) is None
        issues = describe_prediction_issues({
            "expected_shape": "随机描述没有方向词",
            "expected_strong_side": "随便",
            "expected_sign": 1,
        })
        assert issues and "monotonic_increasing" in issues

    def test_enum_and_alias_unaffected(self):
        """合法枚举/别名优先于散文推断，行为不变。"""
        from alphaagent.factor.mining.eval.prediction import normalize_prediction

        pred = normalize_prediction({
            "expected_shape": "inverted_u", "expected_strong_side": "Q5", "expected_sign": 1,
        })
        assert pred["expected_shape"] == "inverted_u"
        assert pred["expected_strong_side"] == "middle"
