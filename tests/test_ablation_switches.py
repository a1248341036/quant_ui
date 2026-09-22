# -*- coding: utf-8 -*-
"""消融开关（cognition_policy / operator_policy / memory_policy.enabled）单元测试。

覆盖三项 P0 消融开关：
  C1  prediction_check_enabled=False  → 评估不注入 prediction_check、软门关闭
  C2  ablation_check_enabled=False    → 门控表达式不注入 ablation_check/ablation_hint
  D2  operator_policy.blacklist       → 命中算子在 eval/submit 前拦截、算子目录同步隐藏
  A1  memory_policy.enabled=False     → normalize 校验默认值

复用 test_prediction_reconciliation 的 _ProfileServiceWithDeciles / _PRED_OK 桩。
"""
from __future__ import annotations

import pytest

from alphaagent.factor.mining.tools import FactorEvalTools
from alphaagent.factor.mining.tools._dispatch import _PREDICTION_SOFT_LIMIT
from alphaagent.factor.mining.research_spec import normalize_research_spec, DEFAULT_RESEARCH_SPEC

# ── 复用已有测试桩 ──────────────────────────────────────────────
from test_prediction_reconciliation import _ProfileServiceWithDeciles, _PRED_OK

_GATED = "GATED_SIGNAL(TS_MEAN($adj_close, 20), RANK($volume), 0.8, true, 0)"
_PLAIN = "TS_MEAN($adj_close, 20)"


# ═════════════════════════════════════════════════════════════════
# C1: prediction_check_enabled=False
# ═════════════════════════════════════════════════════════════════
class TestPredictionCheckDisabled:
    """关闭预测对账后：不注入 prediction_check、缺失 prediction 不记账不拦截。"""

    def _tools(self):
        # 本测试只验证 prediction 软门（C1 消融），不关心同质化熔断器：
        # 循环 4 次同一表达式会触发 HomogenizationSmoothingBlock（2026-09-19
        # 引入），与 prediction 缺失软门无关，故显式禁用。
        return FactorEvalTools(
            _ProfileServiceWithDeciles(),
            "session",
            cognition_policy={"prediction_check_enabled": False},
            homogenization_policy={"enabled": False},
        )

    def test_evaluate_factor_no_prediction_check(self):
        tools = self._tools()
        result = tools.dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": _PLAIN,
            "prediction": _PRED_OK,
        })
        assert result["ok"]
        assert "prediction_check" not in result
        assert "prediction_warning" not in result

    def test_evaluate_factor_missing_prediction_no_warning(self):
        """不传 prediction 也不附 warning、不累计计数。"""
        tools = self._tools()
        for _ in range(_PREDICTION_SOFT_LIMIT + 1):
            result = tools.dispatch("evaluate_factor", {
                "profile_id": "train_screen",
                "multi_line_expr": _PLAIN,
            })
            assert result["ok"]
            assert "prediction_warning" not in result

    def test_evaluate_factor_invalid_prediction_not_blocked(self):
        """携带非法 prediction 也不拦截（校验整体短路）。"""
        from alphaagent.factor.mining.tools._dispatch import _prediction_argument_error
        tools = self._tools()
        err = _prediction_argument_error(
            {"multi_line_expr": "x", "prediction": {"expected_shape": "bogus"}},
            enabled=False,
        )
        assert err is None

    def test_eval_on_train_set_no_prediction_check(self):
        tools = self._tools()
        result = tools.dispatch("eval_on_train_set", {
            "multi_line_expr": _PLAIN,
            "prediction": _PRED_OK,
        })
        assert result["ok"]
        assert "prediction_check" not in result

    def test_eval_on_val_set_no_prediction_check(self):
        tools = self._tools()
        tools.service.eval_val = lambda req: {
            "ok": True,
            "metrics": {"cross_sectional_core": {"ic": 0.03, "icir": 0.4, "decile_mean_label": tools.service.decile_rows}},
            "summary": {"ic": 0.03},
        }
        result = tools.dispatch("eval_on_val_set", {
            "multi_line_expr": _PLAIN,
            "prediction": _PRED_OK,
        })
        assert result["ok"]
        assert "prediction_check" not in result


# ═════════════════════════════════════════════════════════════════
# C2: ablation_check_enabled=False
# ═════════════════════════════════════════════════════════════════
class TestAblationCheckDisabled:
    """关门控自动消融后：门控表达式不注入 ablation_check / ablation_hint。"""

    def _tools(self):
        return FactorEvalTools(
            _ProfileServiceWithDeciles(),
            "session",
            cognition_policy={"ablation_check_enabled": False},
        )

    def test_gated_with_base_expr_no_ablation_check(self):
        tools = self._tools()
        result = tools.dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": _GATED,
            "prediction": _PRED_OK,
            "interaction": {
                "interaction_type": "gated_signal",
                "base_signal": "趋势",
                "condition_signal": "量",
                "economic_mechanism": "放量趋势延续性更强",
                "base_expr": "TS_MEAN($adj_close, 20)",
            },
        })
        assert result["ok"]
        assert "ablation_check" not in result
        assert "ablation_hint" not in result

    def test_gated_without_base_expr_no_hint(self):
        tools = self._tools()
        result = tools.dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": _GATED,
            "prediction": _PRED_OK,
        })
        assert result["ok"]
        assert "ablation_hint" not in result
        assert "ablation_check" not in result

    def test_eval_on_train_set_gated_no_ablation(self):
        tools = self._tools()
        result = tools.dispatch("eval_on_train_set", {
            "multi_line_expr": _GATED,
            "prediction": _PRED_OK,
            "interaction": {
                "interaction_type": "gated_signal",
                "base_expr": "TS_MEAN($adj_close, 20)",
            },
        })
        assert result["ok"]
        assert "ablation_check" not in result


# ═════════════════════════════════════════════════════════════════
# D2: operator_policy.blacklist
# ═════════════════════════════════════════════════════════════════
class TestOperatorBlacklist:
    """算子黑名单：命中在 eval/submit 前拦截并引导；未命中放行；默认空清单不拦截。"""

    def _tools(self, blacklist=None, **kw):
        kw.setdefault("cognition_policy", {"prediction_check_enabled": False, "ablation_check_enabled": False})
        return FactorEvalTools(
            _ProfileServiceWithDeciles(),
            "session",
            operator_policy={"blacklist": blacklist or []},
            **kw,
        )

    # ── 拦截 ──
    def test_evaluate_factor_blocked(self):
        tools = self._tools(["GATED_SIGNAL"])
        r = tools.dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": _GATED,
        })
        assert not r["ok"]
        assert r["error_type"] == "OperatorBlacklistBlock"
        assert r["blocked_operators"] == ["GATED_SIGNAL"]
        assert "基础" in r["error"] or "TS_MEAN" in r["error"]

    def test_eval_on_train_set_blocked(self):
        tools = self._tools(["GATED_SIGNAL"])
        r = tools.dispatch("eval_on_train_set", {"multi_line_expr": _GATED})
        assert not r["ok"]
        assert r["error_type"] == "OperatorBlacklistBlock"

    def test_eval_on_val_set_blocked(self):
        tools = self._tools(["GATED_SIGNAL"])
        tools.service.eval_val = lambda req: {"ok": True, "summary": {"ic": 0.03}}
        r = tools.dispatch("eval_on_val_set", {"multi_line_expr": _GATED})
        assert not r["ok"]
        assert r["error_type"] == "OperatorBlacklistBlock"

    def test_submit_factor_blocked(self):
        tools = self._tools(["GATED_SIGNAL"])
        tools.submit_service = type("S", (), {"submit": staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("reached")))})()
        r = tools.dispatch("submit_factor", {
            "multi_line_expr": _GATED,
            "factor_name": "f1",
            "comment": "c",
        })
        assert not r["ok"]
        assert r["error_type"] == "OperatorBlacklistBlock"

    # ── 多算子 ──
    def test_multiple_blacklisted_ops_all_listed(self):
        tools = self._tools(["GATED_SIGNAL", "CS_GROUP_RANK"])
        r = tools.dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": "CS_GROUP_RANK(GATED_SIGNAL($adj_close, $volume, 0.8, true, 0), industry)",
        })
        assert not r["ok"]
        assert set(r["blocked_operators"]) == {"GATED_SIGNAL", "CS_GROUP_RANK"}

    # ── 未命中放行 ──
    def test_non_blacklisted_passes(self):
        tools = self._tools(["GATED_SIGNAL"])
        r = tools.dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": _PLAIN,
        })
        assert r["ok"]

    # ── 默认空清单 ──
    def test_empty_blacklist_no_block(self):
        tools = FactorEvalTools(_ProfileServiceWithDeciles(), "session")  # 默认无黑名单
        r = tools.dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": _GATED,
            "prediction": _PRED_OK,
        })
        assert r["ok"]  # 默认不拦截门控表达式

    # ── 大小写归一 ──
    def test_blacklist_case_insensitive(self):
        """配置用小写、表达式用大写仍命中。"""
        tools = FactorEvalTools(
            _ProfileServiceWithDeciles(),
            "session",
            operator_policy={"blacklist": ["gated_signal"]},
        )
        r = tools.dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": _GATED,
        })
        assert not r["ok"]
        assert r["blocked_operators"] == ["GATED_SIGNAL"]


# ═════════════════════════════════════════════════════════════════
# research_spec normalize 校验
# ═════════════════════════════════════════════════════════════════
class TestNormalizeAblationPolicies:
    """normalize_research_spec 对消融开关的默认值填充与校验。"""

    def test_cognition_policy_defaults(self):
        spec = normalize_research_spec({})
        assert spec["cognition_policy"]["prediction_check_enabled"] is True
        assert spec["cognition_policy"]["ablation_check_enabled"] is True

    def test_cognition_policy_override(self):
        spec = normalize_research_spec({"cognition_policy": {"prediction_check_enabled": False}})
        assert spec["cognition_policy"]["prediction_check_enabled"] is False
        assert spec["cognition_policy"]["ablation_check_enabled"] is True  # 未改的跟默认

    def test_operator_policy_defaults(self):
        spec = normalize_research_spec({})
        assert spec["operator_policy"]["blacklist"] == []

    def test_operator_policy_blacklist_uppercased(self):
        spec = normalize_research_spec({"operator_policy": {"blacklist": ["gated_signal", "cs_group_rank"]}})
        assert spec["operator_policy"]["blacklist"] == ["GATED_SIGNAL", "CS_GROUP_RANK"]

    def test_operator_policy_unknown_op_rejected(self):
        with pytest.raises(ValueError, match="unknown_operators"):
            normalize_research_spec({"operator_policy": {"blacklist": ["NONEXISTENT_OP"]}})

    def test_memory_policy_enabled_default(self):
        spec = normalize_research_spec({})
        assert spec["memory_policy"]["enabled"] is True

    def test_memory_policy_disabled(self):
        spec = normalize_research_spec({"memory_policy": {"enabled": False}})
        assert spec["memory_policy"]["enabled"] is False


# ═════════════════════════════════════════════════════════════════
# Prompt 算子目录黑名单过滤
# ═════════════════════════════════════════════════════════════════
class TestPromptOperatorCatalogBlacklist:
    """黑名单算子在 prompt 算子目录中隐藏 + 出现禁用提醒；默认无过滤。"""

    def test_blacklisted_op_hidden_with_warning(self):
        from alphaagent.factor.mining.prompt.prompts import build_system_prompt
        text = build_system_prompt(
            research_spec={"operator_policy": {"blacklist": ["GATED_SIGNAL"]}},
            include_operator_catalog=True,
            panel_columns=["adj_close"],
        )
        # 目录签名行不含被禁算子
        catalog_section = text[text.find("### 可用算子"):]
        assert "- `GATED_SIGNAL" not in catalog_section
        # 出现禁用提醒
        assert "已禁用" in text

    def test_no_blacklist_no_warning(self):
        from alphaagent.factor.mining.prompt.prompts import build_system_prompt
        text = build_system_prompt(
            research_spec=None,
            include_operator_catalog=True,
            panel_columns=["adj_close"],
        )
        assert "GATED_SIGNAL(" in text   # 算子可见
        assert "已禁用" not in text       # 无提醒

    def test_default_spec_no_warning(self):
        """传入 DEFAULT_RESEARCH_SPEC（空黑名单）也不出提醒。"""
        from alphaagent.factor.mining.prompt.prompts import build_system_prompt
        spec = normalize_research_spec({})
        text = build_system_prompt(
            research_spec=spec,
            include_operator_catalog=True,
            panel_columns=["adj_close"],
        )
        assert "已禁用" not in text


# ═════════════════════════════════════════════════════════════════
# research_policy_prompt 消融摘要注入
# ═════════════════════════════════════════════════════════════════
class TestPolicySummaryInjection:
    """cognition_policy_summary / operator_policy_summary 在 research_policy_prompt 中的注入。"""

    def test_cognition_summary_when_disabled(self):
        from alphaagent.factor.mining.research_spec import cognition_policy_summary
        spec = normalize_research_spec({"cognition_policy": {"prediction_check_enabled": False, "ablation_check_enabled": False}})
        summary = cognition_policy_summary(spec)
        assert "关闭预测对账" in summary
        assert "关闭门控自动消融" in summary

    def test_cognition_summary_empty_when_default(self):
        from alphaagent.factor.mining.research_spec import cognition_policy_summary
        spec = normalize_research_spec({})
        assert cognition_policy_summary(spec) == ""

    def test_operator_summary_when_blacklist(self):
        from alphaagent.factor.mining.research_spec import operator_policy_summary
        spec = normalize_research_spec({"operator_policy": {"blacklist": ["GATED_SIGNAL"]}})
        summary = operator_policy_summary(spec)
        assert "GATED_SIGNAL" in summary
        assert "禁用" in summary

    def test_operator_summary_empty_when_default(self):
        from alphaagent.factor.mining.research_spec import operator_policy_summary
        spec = normalize_research_spec({})
        assert operator_policy_summary(spec) == ""
