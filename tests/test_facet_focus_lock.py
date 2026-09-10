# -*- coding: utf-8 -*-
"""数据面聚焦硬锁定：勾选聚焦面后，越界表达式在 dispatch 层直接拦截。

覆盖：
- facet_scope_violation 纯函数（放行 / 越界 / 未触面 / 空聚焦恒放行）
- dispatch 层拦截发生在评估与提交之前（不烧评估轮次）
- 未启用聚焦时行为不变
"""
from __future__ import annotations

from alphaagent.factor.mining.memory.expressions import facet_scope_violation
from alphaagent.factor.mining.tools import FactorEvalTools


class _StubSessions:
    def get(self, _sid):
        return object()


class _StubService:
    sessions = _StubSessions()

    def eval_train(self, request):
        return {"ok": False, "error": "stub_reached"}

    def eval_profile(self, request):
        return {"ok": False, "error": "stub_reached"}

    def eval_val(self, request):
        return {"ok": False, "error": "stub_reached"}


def _tools(focus=None):
    return FactorEvalTools(_StubService(), "session", focus_facets=focus)


class TestFacetScopeViolation:
    def test_empty_focus_always_passes(self):
        assert facet_scope_violation("TS_MEAN($adj_close, 20)", None) is None
        assert facet_scope_violation("$ret_5d", ()) is None

    def test_single_focus_allowed(self):
        assert facet_scope_violation(
            "TS_RANK($mgn_balance / $mgn_buy, 20)", ["两融面"]
        ) is None

    def test_multi_focus_allowed(self):
        assert facet_scope_violation(
            "DIVERGENCE_RANK(TS_RANK($mgn_balance, 10), TS_RANK($dt_net_buy, 5))",
            ["两融面", "事件面"],
        ) is None

    def test_outside_facet_blocked(self):
        vio = facet_scope_violation("$dt_amount / $close", ["事件面"])
        assert vio is not None
        assert vio["reason"] == "outside"
        assert vio["outside_facets"] == ["价量面"]
        assert "硬锁定" in vio["message"] and "价量面" in vio["message"]

    def test_unselected_group_sibling_blocked(self):
        vio = facet_scope_violation("$dt_net_buy + $ff_main_net", ["事件面", "两融面"])
        assert vio is not None
        assert vio["outside_facets"] == ["资金面"]

    def test_no_facet_touched_blocked(self):
        vio = facet_scope_violation(
            "CS_RESIDUALIZE(RANK(TS_MEAN($float_cap, 20)), RANK($float_cap))",
            ["两融面"],
        )
        assert vio is not None
        assert vio["reason"] == "no_touch"
        assert "未触及" in vio["message"]

    def test_match_hint_lists_offending_key(self):
        """未触及勾选面时，报错要回显实际命中的列，便于直接改写。"""
        vio = facet_scope_violation("TS_DELTA($close, 5)", ["事件面"])
        assert vio is not None
        assert "$close" in vio["message"]
        assert "价量面" in vio["message"]


class TestImpliedOperatorInputs:
    """算子隐含输入面：勾筹码/拥挤/量能面时，其算子的原始输入列一并放行，
    但"必须触及勾选面"的规则仍然生效——纯价量因子不能借道混入。"""

    def test_chip_factor_with_price_inputs_allowed(self):
        assert facet_scope_violation(
            "CHIP_ENTROPY($close, $low, $high, $volume, 30, $float_cap)", ["筹码面"]
        ) is None

    def test_pure_price_factor_blocked_in_chip_run(self):
        vio = facet_scope_violation("TS_MEAN($adj_close, 20)", ["筹码面"])
        assert vio is not None
        assert vio["reason"] == "no_touch"
        assert "筹码面" in vio["message"]

    def test_volume_operator_with_price_input_allowed(self):
        assert facet_scope_violation(
            "VOLUME_CLOCK_VPIN($adj_close, $volume, 20, 50)", ["量能面"]
        ) is None

    def test_crowd_operator_with_price_input_allowed(self):
        assert facet_scope_violation("CROWD_SHARE($ret, $volume, 20)", ["拥挤面"]) is None

    def test_unrelated_face_still_blocked(self):
        vio = facet_scope_violation("$mgn_balance * $close", ["筹码面"])
        assert vio is not None
        assert vio["outside_facets"] == ["两融面"]

    def test_allowed_facets_reported(self):
        vio = facet_scope_violation("$mgn_balance", ["筹码面"])
        assert vio is not None
        assert set(vio["allowed_facets"]) == {"筹码面", "价量面", "量能面"}


class TestDispatchFacetLock:
    _PRED = {
        "expected_shape": "monotonic_increasing",
        "expected_strong_side": "high_factor",
        "expected_sign": 1,
    }

    def test_evaluate_factor_blocked_before_run(self):
        tools = _tools(["事件面"])
        out = tools.dispatch(
            "evaluate_factor",
            {
                "profile_id": "train_screen",
                "multi_line_expr": "TS_DELTA($adj_close, 5)",
                "prediction": self._PRED,
            },
        )
        assert out["ok"] is False
        assert out["error_type"] == "ToolArgumentsError"
        assert "facet_lock_violation" in out["error"]
        assert out["facet_lock"]["outside_facets"] == ["价量面"]

    def test_eval_on_train_set_blocked(self):
        tools = _tools(["业绩面"])
        out = tools.dispatch(
            "eval_on_train_set",
            {"multi_line_expr": "TS_MEAN($adj_close, 20)"},
        )
        assert out["ok"] is False
        assert out["error_type"] == "ToolArgumentsError"

    def test_eval_on_val_set_blocked(self):
        tools = _tools(["两融面"])
        out = tools.dispatch(
            "eval_on_val_set",
            {"multi_line_expr": "RANK($volume)", "expected_sign": 1},
        )
        assert out["ok"] is False
        assert "两融面" in out["error"]

    def test_submit_factor_blocked(self):
        tools = _tools(["事件面"])
        out = tools.dispatch(
            "submit_factor",
            {
                "multi_line_expr": "$dt_net_buy / $amount",
                "factor_name": "dt_flow_x_volume",
                "comment": "越界测试",
            },
        )
        assert out["ok"] is False
        assert out["error_type"] == "ToolArgumentsError"

    def test_allowed_expression_reaches_service(self):
        tools = _tools(["两融面"])
        out = tools.dispatch(
            "eval_on_train_set",
            {"multi_line_expr": "TS_RANK($mgn_balance, 10)"},
        )
        assert out["ok"] is False
        assert out.get("error") == "stub_reached"  # 未被聚焦门误拦，放行到服务层

    def test_no_focus_no_lock(self):
        tools = _tools(None)
        out = tools.dispatch(
            "eval_on_train_set",
            {"multi_line_expr": "TS_MEAN($adj_close, 20)"},
        )
        assert out.get("error") == "stub_reached"
