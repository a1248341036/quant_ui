# -*- coding: utf-8 -*-
"""LLM 评估 profile 白名单回归：split=full 的交付复检口径禁止挖掘期直接调用。

背景（2026-09-10 deepseek 实测）：模型自行发现可以调
``evaluate_factor(profile_id="production_delivery")``——该口径 split=full
（train∪val∪test），test 段即锁定盲测段（2025+）。挖掘探索期按盲测数据
筛选迭代 = 多重检验烧盲测段，盲测门禁被架空（单轮实测 19 次调用）。

修复：dispatch 白名单只放行 train_screen / validation / size_neutral_validation；
全区间复检只属于 submit_factor 内部链路。
"""
from __future__ import annotations

import pytest

from alphaagent.factor.mining.tools import FactorEvalTools


class _StubService:
    def eval_train(self, request):
        return {"ok": False, "error": "stub_reached_train"}

    def eval_profile(self, request):
        return {"ok": False, "error": "stub_reached_profile"}

    def eval_val(self, request):
        return {"ok": False, "error": "stub_reached_val"}


def _tools():
    return FactorEvalTools(_StubService(), "session")


_PRED = {
    "expected_shape": "monotonic_increasing",
    "expected_strong_side": "high_factor",
    "expected_sign": 1,
}


class TestProfileWhitelist:
    def test_production_delivery_blocked(self):
        out = _tools().dispatch("evaluate_factor", {
            "profile_id": "production_delivery",
            "multi_line_expr": "TS_RANK($mgn_balance, 10)",
            "prediction": _PRED,
        })
        assert out["ok"] is False
        assert out["error_type"] == "ToolArgumentsError"
        assert "profile_not_allowed_for_mining" in out["error"]
        assert "盲测段" in out["error"]

    def test_unknown_profile_blocked_too(self):
        out = _tools().dispatch("evaluate_factor", {
            "profile_id": "some_future_profile",
            "multi_line_expr": "RANK($mgn_balance)",
            "prediction": _PRED,
        })
        assert out["ok"] is False
        assert "profile_not_allowed_for_mining" in out["error"]

    def test_train_screen_passes_through(self):
        out = _tools().dispatch("evaluate_factor", {
            "profile_id": "train_screen",
            "multi_line_expr": "RANK($mgn_balance)",
            "prediction": _PRED,
        })
        assert out.get("error") == "stub_reached_train"  # 未被白名单拦截

    def test_validation_passes_through(self):
        out = _tools().dispatch("evaluate_factor", {
            "profile_id": "validation",
            "multi_line_expr": "RANK($mgn_balance)",
            "prediction": _PRED,
        })
        assert out.get("error") == "stub_reached_profile"
