# -*- coding: utf-8 -*-
"""用户显式调仓频率 = 硬约束回归。

背景（2026-09-11 day_a run 实测）：spec engine_gate.freq 要求 weekly，但
allowed_freqs 含三档，submit 侧优先级 `LLM 传值 > 用户设置`——模型对 20d
label 的基本面因子声明 daily 并入库，研究总结展示"调仓一天"。
修复：用户显式选择时 allowed_freqs 收窄为单值；自动（未选）保留三档自由度。
"""
from __future__ import annotations

import copy

from backend.routers.alphaagent import apply_user_rebalance_freq


def _spec(freq="weekly", allowed=("daily", "weekly", "monthly")):
    return {
        "delivery_policy": {
            "production": {
                "engine_gate": {"enabled": True, "freq": freq, "allowed_freqs": list(allowed)},
            }
        }
    }


class TestUserRebalanceFreqHardConstraint:
    def test_explicit_choice_overrides_freq_and_whitelist(self):
        spec = _spec()
        apply_user_rebalance_freq(spec, "monthly")
        eg = spec["delivery_policy"]["production"]["engine_gate"]
        assert eg["freq"] == "monthly"
        assert eg["allowed_freqs"] == ["monthly"]

    def test_explicit_daily_locks_out_llm_daily_to_weekly_drift(self):
        """实测场景：用户选 weekly 后模型传 daily → submit 白名单直接拒绝。"""
        spec = _spec()
        apply_user_rebalance_freq(spec, "weekly")
        eg = spec["delivery_policy"]["production"]["engine_gate"]
        chosen = "daily"  # 模型声明
        assert chosen not in eg["allowed_freqs"]

    def test_auto_keeps_full_whitelist(self):
        spec = _spec()
        before = copy.deepcopy(spec)
        apply_user_rebalance_freq(spec, None)
        apply_user_rebalance_freq(spec, "")
        assert spec == before  # 未选择时不做任何收窄

    def test_creates_missing_engine_gate_block(self):
        spec = {"delivery_policy": {"production": {}}}
        apply_user_rebalance_freq(spec, "monthly")
        eg = spec["delivery_policy"]["production"]["engine_gate"]
        assert eg["freq"] == "monthly" and eg["allowed_freqs"] == ["monthly"]

    def test_submit_side_precedence_documented(self):
        """submit.py: chosen_freq = LLM 传值 > spec freq；白名单收窄后
        用户设置即成为唯一合法值（LLM 传别的值会被 invalid_rebalance_freq 拒绝）。"""
        engine_cfg = {"freq": "weekly", "allowed_freqs": ["weekly"]}
        llm_declared = "daily"
        chosen_freq = str(llm_declared or engine_cfg.get("freq") or "daily").lower()
        assert chosen_freq not in engine_cfg["allowed_freqs"]  # 被 359 行拒绝
