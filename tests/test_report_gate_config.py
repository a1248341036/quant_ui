# -*- coding: utf-8 -*-
"""复现判定侧的 report_policy 来源（2026-10-01 修）。

缺陷背景：`_report_reproduce_judge` 用 `getattr(self, "report_policy", None)` 取配置，
但**该属性全仓从未被赋值** → 恒为 None → 复现判定读不到 report_policy。
后果（本夜实测）：保真度门槛被**静默跳过**（0 条 `report_fidelity_check` 日志），
`reproduce_min_abs_ic/icir` 也一直吃硬编码默认值。
修法：配置随 run gate（判定侧唯一可靠通道）传递，并用纯函数合并。
"""
from __future__ import annotations

from alphaagent.factor.mining.tools._dispatch import (
    _GATE_CARRIED_POLICY_KEYS,
    _effective_report_policy,
)

FID = {"enabled": True, "require_shared_field": 1}


def test_gate_carries_the_keys_judge_needs():
    """gate 必须携带判定所需键（缺失即静默跳过，正是原缺陷）。"""
    for k in ("reproduce_min_abs_ic", "reproduce_min_icir", "reproduce_fidelity",
              "factor_records_file", "question_queue_file", "inject_factor_records"):
        assert k in _GATE_CARRIED_POLICY_KEYS


def test_gate_values_override_tool_policy():
    out = _effective_report_policy({"reproduce_min_abs_ic": 0.02}, {"reproduce_min_abs_ic": 0.005})
    assert out["reproduce_min_abs_ic"] == 0.005


def test_gate_none_does_not_clobber_tool_policy():
    """gate 里为 None（未配置）的键不得覆盖工具侧已有值。"""
    out = _effective_report_policy({"reproduce_min_abs_ic": 0.02}, {"reproduce_min_abs_ic": None})
    assert out["reproduce_min_abs_ic"] == 0.02


def test_fidelity_config_survives_gate_roundtrip():
    out = _effective_report_policy(None, {"reproduce_fidelity": FID, "qid": "RQ_1", "phase": "reproduce"})
    assert out["reproduce_fidelity"] == FID
    # 与判定无关的网关键（qid/phase）不混入策略
    assert "qid" not in out and "phase" not in out


def test_empty_inputs_are_safe():
    assert _effective_report_policy(None, None) == {}
    assert _effective_report_policy({}, {"qid": "RQ_1"}) == {}
