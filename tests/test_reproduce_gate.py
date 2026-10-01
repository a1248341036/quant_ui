# -*- coding: utf-8 -*-
"""复现门槛：**指标 ∧ 形态要求** + 阈值全部收口统一配置中心（2026-10-01 定稿）。

演进轨迹（用户三次定调）：
1. "必须过最低底线" → 加底线、形态由 OR 转必要条件；
2. "把那个或条件改成且" → 底线 ∧ 指标 ∧ 形态 confirmed；
3. "冗余的删除" + 形态对账 ≈ IC 的另一种说法 → **删底线**（与 min_abs_ic 同值属冗余）、
   形态项由硬编码 `confirmed` 改为**可配档位**，默认 `strong_side`（保留"多头端是否最强"
   这一独立于 IC 的经济含义，丢掉"曲线字符串完全相等"的审美严格）。
"""
from __future__ import annotations

import pytest

from alphaagent.factor.mining import research_spec as rs
from alphaagent.factor.mining.tools._dispatch import (
    _GATE_CARRIED_POLICY_KEYS,
    _reproduce_passes,
)

MIN_IC, MIN_ICIR = 0.010, 0.10


def _pass(ic, icir, shape="partial", side_match=True, req="strong_side"):
    return _reproduce_passes(
        ic, icir, shape, side_match,
        min_ic=MIN_IC, min_icir=MIN_ICIR, shape_requirement=req,
    )[0]


# ── ① 指标 ∧ 形态要求 ────────────────────────────────────────────────────

def test_metrics_and_strong_side_pass():
    """默认档：指标达标 + 强侧匹配 → 通过（形态字符串是否完全一致不影响）。"""
    assert _pass(0.0195, 0.2754) is True                     # partial 也算过（降级要点）
    assert _pass(0.0120, 0.1100) is True
    assert _pass(-0.0195, -0.2754) is True                   # 负方向同强度等价


def test_strong_side_mismatch_is_blocked():
    """强侧不匹配 → 拦（IC 看不见的可交易性维度：收益集中在中间组，多空赚不到钱）。"""
    assert _pass(0.0195, 0.2754, side_match=False) is False
    assert _pass(0.0331, 0.3840, side_match=False) is False


def test_metrics_below_line_is_blocked():
    assert _pass(0.0094, 0.1165) is False                    # |IC| 不足
    assert _pass(0.0200, 0.0500) is False                    # |ICIR| 不足
    assert _pass(0.0009, -0.0068, shape="confirmed") is False  # 真实案例：噪声母本


def test_confirmed_requirement_is_stricter():
    """confirmed 档：只有形态完全一致才过（partial 即使强侧匹配也拦）。"""
    assert _pass(0.0195, 0.2754, shape="confirmed", req="confirmed") is True
    assert _pass(0.0195, 0.2754, shape="partial", req="confirmed") is False
    assert _pass(0.0195, 0.2754, shape="contradicted", req="confirmed") is False


def test_off_requirement_ignores_shape():
    assert _pass(0.0195, 0.2754, shape="contradicted", side_match=False, req="off") is True
    assert _pass(0.0050, 0.0500, shape="confirmed", req="off") is False   # 指标仍必须过


def test_detail_exposes_hits_and_signed_values():
    ok, detail = _reproduce_passes(
        0.0195, -0.2754, "partial", True,
        min_ic=MIN_IC, min_icir=MIN_ICIR, shape_requirement="strong_side",
    )
    assert ok is True
    assert detail["by_metrics"] and detail["by_shape"] and detail["strong_side_match"]
    assert detail["ic"] == 0.0195 and detail["icir"] == -0.2754      # 原始带符号值可追溯
    assert detail["abs_icir"] == 0.2754
    assert detail["shape_requirement"] == "strong_side"
    assert "by_floor" not in detail and "floor_ic" not in detail      # 底线已删净


# ── ② 配置中心：三键可配 + 校验 + gate 携带 ───────────────────────────────

def test_defaults_live_in_config_center():
    rp = rs.build_run_research_spec(rs.DEFAULT_RESEARCH_SPEC)["report_policy"]
    assert rp["reproduce_min_abs_ic"] == 0.010
    assert rp["reproduce_min_icir"] == 0.10
    assert rp["reproduce_shape_requirement"] == "strong_side"
    assert "reproduce_floor_abs_ic" not in rp          # 冗余键已删


def test_overridable_via_spec():
    rp = rs.build_run_research_spec({
        "report_policy": {"reproduce_min_icir": 0.25, "reproduce_shape_requirement": "confirmed"}
    })["report_policy"]
    assert rp["reproduce_min_icir"] == 0.25
    assert rp["reproduce_shape_requirement"] == "confirmed"
    assert rp["reproduce_min_abs_ic"] == 0.010          # 未覆盖项回落默认


def test_invalid_shape_requirement_is_rejected():
    with pytest.raises(ValueError, match="reproduce_shape_requirement"):
        rs.build_run_research_spec({"report_policy": {"reproduce_shape_requirement": "whatever"}})


def test_gate_carries_current_keys_only():
    """判定侧唯一可靠通道必须携带全部生效键，且**不再**携带已删的底线键。"""
    for k in ("reproduce_min_abs_ic", "reproduce_min_icir", "reproduce_shape_requirement"):
        assert k in _GATE_CARRIED_POLICY_KEYS
    assert "reproduce_floor_abs_ic" not in _GATE_CARRIED_POLICY_KEYS
