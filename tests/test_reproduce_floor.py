# -*- coding: utf-8 -*-
"""复现门槛：**底线 ∧ 指标 ∧ 形态对账（三项全中）** + 阈值收口统一配置中心。

用户口径（2026-10-01，两次定调）：
1. "必须过最低底线才能进入发散环节" → 加**强制底线**（此前「指标 OR 形态」形态可单独放行，
   实测 21 次过线里 13 次仅靠形态，最极端 |IC|=0.0009）；
2. "把那个或条件改成且" → 由「底线 ∧ (指标 ∨ 形态)」改为「**底线 ∧ 指标 ∧ 形态**」：
   既要强度达标，也要事先声明的可证伪预测（强侧/形态/方向）成立。
3. 三个阈值必须进统一配置中心（`research_spec.DEFAULT_RESEARCH_SPEC['report_policy']`）。
"""
from __future__ import annotations

from alphaagent.factor.mining import research_spec as rs
from alphaagent.factor.mining.tools._dispatch import (
    _GATE_CARRIED_POLICY_KEYS,
    _reproduce_passes,
)

FLOOR, MIN_IC, MIN_ICIR = 0.010, 0.010, 0.10


def _pass(ic, icir, shape):
    return _reproduce_passes(
        ic, icir, shape, floor_ic=FLOOR, min_ic=MIN_IC, min_icir=MIN_ICIR
    )[0]


# ── ① 三项全中才通过 ─────────────────────────────────────────────────────

def test_all_three_required():
    """底线 + 指标 + 形态 confirmed 全中 → 通过（唯一通过路径）。"""
    assert _pass(0.0195, 0.2754, "confirmed") is True
    assert _pass(-0.0195, -0.2754, "confirmed") is True      # 负方向同强度，等价


def test_shape_alone_is_not_enough():
    """形态对账通过但 |ICIR| 不达线 → 必须被拦（改「且」后的核心行为变化）。"""
    assert _pass(0.0120, 0.0500, "confirmed") is False       # 底线过、形态过，指标未过
    assert _pass(0.0094, 0.1165, "confirmed") is False       # 真实案例：形态过但 |IC|<底线


def test_metrics_alone_is_not_enough():
    """指标达标但形态对账未 confirmed → 必须被拦（另一侧的核心行为变化）。"""
    assert _pass(0.0195, 0.2754, "partial") is False
    assert _pass(0.0195, 0.2754, "contradicted") is False
    assert _pass(0.0195, 0.2754, "") is False                # 未对账（缺 prediction）


def test_floor_still_blocks_extreme_noise():
    """底线仍独立生效：真实案例 rq038_ma_ratio_v2 |IC|=0.0009 形态 confirmed 也拦。"""
    assert _pass(0.0009, -0.0068, "confirmed") is False
    assert _pass(0.0099, 0.50, "confirmed") is False
    assert _pass(-0.0005, 0.30, "confirmed") is False


def test_blocked_when_nothing_matches():
    assert _pass(0.005, 0.05, "partial") is False
    assert _pass(0.005, 0.05, "contradicted") is False
    assert _pass(0.005, 0.05, "unverifiable") is False


def test_detail_exposes_hits_and_signed_values():
    ok, detail = _reproduce_passes(
        0.0195, -0.2754, "confirmed", floor_ic=FLOOR, min_ic=MIN_IC, min_icir=MIN_ICIR
    )
    assert ok is True
    assert detail["by_floor"] and detail["by_metrics"] and detail["by_shape"]
    assert detail["ic"] == 0.0195 and detail["icir"] == -0.2754      # 原始带符号值可追溯
    assert detail["abs_icir"] == 0.2754


# ── ② 配置中心：三个阈值可配、默认值合理、gate 会携带 ─────────────────────

def test_thresholds_live_in_config_center():
    rp = rs.build_run_research_spec(rs.DEFAULT_RESEARCH_SPEC)["report_policy"]
    assert rp["reproduce_floor_abs_ic"] == 0.010
    assert rp["reproduce_min_abs_ic"] == 0.010
    assert rp["reproduce_min_icir"] == 0.10


def test_thresholds_are_overridable_via_spec():
    spec = rs.build_run_research_spec({
        "report_policy": {"reproduce_floor_abs_ic": 0.02, "reproduce_min_icir": 0.25}
    })
    rp = spec["report_policy"]
    assert rp["reproduce_floor_abs_ic"] == 0.02
    assert rp["reproduce_min_icir"] == 0.25
    assert rp["reproduce_min_abs_ic"] == 0.010            # 未覆盖项回落默认


def test_gate_carries_floor_threshold():
    """判定侧唯一的可靠通道必须携带底线（否则门槛静默失效，2026-10-01 踩过）。"""
    assert "reproduce_floor_abs_ic" in _GATE_CARRIED_POLICY_KEYS
    for k in ("reproduce_min_abs_ic", "reproduce_min_icir"):
        assert k in _GATE_CARRIED_POLICY_KEYS
