# -*- coding: utf-8 -*-
"""复现门槛：**底线 ∧ (指标通道 ∨ 形态通道)** + 三个阈值收口到统一配置中心（2026-10-01）。

用户口径（2026-10-01）：
- "必须过最低底线才能进入发散环节" —— 此前判据是「指标达标 OR 形态 confirmed」，
  形态可**单独**放行，实测 21 次过线里 13 次仅靠形态（最极端 |IC|=0.0009）；
- 三个阈值必须进统一配置中心（`research_spec.DEFAULT_RESEARCH_SPEC['report_policy']`），
  不留在业务逻辑里硬编码。
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


# ── ① 底线强制：形态对账不能绕过底线 ──────────────────────────────────────

def test_shape_channel_cannot_bypass_floor():
    """复核真实案例 rq038_ma_ratio_v2：|IC|=0.0009 形态 confirmed → 现在必须被拦。"""
    assert _pass(0.0009, -0.0068, "confirmed") is False
    assert _pass(0.0099, 0.50, "confirmed") is False      # 差一点点也不行（严格底线）
    assert _pass(-0.0005, 0.30, "confirmed") is False


def test_shape_channel_passes_once_above_floor():
    """过底线 + 形态 confirmed → 过（无需 ICIR 达线，这是形态通道的正当用途）。"""
    assert _pass(0.0120, 0.05, "confirmed") is True
    assert _pass(-0.0120, -0.05, "confirmed") is True     # 负方向同样适用（强度按 abs）


def test_metrics_channel_passes_without_shape():
    """指标通道独立成立（形态 partial 也行），但同样受底线约束。"""
    assert _pass(0.0195, 0.2754, "partial") is True
    assert _pass(0.0095, 0.2754, "partial") is False      # 底线未过（|IC|<floor）
    assert _pass(0.0200, 0.0500, "partial") is False      # 底线过但 |ICIR| 不足


def test_blocked_when_neither_channel_nor_floor():
    assert _pass(0.005, 0.05, "partial") is False
    assert _pass(0.005, 0.05, "contradicted") is False
    assert _pass(0.005, 0.05, "unverifiable") is False


def test_detail_exposes_channel_hits_and_signed_values():
    ok, detail = _reproduce_passes(
        -0.0120, -0.05, "confirmed", floor_ic=FLOOR, min_ic=MIN_IC, min_icir=MIN_ICIR
    )
    assert ok is True
    assert detail["by_floor"] is True and detail["by_shape"] is True
    assert detail["by_metrics"] is False                  # |ICIR| 未达线，但形态通道已放行
    assert detail["ic"] == -0.0120 and detail["abs_ic"] == 0.0120   # 原始值可追溯


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
