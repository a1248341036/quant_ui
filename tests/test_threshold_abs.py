# -*- coding: utf-8 -*-
"""强度门槛符号口径统一：IC 与 ICIR 一律按 **abs** 比较（2026-10-01）。

背景：负 IC / 负 ICIR 是**正常信号**（只是方向相反，强度等价）；系统里
`prediction` 形态/方向对账、train↔val 方向一致性、正式库 `abs_gte` 门槛
三处本就"方向无关"。但多处强度判定出现 `abs(ic)` + **带符号** `icir` 混用：
- 复现判定 `_dispatch._report_reproduce_judge`（实测误杀 3 次：
  |IC| 0.0125~0.0225、|ICIR| 0.104~0.234，例 rq045_vwap_bias_to_sz_extreme）
- 海选线判定 `_dispatch`（auto-val 触发用）
- near-miss 判定 `_dispatch`
- 记忆层 verdict 判定 `memory/schema._classify` 与 `memory/diagnostics._classify_train`
本组测试锁定"两侧都 abs"的口径。
"""
from __future__ import annotations

from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from alphaagent.factor.mining.tools._dispatch import _reproduce_strength

_classify_memory = ResearchMemoryStore._classify

MIN_IC, MIN_ICIR = 0.010, 0.10


def _ok(ic, icir):
    return _reproduce_strength(ic, icir, min_ic=MIN_IC, min_icir=MIN_ICIR)[0]


def test_positive_direction_passes():
    assert _ok(0.030, 0.350) is True


def test_negative_direction_passes_too():
    """负 IC / 负 ICIR 是正常方向，强度等价 → 必须与正向同样通过。"""
    assert _ok(-0.030, -0.350) is True


def test_real_miskilled_case_now_passes():
    """实测被误杀的那例：|IC|=0.0225、|ICIR|=0.2335（真实值皆为负）。"""
    assert _ok(-0.0225, -0.2335) is True


def test_weak_stays_weak_regardless_of_sign():
    assert _ok(0.006, 0.080) is False
    assert _ok(-0.006, -0.080) is False
    # 强度不足时符号无关：IC 够但 |ICIR| 不够 → 仍不达标
    assert _ok(-0.030, -0.050) is False


def test_strength_result_keeps_signed_values():
    """返回值要保留原始带符号值（供日志/父本体检展示，避免"日志符号矛盾"）。"""
    _, detail = _reproduce_strength(-0.0225, -0.2335, min_ic=MIN_IC, min_icir=MIN_ICIR)
    assert detail["ic"] == -0.0225 and detail["icir"] == -0.2335
    assert detail["abs_ic"] == 0.0225 and detail["abs_icir"] == 0.2335


# ── 记忆层 verdict：负方向强信号应按强度记账，不得因方向降级 ---------------

def test_memory_classify_accepts_negative_direction_strength():
    verdict, _ = _classify_memory(
        "eval_on_train_set",
        {"ok": True, "split": "train"},
        {"ic": -0.030, "icir": -0.35, "factor_coverage": 0.95},
        "",
    )
    assert verdict == "train_passed", f"负方向强信号应过海选线，实际 {verdict}"


def test_memory_classify_still_weak_when_magnitude_insufficient():
    verdict, _ = _classify_memory(
        "eval_on_train_set",
        {"ok": True, "split": "train"},
        {"ic": -0.006, "icir": -0.08, "factor_coverage": 0.95},
        "",
    )
    assert verdict == "weak"


def test_memory_classify_near_miss_accepts_negative_direction():
    """near-miss 分支同样按 abs：|IC| 落在门槛 80%~100% 且 |ICIR| 达标即 near_miss。"""
    verdict, _ = _classify_memory(
        "eval_on_train_set",
        {"ok": True, "split": "train"},
        {"ic": -0.017, "icir": -0.25, "factor_coverage": 0.95},
        "",
    )
    assert verdict == "near_miss", f"实际 {verdict}"
