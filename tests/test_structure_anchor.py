# -*- coding: utf-8 -*-
"""结构硬锚回归测试（2026-10-02）。

依据：巡检发现题面要求"5 因子夏普率加权合成"，模型却自行换成"上下行波动分解"——
结构要求只在题面（软约束）。本锚做成机械校验：
  ① 声明 structure_ops → 复现须命中 ≥1 个；
  ② 否则按声明字段 → 须用到 ≥ min(声明字段数, min_fields) 个；
  ③ 无结构信息（旧题库）→ applicable=False（向后兼容）。

注：结构锚在 `check_reproduce_fidelity` 内、参照公式匹配之后执行；生产口径下
`require_factor_records=True` 保证派发的题都有参照公式，故这里给可匹配的 records。
"""
from __future__ import annotations

from alphaagent.factor.mining.agent.question_queue import check_reproduce_fidelity

SRC = "nn73/01/2018-05-25_华泰证券_金融工程_华泰单因子测试之财务质量因子.md"
# 参照公式只用通用字段（$close）→ anchor_mode=generic_only_reference，字段锚不添乱
REF = [{"source": SRC, "expr_local": "TS_MEAN($close, 20)", "formula_kind": "verbatim",
        "executable": True}]


def _q(sr: dict) -> dict:
    return {"question_id": "RQ_X", "source": "华泰证券_单因子测试之财务质量因子",
            "primary": {"spec_requirements": sr}}


def test_structure_ops_required():
    out = check_reproduce_fidelity(_q({"structure_ops": ["CS_GROUP_RANK"], "fields": ["$close"]}),
                                   "CS_ZSCORE(TS_MEAN($close, 20))", records=REF)
    assert out["passed"] is False, out
    assert "structure_ops_missing" in out["reason"]
    assert out["structure_anchor"]["mode"] == "structure_ops"


def test_structure_ops_hit_passes():
    out = check_reproduce_fidelity(
        _q({"structure_ops": ["CS_GROUP_RANK"], "fields": ["$close"]}),
        "CS_GROUP_RANK(CS_ZSCORE($close), $industry_sw_l1)", records=REF)
    assert out["passed"] is True, out
    assert out["structure_anchor"]["hit"] == ["CS_GROUP_RANK"]


def test_declared_field_hit_is_diagnostic_only():
    """回放校准（2026-10-02）：不再把"命中声明字段"当硬门——本模式允许字段同族替换，
    当硬门会误杀（回放 24 条真实复现里 19 条被拦，含 live 已过线案例）。
    min_fields ≤ 声明字段数时不应产生任何字段门。"""
    sr = {"fields": ["$close", "$volume"], "min_fields": 2}
    out = check_reproduce_fidelity(_q(sr), "CS_ZSCORE(TS_MEAN($close, 20))", records=REF)
    assert out["passed"] is True, out
    assert out["structure_anchor"]["hit_is_diagnostic_only"] is True
    assert out["structure_anchor"]["hit"] == ["close"]


def test_legacy_question_not_anchored():
    """旧题库没有结构信息 → 不判（行为不变）。"""
    out = check_reproduce_fidelity(_q({}), "CS_ZSCORE($close)", records=REF)
    assert out["passed"] is True
    assert out["structure_anchor"]["applicable"] is False


def test_min_fields_total_is_diagnostic():
    """二次回放校准：`min_fields` 是 LLM 声明值、常虚高（实测某题声明 5 而 2 字段即可表达），
    当硬门会误杀 live 已过线的合理复现 → 只作诊断（记 below_min_fields）。"""
    sr = {"fields": ["$close", "$volume"], "min_fields": 5}
    weak = check_reproduce_fidelity(
        _q(sr), "CS_ZSCORE(DIVIDE(TS_MEAN($close, 20), TS_MEAN($volume, 20)))", records=REF)
    assert weak["passed"] is True, weak
    assert weak["structure_anchor"]["below_min_fields"] is True
    assert weak["structure_anchor"]["need_total_fields"] == 4

    rich = check_reproduce_fidelity(
        _q(sr),
        "CS_ZSCORE(ADD(ADD(ADD(TS_MEAN($close, 20), $volume), $high), $low), $open)",
        records=REF)
    assert rich["passed"] is True, rich
    assert rich["structure_anchor"]["below_min_fields"] is False


def test_complexity_floor_for_weak_declarations():
    """标定实测：59% 的结构题未声明 structure_ops、声明字段常只有 1 个 → 走复杂度地板挡一行式。
    旧题库（无 has_reproducible_structure）不得被此规则影响。"""
    weak_q = dict(_q({"fields": ["$close"], "min_fields": 1}),
                  has_reproducible_structure=True)
    one = check_reproduce_fidelity(weak_q, "CS_ZSCORE($close)", records=REF)
    assert one["passed"] is False, one
    assert "ops_used=1<floor=2" in one["reason"]
    assert one["structure_anchor"]["mode"] == "complexity_floor"

    ok = check_reproduce_fidelity(
        weak_q, "CS_ZSCORE(TS_MEAN(DIVIDE($close, $volume), 20))", records=REF)
    assert ok["passed"] is True, ok


def test_legacy_question_bypasses_floor():
    """旧题库题（无 has_reproducible_structure）→ 复杂度地板不生效（向后兼容）。"""
    legacy = _q({})
    assert legacy.get("has_reproducible_structure") is None
    out = check_reproduce_fidelity(legacy, "CS_ZSCORE($close)", records=REF)
    assert out["passed"] is True, out
    assert out["structure_anchor"]["applicable"] is False
