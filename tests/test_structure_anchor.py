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


def test_declared_fields_coverage():
    sr = {"fields": ["$close", "$volume"], "min_fields": 2}
    bad = check_reproduce_fidelity(_q(sr), "CS_ZSCORE(TS_MEAN($close, 20))", records=REF)
    assert bad["passed"] is False, bad
    assert "declared_fields_hit=1<2" in bad["reason"]
    good = check_reproduce_fidelity(
        _q(sr), "CS_ZSCORE(DIVIDE(TS_MEAN($close, 20), TS_MEAN($volume, 20)))", records=REF)
    assert good["passed"] is True, good


def test_legacy_question_not_anchored():
    """旧题库没有结构信息 → 不判（行为不变）。"""
    out = check_reproduce_fidelity(_q({}), "CS_ZSCORE($close)", records=REF)
    assert out["passed"] is True
    assert out["structure_anchor"]["applicable"] is False


def test_min_fields_total_when_declared_is_incomplete():
    """巡检实测场景：声明 fields=[$close,$volume] 但 min_fields=5
    （抽取只填了"参考公式字段"）→ 仅查声明字段会让锚形同虚设，须额外要求字段总数达标。"""
    sr = {"fields": ["$close", "$volume"], "min_fields": 5}
    weak = check_reproduce_fidelity(
        _q(sr), "CS_ZSCORE(DIVIDE(TS_MEAN($close, 20), TS_MEAN($volume, 20)))", records=REF)
    assert weak["passed"] is False, weak
    assert "fields_used=2<min_fields=5" in weak["reason"]

    rich = check_reproduce_fidelity(
        _q(sr),
        "CS_ZSCORE(ADD(ADD(ADD(TS_MEAN($close, 20), $volume), $high), $low), $open)",
        records=REF)
    assert rich["passed"] is True, rich
    assert rich["structure_anchor"]["need_total_fields"] == 5
