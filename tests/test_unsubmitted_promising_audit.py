# -*- coding: utf-8 -*-
"""收尾审计"已提交集合"的键名回归测试（2026-10-03）。

实测缺陷（run b7d8eee01152）：`loop.submit_record()` 把表达式存在 `multi_line_expr`
（loop.py:79），而审计只读 `expression` → 已提交集合恒空 →
① `unsubmitted_promising` 计数虚高；② 兜底补交重复提交已提交过的因子
（实测 3 个补交对象中 1 个此前已提交）。
"""
from __future__ import annotations

from alphaagent.factor.mining.agent import agentscope_run as ar


def test_reads_multi_line_expr_key():
    """核心回归：submit_record 输出的真实形态（multi_line_expr）必须被识别。"""
    rec = {"turn": 6, "factor_id": "x", "factor_name": "x",
           "multi_line_expr": "CS_ZSCORE($close)", "stored": False}
    keys = ar.submitted_expression_keys([rec])
    assert keys == {ar.canonical_hash("CS_ZSCORE($close)")}


def test_falls_back_to_expression_key():
    keys = ar.submitted_expression_keys([{"expression": "TS_MEAN($close, 5)"}])
    assert keys == {ar.canonical_hash("TS_MEAN($close, 5)")}


def test_prefers_multi_line_expr_when_both_present():
    keys = ar.submitted_expression_keys(
        [{"multi_line_expr": "A + B", "expression": "SOMETHING_ELSE"}])
    assert ar.canonical_hash("A + B") in keys


def test_ignores_blank_and_missing_and_none():
    assert ar.submitted_expression_keys([]) == set()
    assert ar.submitted_expression_keys(None) == set()
    assert ar.submitted_expression_keys([{}, {"multi_line_expr": ""},
                                         {"expression": None}]) == set()


def test_multiple_records_dedupe_by_hash():
    recs = [{"multi_line_expr": "A + B"}, {"multi_line_expr": "A + B"},
            {"multi_line_expr": "C - D"}]
    assert len(ar.submitted_expression_keys(recs)) == 2


# ── 第二层兜底：按因子名（2026-10-03 残余边界）──
# 实测 run ca6371b8bc8c：`dv_vwap_turn_sub120` 用同名迭代多个表达式、已提交 5 次，
# 但 train_passed 那条表达式哈希从未提交 → 仍被判"未提交"，兜底重复提交。

def test_submitted_factor_names_reads_name_and_id():
    recs = [{"factor_name": "dv_vwap_turn_sub120", "factor_id": "dv_vwap_turn_sub120"},
            {"factor_name": "other", "factor_id": None},
            {"factor_id": "only_id"},
            {}]
    assert ar.submitted_factor_names(recs) == {"dv_vwap_turn_sub120", "other", "only_id"}


def test_submitted_factor_names_handles_empty_and_none():
    assert ar.submitted_factor_names([]) == set()
    assert ar.submitted_factor_names(None) == set()
    assert ar.submitted_factor_names([{"factor_name": "  "}]) == set()


def test_same_name_different_expression_is_still_considered_submitted():
    """核心回归：同名因子已提交 → 即使表达式哈希不同，也应视为已提交（不重复补交）。"""
    submits = [{"factor_name": "dv_vwap_turn_sub120", "multi_line_expr": "A_OTHER_EXPR"}]
    assert ar.submitted_factor_names(submits) >= {"dv_vwap_turn_sub120"}
    # 哈希层面确实匹配不上（这正是残余边界的成因）
    assert ar.canonical_hash("THE_EVAL_EXPR") not in ar.submitted_expression_keys(submits)

