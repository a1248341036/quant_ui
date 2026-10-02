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
