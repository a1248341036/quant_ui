# -*- coding: utf-8 -*-
"""课题 method_type 优先排序测试（P1 rotation_policy）。"""

from __future__ import annotations

from alphaagent.factor.mining.agent.question_queue import _apply_prioritize_method_types


def _rows() -> list[dict]:
    return [
        {"question_id": "a", "method_type": "ml_model"},
        {"question_id": "b", "method_type": "timing_rotation"},
        {"question_id": "c", "method_type": "rule_formula"},
        {"question_id": "b2", "method_type": "timing_rotation"},
        {"question_id": "d", "method_type": "event_driven"},
    ]


def test_priority_moves_hits_to_front_stably() -> None:
    spec = {
        "report_policy": {"prioritize_method_types": ["timing_rotation"]},
        "rotation_policy": {"enabled": True},
    }
    out = _apply_prioritize_method_types(_rows(), spec)
    assert [q["question_id"] for q in out] == ["b", "b2", "a", "c", "d"]


def test_priority_keeps_multiple_hits_relative_order() -> None:
    spec = {
        "report_policy": {"prioritize_method_types": ["event_driven", "timing_rotation"]},
        "rotation_policy": {"enabled": True},
    }
    out = _apply_prioritize_method_types(_rows(), spec)
    ids = [q["question_id"] for q in out]
    # 命中题（b/b2/d）保持原相对顺序并整体在队首
    assert ids[:3] == ["b", "b2", "d"]
    assert ids[3:] == ["a", "c"]


def test_rotation_disabled_falls_back_to_original_order() -> None:
    rows = _rows()
    spec = {
        "report_policy": {"prioritize_method_types": ["timing_rotation"]},
        "rotation_policy": {"enabled": False},
    }
    assert _apply_prioritize_method_types(rows, spec) == rows


def test_empty_priority_falls_back_to_original_order() -> None:
    rows = _rows()
    spec = {
        "report_policy": {"prioritize_method_types": []},
        "rotation_policy": {"enabled": True},
    }
    assert _apply_prioritize_method_types(rows, spec) == rows


def test_legacy_queue_without_method_type_unchanged() -> None:
    rows = [{"question_id": "x"}, {"question_id": "y"}]
    spec = {
        "report_policy": {"prioritize_method_types": ["timing_rotation"]},
        "rotation_policy": {"enabled": True},
    }
    assert _apply_prioritize_method_types(rows, spec) == rows
