# -*- coding: utf-8 -*-
"""结构准入 + 题面 B 版（2026-10-02，仅研报模式）回归测试。

背景（26707 次历史评估实测）：单算子因子达双门槛率 1.4%、单字段 0.1%，
而含结构算子 9.1%、3+ 字段 11.2%（差 6~112 倍）→ 复现"一条通用单公式"必然无效。
故：准入只派发**报告提出了可复现结构**的课题；题面以**结构目标**为主体。
"""
from __future__ import annotations

import json

from alphaagent.factor.mining import research_spec as rs
from alphaagent.factor.mining.agent.question_queue import load_question_queue

Q_STRUCT = {
    "question_id": "RQ_S1", "source": "某报告", "topic": "结构题",
    "report_structure": "composite", "has_reproducible_structure": True,
    "reproduction_target": "GMSD 前 2/3 → SIRD 前 50% → PB 最低 1/3 三步筛选",
    "primary": {"kind": "spec_text", "name": "GMSD",
                "spec_requirements": {"fields": ["$funda_gross_margin"], "windows": [20],
                                      "operators": [], "structure_ops": [], "chain": ["rank"],
                                      "min_fields": 3}},
}
Q_PLAIN = {"question_id": "RQ_S2", "source": "某报告2", "topic": "通用单因子题",
           "report_structure": "none", "has_reproducible_structure": False,
           "reproduction_target": ""}


def _write(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return p


def test_filters_to_structured_only(tmp_path):
    qp = _write(tmp_path, [Q_STRUCT, Q_PLAIN])
    spec = {"report_policy": {"require_report_structure": True,
                              "require_factor_records": False,
                              "question_queue_file": str(qp)}}
    kept = load_question_queue(spec)
    assert [q["question_id"] for q in kept] == ["RQ_S1"]


def test_off_keeps_all(tmp_path):
    qp = _write(tmp_path, [Q_STRUCT, Q_PLAIN])
    spec = {"report_policy": {"require_report_structure": False,
                              "require_factor_records": False,
                              "question_queue_file": str(qp)}}
    assert len(load_question_queue(spec)) == 2


def test_legacy_bank_without_field_is_not_filtered(tmp_path):
    """旧 1200 题库没有 has_reproducible_structure 字段 → 不筛（行为不变）。"""
    qp = _write(tmp_path, [{"question_id": "RQ_L1", "source": "旧报告", "topic": "旧题"}])
    spec = {"report_policy": {"require_report_structure": True,
                              "require_factor_records": False,
                              "question_queue_file": str(qp)}}
    assert len(load_question_queue(spec)) == 1


def test_falls_back_when_all_filtered_out(tmp_path):
    """全部被判 none → 回退原题库（宁可不筛，也不让 run 无题可做）。"""
    qp = _write(tmp_path, [Q_PLAIN])
    spec = {"report_policy": {"require_report_structure": True,
                              "require_factor_records": False,
                              "question_queue_file": str(qp)}}
    assert len(load_question_queue(spec)) == 1


def test_config_center_key():
    rp = rs.build_run_research_spec(rs.DEFAULT_RESEARCH_SPEC)["report_policy"]
    assert rp["require_report_structure"] is True
    off = rs.build_run_research_spec(
        {"report_policy": {"require_report_structure": False}})["report_policy"]
    assert off["require_report_structure"] is False
