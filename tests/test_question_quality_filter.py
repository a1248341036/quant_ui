# -*- coding: utf-8 -*-
"""质量标签准入门回归测试（2026-10-03）。

背景：实测单题最多烧 49~54 次判定却从未过线（占当夜判定预算 47%）——
`scripts/label_question_quality.py` 依已落盘日志 + 状态机打出 `perf_label`，
准入门按标签排除"效果不好"的题。
"""
from __future__ import annotations

import json

from alphaagent.factor.mining import research_spec as rs
from alphaagent.factor.mining.agent.question_queue import load_question_queue


def _q(qid: str, label: str | None) -> dict:
    r = {"question_id": qid, "source": f"s_{qid}", "topic": qid,
         "has_reproducible_structure": True}
    if label is not None:
        r["perf_label"] = label
    return r


def _bank(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return p


def _spec(qp, **rp):
    rp.setdefault("exclude_perf_labels", ["deprecated", "suspect"])
    rp.setdefault("exclude_method_types", ["ml_model", "graph_deep"])
    return {"report_policy": {"require_report_structure": True,
                              "require_factor_records": False,
                              "question_queue_file": str(qp), **rp}}


def test_excludes_deprecated_and_suspect(tmp_path):
    qp = _bank(tmp_path, [_q("A", "ok"), _q("B", "untried"),
                          _q("C", "suspect"), _q("D", "deprecated")])
    kept = sorted(q["question_id"] for q in load_question_queue(_spec(qp)))
    assert kept == ["A", "B"]


def test_configurable_labels(tmp_path):
    qp = _bank(tmp_path, [_q("A", "ok"), _q("C", "suspect"), _q("D", "deprecated")])
    kept = sorted(q["question_id"] for q in
                  load_question_queue(_spec(qp, exclude_perf_labels=["deprecated"])))
    assert kept == ["A", "C"]


def test_off_keeps_all(tmp_path):
    qp = _bank(tmp_path, [_q("A", "deprecated"), _q("B", "suspect")])
    assert len(load_question_queue(_spec(qp, exclude_perf_labels=[]))) == 2


def test_legacy_bank_without_label_not_filtered(tmp_path):
    qp = _bank(tmp_path, [_q("A", None), _q("B", None)])
    assert len(load_question_queue(_spec(qp))) == 2


def test_empty_result_falls_back(tmp_path):
    qp = _bank(tmp_path, [_q("A", "deprecated")])
    assert len(load_question_queue(_spec(qp))) == 1     # 回退原题库


def test_config_center_default_and_validation():
    rp = rs.build_run_research_spec(rs.DEFAULT_RESEARCH_SPEC)["report_policy"]
    assert rp["exclude_perf_labels"] == ["deprecated", "suspect"]
    off = rs.build_run_research_spec({"report_policy": {"exclude_perf_labels": []}})["report_policy"]
    assert off["exclude_perf_labels"] == []
