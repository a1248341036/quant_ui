# -*- coding: utf-8 -*-
"""方法类型准入（P2，2026-10-02）：默认排除 ml_model / graph_deep。

背景：这类研报的核心是训练出来的模型/网络结构（GNN、Transformer、XGBoost…），DSL 表达不了，
硬出题只能得到"用任意表达式近似"的伪复现（实测 RQ_7810a8：目标写"异构图 GNN+两阶段训练"，
实际复现成"成交额/VWAP 溢价残差"）。分类由 LLM 产出（scripts/classify_report_methods.py）。
"""
from __future__ import annotations

import json

from alphaagent.factor.mining import research_spec as rs
from alphaagent.factor.mining.agent.question_queue import load_question_queue


def _q(qid: str, method: str | None) -> dict:
    r = {"question_id": qid, "source": f"某报告_{qid}", "topic": qid,
         "has_reproducible_structure": True}
    if method is not None:
        r["method_type"] = method
    return r


def _bank(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return p


def _spec(qp, **rp):
    rp.setdefault("exclude_method_types", ["ml_model", "graph_deep"])   # 生产口径：normalize 后必带
    return {"report_policy": {"require_report_structure": True,
                              "require_factor_records": False,
                              "question_queue_file": str(qp), **rp}}


def test_excludes_ml_and_graph(tmp_path):
    qp = _bank(tmp_path, [_q("A", "rule_formula"), _q("B", "ml_model"),
                          _q("C", "graph_deep"), _q("D", "portfolio_combo")])
    kept = [q["question_id"] for q in load_question_queue(_spec(qp))]
    assert sorted(kept) == ["A", "D"]


def test_configurable_whitelist_of_methods(tmp_path):
    """把某类从排除表移除即可放开（例如只想排除图深度）。"""
    qp = _bank(tmp_path, [_q("A", "rule_formula"), _q("B", "ml_model"), _q("C", "graph_deep")])
    kept = [q["question_id"] for q in load_question_queue(
        _spec(qp, exclude_method_types=["graph_deep"]))]
    assert sorted(kept) == ["A", "B"]


def test_off_keeps_all(tmp_path):
    qp = _bank(tmp_path, [_q("A", "ml_model")])
    kept = [q["question_id"] for q in load_question_queue(
        _spec(qp, exclude_method_types=[]))]
    assert kept == ["A"]


def test_legacy_bank_without_field_not_filtered(tmp_path):
    qp = _bank(tmp_path, [_q("A", None), _q("B", None)])
    assert len(load_question_queue(_spec(qp))) == 2


def test_empty_result_falls_back(tmp_path):
    qp = _bank(tmp_path, [_q("A", "ml_model")])
    assert len(load_question_queue(_spec(qp))) == 1      # 回退原题库，不让 run 无题


def test_config_center_default_and_override():
    rp = rs.build_run_research_spec(rs.DEFAULT_RESEARCH_SPEC)["report_policy"]
    assert rp["exclude_method_types"] == ["ml_model", "graph_deep"]
    off = rs.build_run_research_spec(
        {"report_policy": {"exclude_method_types": []}})["report_policy"]
    assert off["exclude_method_types"] == []
