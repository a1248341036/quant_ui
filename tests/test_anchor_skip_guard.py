# -*- coding: utf-8 -*-
"""连续锚失败护栏（#1）+ 锁轮数收口（#3）回归测试，2026-10-03。

背景：2026-10-03 整夜实测 —— 唯一一道"机器学习"类题连续被原文锚拦截 **93 次**
（占当夜全部拦截的 69%），整轮预算烧在"DSL 表达不了"的题上；同时发现
`reproduce_lock_rounds` 的默认值 3 只存在于 `report_channels` 的 `.get(...,3)`，
未进配置中心（违反三件套）。
"""
from __future__ import annotations

import json

import pytest

from alphaagent.factor.mining import research_spec as rs
from alphaagent.factor.mining.agent.question_queue import (
    get_question_for_turn,
    note_anchor_block,
)


# ── #1a 纯函数：计数/清零/阈值 ──

def test_counter_increments_and_triggers_at_threshold():
    counts: dict[str, int] = {}
    assert [note_anchor_block(counts, "Q1", False, 3) for _ in range(3)] == [False, False, True]
    assert counts["Q1"] == 3


def test_pass_resets_counter():
    counts: dict[str, int] = {}
    note_anchor_block(counts, "Q1", False, 3)
    note_anchor_block(counts, "Q1", False, 3)
    assert note_anchor_block(counts, "Q1", True, 3) is False   # 锚达标 → 清零
    assert counts["Q1"] == 0
    # 清零后再连续两次不应触发
    assert note_anchor_block(counts, "Q1", False, 3) is False
    assert note_anchor_block(counts, "Q1", False, 3) is False


def test_threshold_zero_disables_guard():
    counts: dict[str, int] = {}
    for _ in range(50):
        assert note_anchor_block(counts, "Q1", False, 0) is False


def test_blank_qid_is_noop():
    counts: dict[str, int] = {}
    assert note_anchor_block(counts, "", False, 1) is False
    assert counts == {}


def test_counts_are_per_question():
    counts: dict[str, int] = {}
    assert note_anchor_block(counts, "Q1", False, 2) is False
    assert note_anchor_block(counts, "Q2", False, 2) is False   # 另一题不受影响
    assert note_anchor_block(counts, "Q1", False, 2) is True


# ── #1b 选题器：exclude_qids 生效 ──

def _bank(tmp_path, qids):
    p = tmp_path / "q.jsonl"
    rows = [{"question_id": q, "source": f"src_{q}", "topic": q,
             "has_reproducible_structure": True} for q in qids]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return p


def _spec(qp):
    return {"report_policy": {"require_report_structure": True,
                              "require_factor_records": False,
                              "question_queue_file": str(qp)}}


def test_exclude_qids_skips_that_question(tmp_path):
    qp = _bank(tmp_path, ["Q1", "Q2", "Q3"])
    spec = _spec(qp)
    # 不排除：turn 0 → Q1
    assert get_question_for_turn(0, spec=spec)["question_id"] == "Q1"
    # 排除 Q1：池变 [Q2, Q3]，turn 0 → Q2，turn 1 → Q3，turn 2 → Q2
    got = [get_question_for_turn(t, spec=spec, exclude_qids={"Q1"})["question_id"]
           for t in (0, 1, 2)]
    assert got == ["Q2", "Q3", "Q2"]
    assert "Q1" not in got


def test_exclude_all_falls_back_to_original_pool(tmp_path):
    """过滤后为空 → 回退原池（宁可不跳，也不让 run 无题可做）。"""
    qp = _bank(tmp_path, ["Q1", "Q2"])
    spec = _spec(qp)
    q = get_question_for_turn(0, spec=spec, exclude_qids={"Q1", "Q2"})
    assert q is not None and q["question_id"] in ("Q1", "Q2")


def test_exclude_qids_none_is_unchanged(tmp_path):
    qp = _bank(tmp_path, ["Q1", "Q2"])
    spec = _spec(qp)
    assert get_question_for_turn(0, spec=spec, exclude_qids=None)["question_id"] == "Q1"
    assert get_question_for_turn(0, spec=spec, exclude_qids=[])["question_id"] == "Q1"


# ── #3 配置中心三件套：默认值 + 覆盖 + 越界被拒 ──

def test_config_defaults_present():
    rp = rs.build_run_research_spec(rs.DEFAULT_RESEARCH_SPEC)["report_policy"]
    assert rp["reproduce_lock_rounds"] == 3          # 本次收口**不改值**（有实测依据）
    assert rp["anchor_block_skip_threshold"] == 8


def test_config_overrides_apply():
    rp = rs.build_run_research_spec({"report_policy": {
        "reproduce_lock_rounds": 5, "anchor_block_skip_threshold": 0}})["report_policy"]
    assert rp["reproduce_lock_rounds"] == 5
    assert rp["anchor_block_skip_threshold"] == 0     # 0 = 关闭护栏


@pytest.mark.parametrize("bad", [
    {"reproduce_lock_rounds": 13},
    {"reproduce_lock_rounds": -1},
    {"anchor_block_skip_threshold": 201},
    {"anchor_block_skip_threshold": -1},
])
def test_config_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        rs.build_run_research_spec({"report_policy": bad})
