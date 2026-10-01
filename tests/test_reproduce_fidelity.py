# -*- coding: utf-8 -*-
"""复现保真度「原文锚」+ 只派发有公式课题（2026-10-01，仅研报模式）的回归测试。

背景：昨夜实测有原文公式的 10 个课题，复现公式与原文的字段 Jaccard 平均 0.062、
算子 Jaccard 0.085——模型靠题面约束"脱稿自由探索"，用容易过线的通用因子
（5 日反转/Amihud/PE 动量）顶替原文机制。故：
  ① ^report_policy.require_factor_records`：只派发有原文公式的课题；
  ② `report_policy.reproduce_fidelity`：复现过线前校验与原文公式的字段/算子重叠。
"""
from __future__ import annotations

import json

from alphaagent.factor.mining import research_spec as rs
from alphaagent.factor.mining.agent.question_queue import (
    check_reproduce_fidelity,
    load_question_queue,
    resolve_reproduce_fidelity,
)

Q = {"question_id": "RQ_T1", "source": "华泰证券_单因子测试之财务质量因子"}
SRC = "nn73/01/2018-05-25_华泰证券_金融工程_华泰单因子测试之财务质量因子.md"


def _rec(expr: str, *, source: str = SRC) -> dict:
    return {"source": source, "expr_local": expr, "formula_kind": "verbatim", "executable": True}


# ── ① 保真度校验 ──────────────────────────────────────────────────────────

def test_no_reference_does_not_block():
    """报告不在覆盖集（无原文公式）→ 不因缺参照卡死。"""
    out = check_reproduce_fidelity(Q, "RANK($close)", records=[])
    assert out["passed"] is True
    assert out["reason"] == "no_reference"
    assert out["n_records"] == 0


def test_off_reference_reproduction_is_blocked():
    """复现公式完全没用到原文特有字段 → 判不过（这正是昨夜 RQ_034/040/048 的形态）。"""
    records = [_rec("DIVIDE($funda_ocf, $funda_total_revenue)"),
               _rec("DIVIDE($funda_current_liabilities, $funda_total_liabilities)")]
    # 教科书 Amihud：只用通用行情字段
    out = check_reproduce_fidelity(Q, "CS_ZSCORE(TS_MEAN(DIVIDE(ABS($ret), $amount), 5))",
                                   records=records)
    assert out["passed"] is False
    assert out["reason"].startswith("off_reference:")
    assert out["shared_specific_fields"] == []


def test_faithful_reproduction_passes():
    """复现公式命中原文特有字段 → 过。"""
    records = [_rec("DIVIDE($funda_ocf, $funda_total_revenue)")]
    out = check_reproduce_fidelity(Q, "RANK(CS_WINSORIZE(DIVIDE($funda_ocf, $funda_total_revenue), 0.01, 0.99))",
                                   records=records)
    assert out["passed"] is True
    assert "funda_ocf" in out["shared_specific_fields"]
    assert out["field_jaccard"] > 0


def test_generic_market_field_alone_is_not_an_anchor():
    """只共享 $adj_close 这类通用行情字段，不算锚定证据。"""
    records = [_rec("TS_MEAN($adj_close, 20)"), _rec("DIVIDE($funda_ocf, $funda_total_revenue)")]
    out = check_reproduce_fidelity(Q, "NEG(TS_PCTCHANGE($adj_close, 5))", records=records)
    assert out["passed"] is False, out
    assert out["shared_fields"] == ["adj_close"]
    assert out["shared_specific_fields"] == []


def test_jaccard_threshold_enforced():
    """显式 Jaccard 下限生效（默认 0 不启用）。"""
    records = [_rec("DIVIDE($funda_ocf, $funda_total_revenue), $funda_net_profit, $funda_roe")]
    spec = {"report_policy": {"reproduce_fidelity": {"enabled": True,
                                                     "require_shared_field": 0,
                                                     "min_field_jaccard": 0.5}}}
    out = check_reproduce_fidelity(Q, "RANK($funda_ocf)", spec=spec,
                                   records=records)
    assert out["passed"] is False
    assert "field_jaccard" in out["reason"]


def test_resolve_defaults_and_override():
    d = resolve_reproduce_fidelity(None)
    assert d["enabled"] is True and d["require_shared_field"] == 1
    o = resolve_reproduce_fidelity({"report_policy": {"reproduce_fidelity": {
        "require_shared_field": 3, "min_shared_ops": 2}}})
    assert o["require_shared_field"] == 3 and o["min_shared_ops"] == 2
    # 未知键不污染配置
    o2 = resolve_reproduce_fidelity({"report_policy": {"reproduce_fidelity": {"bogus": 1}}})
    assert "bogus" not in o2


# ── ② 只派发有原文公式的课题 ──────────────────────────────────────────────

def _write_queue(tmp_path, questions):
    p = tmp_path / "questions.jsonl"
    p.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n",
                 encoding="utf-8")
    return p


def _write_records(tmp_path, records):
    p = tmp_path / "records.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                 encoding="utf-8")
    return p


def test_require_factor_records_keeps_only_covered(tmp_path):
    qp = _write_queue(tmp_path, [Q, {"question_id": "RQ_T2", "source": "某券商_完全不相干的周报标题"}])
    rp = _write_records(tmp_path, [_rec("DIVIDE($funda_ocf, $funda_total_revenue)")])
    spec = {"report_policy": {"require_factor_records": True,
                              "question_queue_file": str(qp),
                              "factor_records_file": str(rp)}}
    kept = load_question_queue(spec)
    assert [q["question_id"] for q in kept] == ["RQ_T1"]


def test_require_factor_records_off_keeps_all(tmp_path):
    qp = _write_queue(tmp_path, [Q, {"question_id": "RQ_T2", "source": "某券商_完全不相干的周报标题"}])
    rp = _write_records(tmp_path, [_rec("DIVIDE($funda_ocf, $funda_total_revenue)")])
    spec = {"report_policy": {"require_factor_records": False,
                              "question_queue_file": str(qp),
                              "factor_records_file": str(rp)}}
    assert len(load_question_queue(spec)) == 2


def test_require_factor_records_falls_back_when_no_records(tmp_path):
    qp = _write_queue(tmp_path, [Q])
    spec = {"report_policy": {"require_factor_records": True,
                              "question_queue_file": str(qp),
                              "factor_records_file": str(tmp_path / "missing.jsonl")}}
    # 清单缺失 → 不筛（宁可不筛也不要 run 无题可做）
    assert len(load_question_queue(spec)) == 1


# ── ③ spec 校验 ───────────────────────────────────────────────────────────

def test_spec_validates_new_report_policy_keys():
    spec = rs.build_run_research_spec(rs.DEFAULT_RESEARCH_SPEC)
    rp = spec["report_policy"]
    assert rp["require_factor_records"] is True
    fid = rp["reproduce_fidelity"]
    assert fid["enabled"] is True
    assert fid["require_shared_field"] == 1
    assert fid["min_field_jaccard"] == 0.0 and fid["min_op_jaccard"] == 0.0
    assert fid["min_shared_ops"] == 0


# ── ④ 边界：原文公式全用通用行情字段的报告 ────────────────────────────────
# 实测 361 篇覆盖报告里 83 篇（23%）属于此类（如纯 VWAP/close 乖离、K 线高低点）。
# 若一律要求"命中报告特有字段"，这些课题下**忠实复现也永远过不了**（参照集本身没有
# 特有字段）→ 必须退化为按全量字段判定。

GENERIC_Q = {"question_id": "RQ_T9", "source": "东方证券_反转因子择时研究"}
GENERIC_SRC = "nn73/01/2018-02-22_《因子选股系列研究三十三》_反转因子择时研究_东方证券.md"


def _grec(expr: str) -> dict:
    return {"source": GENERIC_SRC, "expr_local": expr, "formula_kind": "verbatim",
            "executable": True}


def test_generic_only_reference_allows_faithful_reproduction():
    """原文只含通用行情字段时，忠实复现（共享该字段）必须能通过。"""
    records = [_grec("TS_PCTCHANGE($close, 60)"), _grec("TS_STD($close, 20)")]
    out = check_reproduce_fidelity(GENERIC_Q, "RANK(TS_PCTCHANGE($close, 20))", records=records)
    assert out["passed"] is True, out
    assert out["anchor_mode"] == "generic_only_reference"
    assert out["shared_fields"] == ["close"]
    assert out["ref_specific_fields"] == []


def test_generic_only_reference_still_blocks_no_overlap():
    """退化模式下仍要求有共享字段：完全不沾原文的复现照样拦。"""
    records = [_grec("TS_PCTCHANGE($close, 60)")]
    out = check_reproduce_fidelity(GENERIC_Q, "DIVIDE($funda_ocf, $funda_total_revenue)",
                                   records=records)
    assert out["passed"] is False, out
    assert out["anchor_mode"] == "generic_only_reference"


def test_specific_reference_does_not_degrade():
    """有特有字段的参照集仍走严格模式（通用字段不算锚）。"""
    records = [_rec("DIVIDE($funda_ocf, $funda_total_revenue)")]
    out = check_reproduce_fidelity(Q, "NEG(TS_PCTCHANGE($adj_close, 5))", records=records)
    assert out["passed"] is False
    assert out["anchor_mode"] == "specific"
