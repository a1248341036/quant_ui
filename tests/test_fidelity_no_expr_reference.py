# -*- coding: utf-8 -*-
"""保真度锚回归：参照记录**存在但无表达式文本**时不得把课题永久卡死。

实测（2026-10-02，run 5ce845512076 / 课题 RQ_13e253）：
该报告 13 条 factor_records 全是"联发科:营收 / 晶圆进口金额"这类只有指标名、无公式的记录，
→ ref_fields=∅ → `require_shared_field=1` 判 `shared_fields=0<1` **永远失败**，
该课题 64 条复现判定 0 PASS（模型其实用了题面声明的 `$funda_total_revenue`）。
v2 抽取后 spec_text 类报告（占多数）都会命中此路径。
"""
from __future__ import annotations

from alphaagent.factor.mining.agent.question_queue import check_reproduce_fidelity

SRC = "nn73/01/2022-05-12_国泰君安证券_金融工程_电子产业链基本面量化及策略配置.md"
NO_EXPR_RECS = [
    {"source": SRC, "name": "联发科:营收", "expr_local": None, "expr_raw": None},
    {"source": SRC, "name": "晶圆(化学品)进口金额", "expr_local": "", "expr_raw": ""},
]


def _q(sr: dict) -> dict:
    return {"question_id": "RQ_13e253", "source": "国泰君安证券_电子产业链基本面量化及策略配置",
            "has_reproducible_structure": True, "primary": {"spec_requirements": sr}}


def test_no_expr_reference_does_not_block():
    # 复现用到 2 个字段 + 2 个算子（满足弱声明复杂度地板），应与模型实际写法一致
    out = check_reproduce_fidelity(
        _q({"fields": ["$funda_total_revenue"], "min_fields": 1}),
        "CS_ZSCORE(TS_MEAN(DIVIDE($funda_total_revenue, $funda_total_assets), 63))",
        records=NO_EXPR_RECS)
    assert out["passed"] is True, out
    assert out["anchor_mode"] == "no_reference_expr"
    assert out["n_records"] == 2


def test_no_expr_reference_still_respects_complexity_floor():
    """退化锚不等于放行：弱声明的复杂度地板仍然生效（挡一行式）。"""
    out = check_reproduce_fidelity(_q({"fields": ["$funda_total_revenue"], "min_fields": 1}),
                                   "CS_ZSCORE($funda_total_revenue)", records=NO_EXPR_RECS)
    assert out["passed"] is False
    assert "ops_used=1<floor=2" in out["reason"]
