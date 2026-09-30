"""因子表达式工具单测：字段/算子别名修复 + 构造模板 + 算子树（锁住行为，防回归）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._factor_expr_tools import (  # noqa: E402
    apply_alias,
    apply_field_alias,
    parse_op_tree,
    try_construct,
)
from scripts.extract_factor_records import validate_expr  # noqa: E402


def test_operator_alias_repairs_repo_names():
    fixed, notes = apply_alias("SUM($volume, 20)")
    assert fixed == "TS_SUM($volume, 20)"
    assert "SUM->TS_SUM" in notes
    assert validate_expr(fixed)[0] is True


def test_field_alias_maps_research_names_to_panel_fields():
    fixed, notes = apply_field_alias("DIVIDE(TS_SUM($dividend_ttm, 4), $close)")
    assert "$dv_ttm" in fixed and "$dividend_ttm" not in fixed
    assert "dividend_ttm->dv_ttm" in notes
    assert validate_expr(fixed)[0] is True


def test_unknown_field_still_rejected():
    """面板未暴露的字段（如 ebitda 只在数据湖 fina_indicator 里）必须仍被拒——不允许硬映射。"""
    ok, why = validate_expr("$ebitda")
    assert ok is False and why.startswith("unknown_field")


def test_construct_templates_use_existing_operators_only():
    for name, definition in (("ILLIQ", "非流动性"), ("CGO_3M", "处置效应"), ("IVRFF", "特异波动率")):
        expr, kw = try_construct(name, definition)
        assert expr and kw
        ok, why = validate_expr(expr)
        assert ok is True, f"{name}: {why}"


def test_op_tree_lists_operators_and_is_stable():
    tree, ops = parse_op_tree("TS_MEAN(DIVIDE(TS_SUM($volume), $amount), 20)")
    assert ops == ("TS_MEAN", "DIVIDE", "TS_SUM")
    assert tree.startswith("TS_MEAN(") and "$amount" in tree
