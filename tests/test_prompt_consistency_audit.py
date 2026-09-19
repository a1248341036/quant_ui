# -*- coding: utf-8 -*-
"""测试 Prompt 静态自相矛盾消除与规范对齐（对应 docs/specs/alphaagent_prompt_contradictions_repair_spec_v2.md S1~S8）。"""

import pytest
from alphaagent.factor.mining.prompts import build_system_prompt
from alphaagent.factor.mining.research_spec import default_research_spec


def test_no_multiply_in_direct_operators():
    """S2: 基础运算算子中不得将 MULTIPLY 标为语义自明直用。"""
    spec = default_research_spec("technical")
    prompt = build_system_prompt(
        include_operator_catalog=True,
        research_spec=spec,
        asset_type="stock",
    )
    # 算子列表中应该明确提示 MULTIPLY 禁用，基础四则不应裸写 MULTIPLY(df1, df2) 直用
    assert "ADD/SUBTRACT/MULTIPLY/DIVIDE(df1, df2)` 逐元素四则" not in prompt
    assert "ADD/SUBTRACT/DIVIDE(df1, df2)` 逐元素运算" in prompt
    assert "默认禁止 MULTIPLY" in prompt


def test_batch_size_parameterized_in_prompt():
    """S1: 并发数量按 max_tool_calls_per_round 动态参数化渲染，不得硬编码 12~20。"""
    spec = default_research_spec("technical")
    prompt_8 = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
        max_tool_calls_per_round=8,
    )
    assert "12~20 条" not in prompt_8
    assert "12~20 次" not in prompt_8
    assert "≥4 条" in prompt_8 or "≥4" in prompt_8

    prompt_16 = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
        max_tool_calls_per_round=16,
    )
    assert "≥8 条" in prompt_16 or "≥8" in prompt_16


def test_turnover_threshold_aligned():
    """S4: 行为准则中换手率硬红线必须明确与系统 0.50 门槛对齐。"""
    spec = default_research_spec("technical")
    prompt = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
    )
    assert "avg_daily_side_turnover <= 0.50" in prompt
    assert "设计目标建议控制在 `<= 0.40`" in prompt


def test_orthogonality_tiers_clarified():
    """S3: 正交性三级阶梯说明清晰分层（0.70 / 0.50 / 0.40）。"""
    spec = default_research_spec("technical")
    prompt = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
    )
    assert "三级截面相关性阶梯" in prompt
    assert "0.70" in prompt
    assert "0.50" in prompt
    assert "0.40" in prompt


def test_negative_ic_delivery_reversal():
    """S7: 负 IC 明确说明交付前必须顶层包裹 NEG() 翻转转正。"""
    spec = default_research_spec("technical")
    prompt = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
    )
    assert "必须在表达式顶层包裹 `NEG()`" in prompt or "必须顶层套 `NEG()`" in prompt


def test_behavior_rules_decoupled_from_layer_numbers():
    """S8: behavior_rules 解耦第一层、第二层等跨模块章节序号引用。"""
    spec = default_research_spec("technical")
    prompt = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
    )
    assert "标准见第一层" not in prompt
    assert "第二层轨道 D 的硬约束" not in prompt
