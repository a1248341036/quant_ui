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
    # 实际措辞：operator 说明行写 "默认禁止特征间使用 MULTIPLY 乘法"
    assert "默认禁止特征间使用 MULTIPLY" in prompt


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
    # 实际措辞：D 轨标题写 "若本轮提交 8 条则至少 4 条为 D 轨"
    assert "至少 4 条" in prompt_8 or "4 条" in prompt_8

    prompt_16 = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
        max_tool_calls_per_round=16,
    )
    assert "至少 8 条" in prompt_16 or "8 条" in prompt_16


def test_turnover_threshold_aligned():
    """S4（2026-09-22 分档版）/ 2026-10-04（label↔freq 强制一致）: 换手硬红线按
    engine_gate.freq 分档渲染，与 delivery_checker 零漂移。

    2026-10-04 起 label 与调仓频率强制一致：technical 档默认已从 weekly 收成
    **daily**（label_1d）→ 渲染 0.50；weekly 档由 label_5d 的 technical_weekly 承担 → 0.65。
    指标名同步为调仓口径 `avg_rebalance_side_turnover`（daily↔1d 时数值与日频一致）。
    """
    spec = default_research_spec("technical")
    assert spec["delivery_policy"]["production"]["engine_gate"]["freq"] == "daily"
    prompt = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
    )
    # daily 档（technical 默认，label_1d）：渲染分档值 0.5
    assert "avg_rebalance_side_turnover <= 0.5" in prompt
    # behavior_rules 中 0.4 带反引号渲染（预警线固定，不随 freq 分档）
    assert "`0.4` 为 diagnostics 诊断预警线" in prompt

    # weekly 档（label_5d 的 technical_weekly）：渲染 0.65
    prompt_weekly = build_system_prompt(
        include_operator_catalog=False,
        research_spec=default_research_spec("technical_weekly"),
        asset_type="stock",
    )
    assert "avg_rebalance_side_turnover <= 0.65" in prompt_weekly


def test_orthogonality_tiers_clarified():
    """S3: 正交性三级阶梯说明清晰分层（0.70 / 0.50 / 0.40）。"""
    spec = default_research_spec("technical")
    prompt = build_system_prompt(
        include_operator_catalog=False,
        research_spec=spec,
        asset_type="stock",
    )
    assert "三级截面相关性阶梯" in prompt
    # 0.70 = DSL 防爆线（硬编码）；0.5/0.4 = 动态渲染 str(0.5)/str(0.4)
    assert "0.70" in prompt
    assert "0.5" in prompt
    assert "0.4" in prompt


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
