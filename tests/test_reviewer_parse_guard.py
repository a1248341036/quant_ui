# -*- coding: utf-8 -*-
"""FactorReviewer 输出解析兜底：**绝不因格式问题把因子判死**（2026-10-01 修）。

实测背景（run fcc9a4618fe2）：模型在字符串值里写了未转义引号
（``...split="val" 结果与 ablation ...``），json.loads 在 char 425 报
``Expecting ',' delimiter``；输出本身**完整未截断**、且有 ```json 围栏。
旧实现把任何解析失败一律合成 ``verdict=reject`` → 因格式问题硬拦候选池
（delivery/submit.py:890）。
"""
from __future__ import annotations

from alphaagent.factor.mining.agent.factor_reviewer import FactorReviewer

# 近乎原样的真实畸形输出（未转义引号 + ```json 围栏）
REAL_MALFORMED = (
    '```json\n'
    '{"verdict":"reject","novelty":"low","canonical_form":"短期反转(5日)按股东数变化分组 + 行业/市值中性化",'
    '"reasons":["因子主信号 rev5 与 ablation 中 rev5 组内排序变体的 train/val IC 相近",'
    '"monthly_corr_robustness 仅 24 个月，split="val" 结果与 ablation 一致"],'
    '"required_changes":["补充独立经济假设"]}\n'
    '```'
)


def test_recovers_verdict_from_unescaped_quotes():
    """真实畸形样本：必须捞回模型的真实判定，而不是判成"不可解析"。"""
    v = FactorReviewer._parse_verdict(REAL_MALFORMED)
    assert v["verdict"] == "reject"            # 模型本来判的就是 reject
    assert v["novelty"] == "low"
    assert v["source"] == "agentscope_factor_reviewer_repaired"
    assert "容错解析" in v["canonical_form"]


def test_fenced_valid_json_parses_normally():
    raw = '```json\n{"verdict":"approve","novelty":"medium","canonical_form":"x"}\n```'
    v = FactorReviewer._parse_verdict(raw)
    assert v["verdict"] == "approve"
    assert v["novelty"] == "medium"
    assert v["source"] == "agentscope_factor_reviewer"


def test_unparseable_output_degrades_to_revise_not_reject():
    """连 verdict 都捞不到时，必须降级为 revise（不阻断），不得合成 reject。"""
    for raw in ("模型输出了一堆散文，没有 JSON", "", "```\n{\"foo\":\n```"):
        v = FactorReviewer._parse_verdict(raw)
        assert v["verdict"] == "revise", f"不应 fail-closed 判死: {raw!r} -> {v['verdict']}"
        assert v["source"] == "reviewer_parse_guard"


def test_truncated_json_also_degrades_to_revise():
    """截断样本（无收尾括号）同样不得判 reject。"""
    raw = '{"verdict":"approve","novelty":"high","reasons":["被 max_tokens 截断'
    v = FactorReviewer._parse_verdict(raw)
    assert v["verdict"] in {"approve", "revise"}
    assert v["verdict"] != "reject"
