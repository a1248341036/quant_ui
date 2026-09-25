# -*- coding: utf-8 -*-
"""S1-S4 挖掘期 skill 能力的回归测试（feat/mining-skills）。

- S1 precheck_expression：结构风险静态预检（纯 AST，无数据依赖）
- S2 run_efficiency：本 run 效率自省块（train/val 统计，不触达 test 段）
- S3 turnover_budget：换手预算与 label×调仓频率匹配 prompt 模块
- S4 model_adaptation：按模型注入行为约束（空 model 不渲染 → 黄金基线不变）

关联：docs/specs/alphaagent_mining_skill_candidates.md
"""
from __future__ import annotations

from datetime import datetime

import pytest

from alphaagent.factor.mining.prompt.modules import DEFAULT_MODULES
from alphaagent.factor.mining.prompts import build_system_prompt_with_report


# ── S1: precheck_expression ─────────────────────────────────────────────


def _precheck(expr: str):
    from alphaagent.factor.mining.tools._precheck import precheck_expression

    return precheck_expression(expr)["result"]


def test_s1_gate_collapse_detected():
    r = _precheck("g = GATED_SIGNAL(RANK($close), RANK($volume), 0.9)\nCS_ZSCORE(g)")
    assert any(x["kind"] == "gate_collapse" and x["risk"] == "high" for x in r["risks"])


def test_s1_gate_low_threshold_no_risk():
    r = _precheck("GATED_SIGNAL($ret, $volume, 0.5)")
    assert not any(x["kind"] == "gate_collapse" for x in r["risks"])


def test_s1_piecewise_middle_band_detected():
    r = _precheck("PIECEWISE_STATE($ret, $volume, 0.2, 0.8, -1, 1, 0.0)")
    assert any(x["kind"] == "piecewise_middle_band" for x in r["risks"])


def test_s1_fillna_sparse_detected():
    r = _precheck("FILLNA($funda_roe, 0)")
    assert any(x["kind"] == "sparse_fillna_zero" for x in r["risks"])


def test_s1_clean_expression_no_risk():
    r = _precheck("CS_ZSCORE(TS_MEAN(RANK($ret), 5))")
    assert r["risk_count"] == 0


def test_s1_multi_line_intermediate_vars_are_scanned():
    # parse_ast 只返回最终行；赋值行的 GATED_SIGNAL 必须也被扫描到
    r = _precheck("g = GATED_SIGNAL($ret, $volume, 0.9)\nCS_GROUP_RANK(g, $industry_sw_l1)")
    assert any(x["kind"] == "gate_collapse" for x in r["risks"])


# ── S2: run_efficiency 自省块 ──────────────────────────────────────────


def _live_state(n_eval: int = 0, n_submit: int = 0):
    from alphaagent.factor.mining.run_metrics import (
        new_live_state,
        observe_event,
    )

    st = new_live_state(datetime.now())
    for _ in range(n_eval):
        observe_event(st, "tool_results", {
            "results": [{"name": "evaluate_factor", "result": {"ok": True}}]})
    for _ in range(n_submit):
        observe_event(st, "tool_results", {
            "results": [{"name": "submit_factor", "result": {"ok": True, "stored": True}}]})
    return st


def test_s2_efficiency_block_gives_facts_not_orders():
    from alphaagent.factor.mining.agent.agentscope_run import _efficiency_self_check

    st = _live_state(n_eval=10, n_submit=0)
    text = _efficiency_self_check(st, [])
    assert text, "n_eval>=3 时应注入自省块"
    assert "train 10 次" in text
    assert "0 提交" in text  # 事实，不是指令
    # 族统计在无表达式时不报错
    assert isinstance(text, str)


def test_s2_efficiency_block_early_returns_below_3_evals():
    from alphaagent.factor.mining.agent.agentscope_run import _efficiency_self_check

    st = _live_state(n_eval=2)
    assert _efficiency_self_check(st, []) == ""


def test_s2_efficiency_block_family_cluster_hint():
    from alphaagent.factor.mining.agent.agentscope_run import _efficiency_self_check

    st = _live_state(n_eval=8)
    rows = [{"expression": "CS_GROUP_RANK(RANK($ret), $volume)"}] * 6
    text = _efficiency_self_check(st, rows)
    assert text
    # 同一族集中 ≥60% 时应给出"需评估是否饱和"的提示
    assert "是否已饱和" in text or "饱和" in text


# ── S3: turnover_budget 模块 ───────────────────────────────────────────


def test_s3_turnover_budget_module_registered_and_renders():
    names = [m.name for m in DEFAULT_MODULES]
    assert "turnover_budget" in names
    text, report = build_system_prompt_with_report(
        include_operator_catalog=True, label_col="label_1d_open_to_open",
        include_fundamentals=True, research_spec=None, asset_type="stock",
    )
    row = next(r for r in report if r["module"] == "turnover_budget")
    assert row["chars"] > 0 and not row["required_empty"]
    assert "换手预算" in text


def test_s3_mismatch_verdict_for_1d_weekly():
    from alphaagent.factor.mining.prompt.modules.turnover_budget import _match_verdict

    verdict, _ = _match_verdict(1, "weekly")
    assert verdict == "mismatch"


def test_s3_match_verdict_for_long_label():
    from alphaagent.factor.mining.prompt.modules.turnover_budget import _match_verdict

    # 20d label vs monthly(21d) 属边界 → warn（保守，不算错配）
    assert _match_verdict(20, "monthly")[0] == "warn"
    # 30d 慢信号 vs monthly 明确匹配
    assert _match_verdict(30, "monthly")[0] == "ok"
    # 10d vs weekly(5d) 匹配
    assert _match_verdict(10, "weekly")[0] == "ok"


# ── S4: model_adaptation 模块 ──────────────────────────────────────────


def test_s4_empty_model_does_not_render():
    text, report = build_system_prompt_with_report(
        include_operator_catalog=True, label_col="label_1d_open_to_open",
        include_fundamentals=True, research_spec=None, asset_type="stock",
        model_name="",
    )
    row = next(r for r in report if r["module"] == "model_adaptation")
    assert row["chars"] == 0 and not row["enabled"]


def test_s4_deepseek_renders_behavior():
    text, _ = build_system_prompt_with_report(
        include_operator_catalog=True, label_col="label_1d_open_to_open",
        include_fundamentals=True, research_spec=None, asset_type="stock",
        model_name="deepseek-v4-flash",
    )
    assert "DeepSeek" in text and "promising" in text


def test_s4_gemini_renders_turnover_warning():
    text, _ = build_system_prompt_with_report(
        include_operator_catalog=True, label_col="label_1d_open_to_open",
        include_fundamentals=True, research_spec=None, asset_type="stock",
        model_name="gemini-3.8-flash",
    )
    assert "Gemini" in text and "换手" in text


def test_s4_default_cases_have_no_model_adaptation_content():
    """黄金基线 4 case 均不传 model_name → model_adaptation 不得注入。"""
    for label_col in ("label_1d_open_to_open", "label_20d_close_to_close"):
        text, report = build_system_prompt_with_report(
            include_operator_catalog=True, label_col=label_col,
            include_fundamentals=True, research_spec=None, asset_type="stock",
        )
        row = next(r for r in report if r["module"] == "model_adaptation")
        assert row["chars"] == 0
