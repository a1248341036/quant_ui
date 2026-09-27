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


def test_s1_gate_low_threshold_also_flagged():
    """2026-09-27 标定修正：合法区间内 GATED_SIGNAL 必然塌缩，threshold=0.5 同样要报。

    旧断言是 `no risk`——那正是把 bug 编码成测试：实测 threshold=0.5 → 常数簇 50%
    → 十分位只有 6 组 < 8，必然 stage_one 失败。
    详见 docs/specs/alphaagent_decile_collapse_root_fix_spec.md §2.1。
    """
    r = _precheck("GATED_SIGNAL($ret, $volume, 0.5)")
    assert any(x["kind"] == "gate_collapse" and x["risk"] == "high" for x in r["risks"])
    assert r["blocked"] is True


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
    # 事实陈述，不是祈使指令（"应/需/必须"均不应出现）
    assert "submit_factor 调用 0 次" in text
    for word in ("应直接", "必须", "需评估"):
        assert word not in text, f"自省块不应含指令性措辞: {word}"
    assert isinstance(text, str)


def test_s2_efficiency_block_early_returns_below_3_evals():
    from alphaagent.factor.mining.agent.agentscope_run import _efficiency_self_check

    st = _live_state(n_eval=2)
    assert _efficiency_self_check(st, []) == ""


def test_s2_efficiency_block_family_cluster_hint():
    from alphaagent.factor.mining.agent.agentscope_run import _efficiency_self_check

    st = _live_state(n_eval=8)
    rows = [{"name": "evaluate_factor", "expression": "CS_GROUP_RANK(RANK($ret), $volume)"}] * 6
    text = _efficiency_self_check(st, rows)
    assert text
    # 同族集中时应给出事实陈述（该族出现次数），不含祈使指令
    assert "集中于该族" in text


def test_s2_family_stats_exclude_non_eval_tools():
    """P2-2：族统计只算评估类工具；submit/precheck 的 expression 不计入。"""
    from alphaagent.factor.mining.agent.agentscope_run import _efficiency_self_check

    st = _live_state(n_eval=8)
    rows = (
        [{"name": "evaluate_factor", "expression": "CS_GROUP_RANK(RANK($ret), $volume)"}] * 6
        + [{"name": "submit_factor", "expression": "GATED_SIGNAL($ret, $volume, 0.9)"}]
        + [{"name": "precheck_expression", "expression": "PIECEWISE_STATE($ret, $volume, 0.2, 0.8)"}]
    )
    text = _efficiency_self_check(st, rows)
    # 评估类计入 → 族分布行出现
    assert "集中于该族" in text
    # 非评估类的表达式不得进入自省块（含族统计与全文）
    assert "GATED_SIGNAL" not in text and "PIECEWISE_STATE" not in text


def test_s1_dispatch_route():
    """P3-6：dispatch 路由直达 precheck_expression（纯函数之外的接线验证）。"""
    from alphaagent.factor.mining.tools._dispatch import _DispatchMixin

    class _MiniTools(_DispatchMixin):
        def __init__(self) -> None:
            self.service = None
            self.session_id = "x"
            self.submit_service = None
            self.memory_store = None
            self.focus_facets = ()
            self.cognition_policy = {}
            self.operator_policy = {"blacklist": ()}
            self.homogenization_policy = {"enabled": False}
            self._recent_evals = []

    r = _MiniTools().dispatch("precheck_expression", {"multi_line_expr": "GATED_SIGNAL($ret, $volume, 0.9)"})
    assert r.get("ok") is True
    assert r["result"]["risks"][0]["kind"] == "gate_collapse"

    bad = _MiniTools().dispatch("precheck_expression", {"multi_line_expr": ""})
    assert bad.get("ok") is False and bad.get("error_type") == "ToolArgumentsError"


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


# ── S0: promising 语义 + 强制 val（feat/promising-to-val-gate）────────────────


def test_promising_conclusion_marks_train_passed_semantics():
    """promising verdict 文案明确'训练样本海选过线，非质量结论，必须 val'。"""
    from alphaagent.factor.mining.memory.schema import SchemaMixin

    metrics = {"ic": 0.024, "icir": 0.3, "factor_coverage": 0.92}
    result = {"ok": True, "split": "train", "metrics": metrics}
    verdict, conclusion = SchemaMixin._classify("eval_on_train_set", result, metrics, "")
    assert verdict == "promising"
    assert "训练样本海选过线" in conclusion
    assert "eval_on_val_set" in conclusion


class _AutoValService:
    """模拟 StockEvalService.eval_val：记录调用，返回固定 val 结果。"""

    def __init__(self, val_ok: bool = True) -> None:
        self.val_ok = val_ok
        self.calls: list[dict] = []

    def eval_val(self, req):  # noqa: ANN001
        self.calls.append({"expr": req.multi_line_expr, "sign": req.expected_sign})
        if not self.val_ok:
            return {"ok": True, "summary": {"ic": 0.005, "icir": 0.05, "factor_coverage": 0.9}}
        return {"ok": True, "sign_check": {"matches_expected_sign": True},
                "summary": {"ic": 0.02, "icir": 0.25, "factor_coverage": 0.92}}


def _promising_result():
    return {"ok": True, "split": "train",
            "metrics": {"cross_sectional_core": {"ic": 0.024, "icir": 0.3, "factor_coverage": 0.92}}}


def _mini_tools(svc):  # noqa: ANN001
    from alphaagent.factor.mining.tools._dispatch import _DispatchMixin

    class _T(_DispatchMixin):
        def __init__(self, service) -> None:  # noqa: ANN001
            self.service = service
            self.session_id = "s1"
            self.cognition_policy = {"force_val_on_promising": True}
            self._auto_val_done = set()

    return _T(svc)


def test_auto_val_verify_runs_on_promising_and_passes():
    svc = _AutoValService(val_ok=True)
    t = _mini_tools(svc)
    result = _promising_result()
    t._auto_val_verify(result, "RANK($ret)", "f1")
    assert len(svc.calls) == 1 and svc.calls[0]["sign"] == 1
    vv = result.get("val_verification")
    assert vv is not None and vv["passed"] is True
    assert result.get("val_failed") is None


def test_auto_val_verify_marks_failed_when_val_weak():
    svc = _AutoValService(val_ok=False)
    t = _mini_tools(svc)
    result = _promising_result()
    t._auto_val_verify(result, "RANK($ret)", "f1")
    assert result.get("val_failed") is True
    assert "样本外验证未通过" in result["val_verification"]["guidance"]


def test_auto_val_verify_skips_below_promising_line():
    svc = _AutoValService(val_ok=True)
    t = _mini_tools(svc)
    weak = {"ok": True, "split": "train",
            "metrics": {"cross_sectional_core": {"ic": 0.005, "icir": 0.05, "factor_coverage": 0.9}}}
    t._auto_val_verify(weak, "RANK($ret)", "f1")
    assert svc.calls == [] and "val_verification" not in weak


def test_auto_val_verify_dedups_by_expression():
    svc = _AutoValService(val_ok=True)
    t = _mini_tools(svc)
    t._auto_val_verify(_promising_result(), "RANK($ret)", "f1")
    t._auto_val_verify(_promising_result(), "RANK($ret)", "f2")
    assert len(svc.calls) == 1  # 同一表达式只自动 val 一次


def test_guaranteed_positives_excludes_promising(tmp_path):
    """2026-09-25：promising 不再进入跨族保底正向父本（_guaranteed_positives）。"""
    import tests.test_research_memory_v3 as T
    from alphaagent.factor.mining.research_memory import ResearchMemoryStore

    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(run_id="r1", row=T._eval_row("eval_on_train_set", "RANK($ret)", "p1", ic=0.024))
    assert store._guaranteed_positives(k=5) == []
