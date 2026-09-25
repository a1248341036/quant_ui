# 回归：提交期审查包注入完整消融矩阵（逐表达式 × 各口径）
from collections import defaultdict

from alphaagent.factor.mining.factor_reviewer import FactorReviewer


def _mk():
    r = FactorReviewer.__new__(FactorReviewer)
    r.evaluations = defaultdict(lambda: defaultdict(list))
    r.eval_expr_head = {}
    r.reviews = {}
    return r


def test_ablation_matrix_covers_all_exprs_and_splits():
    r = _mk()
    combo = "COMBO"
    r.evaluations[combo]["train"].append({"summary": {"ic": 0.0285, "icir": 0.519}})
    r.eval_expr_head[combo] = "combo expr"

    # 腿级：同表达式先 train 后 val 两条记录 → 取各自最新
    leg_gap = "GAP"
    r.evaluations[leg_gap]["train"].append({"summary": {"ic": 0.021, "icir": 0.2313}})
    r.evaluations[leg_gap]["val"].append({
        "metrics": {"ic": 0.019, "icir": 0.21},
        "monthly_corr_robustness": {"share_months_ic_positive": 0.889},
    })
    r.eval_expr_head[leg_gap] = "gap only expr"

    # profile 口径（如 size_neutral_validation）也要进矩阵
    leg_vwap = "VWAP"
    r.evaluations[leg_vwap]["size_neutral_validation"].append({
        "summary": {"ic": 0.0212, "icir": 0.337},
    })
    r.eval_expr_head[leg_vwap] = "vwap resid expr"

    rows = r.ablation_matrix(combo)
    assert len(rows) == 2
    by_head = {row["expr_head"]: row for row in rows}

    gap_row = by_head["gap only expr"]
    assert gap_row["splits"]["train"]["ic"] == 0.021
    assert gap_row["splits"]["val"]["ic"] == 0.019
    assert gap_row["splits"]["val"]["monthly_ic_positive"] == 0.889

    vwap_row = by_head["vwap resid expr"]
    assert vwap_row["splits"]["size_neutral_validation"]["icir"] == 0.337

    # 自身排除
    assert all(row["expr_head"] != "combo expr" for row in rows)


def test_matrix_skips_exprs_without_metrics():
    r = _mk()
    r.evaluations["X"]["train"].append({"ok": True})  # 无指标字段
    r.eval_expr_head["X"] = "empty"
    assert r.ablation_matrix("COMBO") == []


def test_extract_eval_metrics_prefers_summary_over_metrics():
    entry = {
        "summary": {"ic": 0.1},
        "metrics": {"ic": 0.9},
    }
    out = FactorReviewer._extract_eval_metrics(entry)
    assert out["ic"] == 0.1


# ---------------------------------------------------------------------------
# 证据口径统计预检只在 validation 阶段做；pre_submit 的统计裁决归 stage_one
# （ingest 口径，无 winsorize），避免同一门槛套两个口径产出矛盾 revise。
# ---------------------------------------------------------------------------

def test_metric_precheck_only_at_validation_stage():
    import types

    r = _mk()
    r.config = types.SimpleNamespace(research_spec={})
    expr = "TS_MEAN($close, 5)"
    r.record_evaluation("train", {"multi_line_expr": expr},
                        {"ok": True, "summary": {"ic": 0.001, "icir": 0.05, "factor_coverage": 0.99}})
    r.record_evaluation("val", {"multi_line_expr": expr},
                        {"ok": True, "summary": {"ic": 0.005, "icir": 0.08, "factor_coverage": 0.99}})
    # validation 阶段：证据口径预检生效（早期「别浪费提交名额」反馈）
    precheck = r._metric_precheck(expr, stage="validation")
    assert precheck is not None and precheck["verdict"] == "revise"
    assert precheck["source"] == "research_spec_metric_precheck"
    # pre_submit：stage_one 已按 ingest 口径裁决完毕且因子已入池，预检不再运行
    assert r._metric_precheck(expr, stage="pre_submit") is None


# ---------------------------------------------------------------------------
# 回归（2026-09-24）：evaluate_factor 走 legacy 扁平结构时，审查证据 summary 被清空，
# 导致 _metric_precheck 把 train 指标全取 0 → 每次 val 验证必然判 revise →
# verdict 恒为 revise_required（覆盖数值达标的 validated 分支）→「验证通过」永不出现，
# 因子永远不进候选池（整夜 6 run 0 候选的根因）。
# ---------------------------------------------------------------------------

def test_profile_result_for_reviewer_accepts_legacy_flat_shape():
    """legacy 扁平口径（summary 在顶层）必须原样保留，不得清空。"""
    from alphaagent.factor.mining.agent.agentscope_tools import _profile_result_for_reviewer

    legacy = {
        "ok": True,
        "split": "train",
        "summary": {"ic": 0.025, "icir": 0.3291, "factor_coverage": 0.9604},
        "monthly_corr_robustness": {"share_months_ic_positive": 0.83},
        "rule_results": [{"passed": True}],
    }
    out = _profile_result_for_reviewer(legacy)
    assert out["summary"]["ic"] == 0.025
    assert out["summary"]["icir"] == 0.3291
    assert out["summary"]["factor_coverage"] == 0.9604
    assert out["monthly_corr_robustness"]["share_months_ic_positive"] == 0.83
    assert out["rule_results"] == [{"passed": True}]


def test_profile_result_for_reviewer_still_accepts_engine_native_shape():
    """引擎原生口径（指标嵌在 metrics 下）回归保护。"""
    from alphaagent.factor.mining.agent.agentscope_tools import _profile_result_for_reviewer

    native = {
        "ok": True,
        "split": "train",
        "metrics": {
            "cross_sectional_core": {"ic": 0.031, "icir": 0.35, "factor_coverage": 0.95},
            "monthly_robustness": {"share_months_ic_positive": 0.9},
        },
    }
    out = _profile_result_for_reviewer(native)
    assert out["summary"]["ic"] == 0.031
    assert out["monthly_corr_robustness"]["share_months_ic_positive"] == 0.9


def test_profile_result_for_reviewer_tolerates_empty_result():
    """无任何指标字段时返回空 dict，不抛异常。"""
    from alphaagent.factor.mining.agent.agentscope_tools import _profile_result_for_reviewer

    out = _profile_result_for_reviewer({})
    assert out["summary"] == {}
    assert out["monthly_corr_robustness"] == {}


def test_metric_precheck_no_revise_with_legacy_train_evidence():
    """端到端回归：legacy 口径 train 证据 + 数值达标 val → 预检不产出 revise。

    修复前该用例必失败（train summary 被清空 → 四条 revise 理由）。
    """
    import types

    from alphaagent.factor.mining.agent.agentscope_tools import _profile_result_for_reviewer

    r = _mk()
    r.config = types.SimpleNamespace(research_spec={})
    expr = "TS_MEAN($close, 5)"
    # train 证据按 evaluate_factor 真实路径构造（经 _profile_result_for_reviewer 包装）
    r.record_evaluation(
        "train",
        {"multi_line_expr": expr},
        _profile_result_for_reviewer({
            "ok": True,
            "split": "train",
            "summary": {"ic": 0.025, "icir": 0.3291, "factor_coverage": 0.9604},
        }),
    )
    # val 证据按 eval_on_val_set 真实路径构造（原始 result，扁平 summary）
    r.record_evaluation(
        "val",
        {"multi_line_expr": expr},
        {"ok": True, "summary": {"ic": 0.0306, "icir": 0.3732, "factor_coverage": 0.9754}},
    )
    assert r._metric_precheck(expr, stage="validation") is None
