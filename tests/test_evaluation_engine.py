from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from alphaagent.factor.evaluation.engine import EvaluationEngine
from alphaagent.factor.evaluation.profile import EvaluationProfile, default_evaluation_profiles
from alphaagent.factor.mining.context import StockEvalContext
from alphaagent.factor.mining.schemas import EvalProfileRequest
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.mining.session import StockEvalSession


def _session() -> StockEvalSession:
    dates = pd.date_range("2024-01-01", periods=16, freq="B")
    instruments = [f"S{i:03d}" for i in range(40)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    cross_section = np.tile(np.linspace(-1, 1, len(instruments)), len(dates))
    drift = np.repeat(np.linspace(0, 0.05, len(dates)), len(instruments))
    close = 10 + cross_section + drift
    panel = pd.DataFrame(
        {
            "adj_close": close,
            "open": close * (1 + 0.001 * cross_section),
            "high": close * 1.01,
            "low": close * 0.99,
            "amount": np.tile(np.linspace(2e7, 8e7, len(instruments)), len(dates)),
            "turnover_rate": np.tile(np.linspace(0.5, 3.0, len(instruments)), len(dates)),
            "float_cap": np.tile(np.linspace(1e8, 5e10, len(instruments)), len(dates)),
            "label_1d_open_to_open": 0.002 * cross_section + 0.0001 * drift,
        },
        index=index,
    )
    return StockEvalSession(
        session_id="test",
        ctx=StockEvalContext(
            panel_path=Path("panel.parquet"),
            train_start="2024-01-01",
            train_end="2024-01-10",
            val_start="2024-01-11",
            val_end="2024-01-31",
        ),
        panel=panel,
    )


def test_default_profiles_execute_dsl_and_emit_evidence() -> None:
    session = _session()
    engine = EvaluationEngine(default_evaluation_profiles())

    raw = engine.evaluate(
        session,
        profile_id="validation",
        multi_line_expr="$adj_close",
        factor_name="price_level",
    )
    neutral = engine.evaluate(
        session,
        profile_id="size_neutral_validation",
        multi_line_expr="$adj_close",
        factor_name="price_level_size_neutral",
    )

    assert raw["ok"] and neutral["ok"]
    assert raw["profile_hash"] != neutral["profile_hash"]
    assert raw["metrics"]["cross_sectional_core"]["n_days"] > 0
    assert "nw_t_ls" in raw["metrics"]["mls_fmb"]
    assert "skewness" in raw["metrics"]["ic_series_diagnostics"]
    assert "net_ir_annual" in raw["metrics"]["long_short_portfolio"]
    assert "size_residualize" in neutral["transforms_applied"]


def test_profile_rules_are_evaluated() -> None:
    session = _session()
    profile = EvaluationProfile(
        profile_id="rule_test",
        split="val",
        metrics=({"plugin": "cross_sectional_core"},),
        rules=({"metric": "cross_sectional_core.ic", "op": "gte", "value": 0.0},),
    )
    result = EvaluationEngine({"rule_test": profile}).evaluate(
        session,
        profile_id="rule_test",
        multi_line_expr="$adj_close",
    )
    assert result["ok"]
    assert result["rule_results"][0]["passed"]


def test_service_profile_evaluation_registers_candidate() -> None:
    session = _session()
    service = StockEvalService(max_parallel_eval=1)
    service.sessions._sessions[session.session_id] = session
    result = service.eval_profile(
        EvalProfileRequest(
            session_id=session.session_id,
            profile_id="validation",
            multi_line_expr="$adj_close",
            factor_name="price_level",
        )
    )
    assert result["candidate"]["candidate_id"].startswith("cand_")
    assert result["candidate_state"] in {"validation_evaluated", "validation_rejected"}


def test_train_eval_and_profile_use_same_engine_pipeline() -> None:
    """收敛验证：eval_train（LLM 旧契约）与 eval_profile train_screen（因子实验室）
    走同一 EvaluationEngine + 同一 transforms，核心指标必须一致。"""
    from alphaagent.factor.mining.schemas import EvalTrainRequest

    session = _session()
    service = StockEvalService(max_parallel_eval=1)
    service.sessions._sessions[session.session_id] = session
    expr = "$adj_close"

    train_result = service.eval_train(
        EvalTrainRequest(
            session_id=session.session_id,
            multi_line_expr=expr,
            factor_name="price_level",
        )
    )
    assert train_result["ok"]

    profile_result = service.eval_profile(
        EvalProfileRequest(
            session_id=session.session_id,
            profile_id="train_screen",
            multi_line_expr=expr,
            factor_name="price_level",
        )
    )
    assert profile_result["ok"]

    # 同一引擎 → 同一 transform 管线：都只 winsorize+zscore，无 size_residualize
    assert profile_result["transforms_applied"] == [
        "cross_sectional_winsorize", "cross_sectional_zscore",
    ]

    train_summary = train_result["summary"]
    prof_metrics = profile_result["metrics"]["cross_sectional_core"]
    # eval_train 经 format_eval_response 四舍五入到 4 位；eval_profile 为原始值。
    # 收敛要求：同一引擎计算，舍入后的 LLM 契约值与原始值一致。
    for key in ("ic", "icir", "rank_ic", "n_days", "n_instruments",
                "factor_coverage", "cs_pearson_autocorr"):
        expected = round(float(prof_metrics[key]), 4)
        actual = float(train_summary[key])
        assert abs(expected - actual) < 1e-9, f"{key}: {expected} vs {actual}"

    # 同一引擎 → 同一月度稳健性（monthly_corr_robustness 经 4 位舍入）
    expected_monthly = round(float(profile_result["metrics"]["monthly_robustness"]["mean_monthly_ic"]), 4)
    assert abs(expected_monthly - float(train_result["monthly_corr_robustness"]["mean_monthly_ic"])) < 1e-9


def test_base_transforms_do_not_include_size_residualize() -> None:
    """三轨主口径：主 profile 不市值残差化；市值中性只在诊断 profile。"""
    profiles = default_evaluation_profiles()
    base = [t["plugin"] for t in profiles["train_screen"].transforms]
    assert "size_residualize" not in base
    neutral = [t["plugin"] for t in profiles["size_neutral_validation"].transforms]
    assert "size_residualize" in neutral


def test_size_neutral_decay_diagnostic_present() -> None:
    """三轨诊断：主评估输出含 size_neutral_decay（不参与门槛的辅助字段）。"""
    session = _session()
    engine = EvaluationEngine(default_evaluation_profiles())
    raw = engine.evaluate(
        session,
        profile_id="train_screen",
        multi_line_expr="$adj_close",
    )
    diag = raw["metrics"]["size_neutral_decay"]
    assert "size_neutral_ic" in diag
    assert "size_neutral_abs_ic_decay" in diag
    # 主口径 metrics 里的规则判定不依赖该字段
    assert all("size_neutral" not in str(rule.get("metric", ""))
               for rule in raw["rule_results"])


def test_evaluate_include_charts_false_skips_visualization_only() -> None:
    """include_charts=False（挖掘批量评估路径）跳过图表数据，metrics 保持一致。"""
    session = _session()
    engine = EvaluationEngine(default_evaluation_profiles())
    expr = "$adj_close"

    with_charts = engine.evaluate(
        session, profile_id="train_screen", multi_line_expr=expr,
        include_charts=True,
    )
    without_charts = engine.evaluate(
        session, profile_id="train_screen", multi_line_expr=expr,
        include_charts=False,
    )

    assert with_charts["ok"] and without_charts["ok"]
    assert without_charts["chart_data"] is None
    assert isinstance(with_charts["chart_data"], dict) and with_charts["chart_data"]
    def _metrics_eq(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(_metrics_eq(a[k], b[k]) for k in a)
        if isinstance(a, float) and isinstance(b, float):
            return a == b or (np.isnan(a) and np.isnan(b))
        return a == b

    assert _metrics_eq(without_charts["metrics"], with_charts["metrics"])
    assert without_charts["passed"] == with_charts["passed"]
    assert without_charts["timing_ms"]["total_ms"] <= with_charts["timing_ms"]["total_ms"] + 50.0
