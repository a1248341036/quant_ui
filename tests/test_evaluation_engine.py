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
