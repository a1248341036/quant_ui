# -*- coding: utf-8 -*-
"""换手率预筛单元测试：mock 引擎返回，验证 _eval_train_two_stage 短路逻辑。

不依赖真实 panel/DSL，直接 mock evaluation_engine.evaluate 的返回值，
确认 lite 过线 + 高换手 → screen_stage="turnover_rejected"。
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd

from alphaagent.factor.evaluation.profile import resolve_profiles
from alphaagent.factor.mining.eval.schemas import EvalTrainRequest
from alphaagent.factor.mining.eval.service import StockEvalService
from alphaagent.factor.mining.eval.session import StockEvalSession
from alphaagent.factor.mining.context import StockEvalContext


def _make_service():
    profiles = resolve_profiles({})
    service = StockEvalService(max_parallel_eval=1, profiles=profiles)
    # 空 panel，不会真正跑引擎（mock 掉）
    idx = pd.MultiIndex.from_product(
        [["2021-01-04"], ["S000"]], names=["datetime", "instrument"]
    )
    panel = pd.DataFrame(
        {"label_1d_close_to_close": [0.0], "close": [100.0], "float_cap": [1e6]},
        index=idx,
    )
    ctx = StockEvalContext(panel_path="unused://", label_col="label_1d_close_to_close")
    session = StockEvalSession(session_id="mock", ctx=ctx, panel=panel)
    service.sessions._sessions[session.session_id] = session
    return service, session


def _lite_pass_result():
    """lite 过线的结果（IC/ICIR/coverage 全过门槛）。"""
    return {
        "ok": True,
        "profile_id": "train_screen_lite",
        "split": "train",
        "metrics": {
            "cross_sectional_core": {
                "ic": 0.030,
                "icir": 0.35,
                "rank_ic": 0.04,
                "factor_coverage": 0.93,
                "n_days": 300,
                "n_instruments": 60,
                "decile_mean_label": [{"decile": d, "mean_label": 0.001 * d} for d in range(1, 11)],
            }
        },
        "timing_ms": {"dsl_eval_ms": 1, "transforms_ms": 1, "metrics_ms": 100, "total_ms": 102},
    }


def _turnover_result(turnover: float):
    """train_screen_turnover profile 的结果。"""
    return {
        "ok": True,
        "profile_id": "train_screen_turnover",
        "split": "train",
        "metrics": {
            "quantile_portfolio": {
                "avg_daily_side_turnover": turnover,
                "avg_rebalance_side_turnover": turnover * 0.8,
                "top_group_annualized_return": 0.15,
                "top_group_annualized_excess_return": 0.08,
                "direction": 1,
            }
        },
        "timing_ms": {"dsl_eval_ms": 1, "transforms_ms": 1, "metrics_ms": 200, "total_ms": 202},
    }


def _full_result():
    """全量 train_screen 结果。"""
    return {
        "ok": True,
        "profile_id": "train_screen",
        "split": "train",
        "metrics": {
            "cross_sectional_core": {
                "ic": 0.030,
                "icir": 0.35,
                "rank_ic": 0.04,
                "factor_coverage": 0.93,
                "n_days": 300,
                "n_instruments": 60,
                "decile_mean_label": [{"decile": d, "mean_label": 0.001 * d} for d in range(1, 11)],
            },
            "quantile_portfolio": {
                "avg_daily_side_turnover": 0.3,
                "top_group_annualized_return": 0.15,
            },
        },
        "timing_ms": {"dsl_eval_ms": 1, "transforms_ms": 1, "metrics_ms": 5000, "total_ms": 5002},
    }


def test_high_turnover_short_circuited():
    """lite 过线 + 换手率 > 0.5 → screen_stage=turnover_rejected，不跑全量。"""
    os.environ["ALPHA_EVAL_TURNOVER_PRESCREEN"] = "1"
    service, session = _make_service()

    # mock 引擎：第 1 次调 lite（过线），第 2 次调 turnover（0.89 超标），不应有第 3 次
    call_count = {"n": 0}

    def fake_evaluate(_session, *, profile_id, **kwargs):
        call_count["n"] += 1
        if profile_id == "train_screen_lite":
            return _lite_pass_result()
        elif profile_id == "train_screen_turnover":
            return _turnover_result(0.89)
        elif profile_id == "train_screen":
            return _full_result()
        return {"ok": False, "error": f"unexpected profile {profile_id}"}

    with patch.object(service.evaluation_engine, "evaluate", side_effect=fake_evaluate):
        req = EvalTrainRequest(
            session_id=session.session_id,
            multi_line_expr="f = TEST\nf",
            factor_name="high_to",
            include_detail_tables=False,
            label_quantile_n=10,
        )
        resp = service.eval_train(req)

    assert resp.get("ok"), resp.get("error")
    assert resp.get("screen_stage") == "turnover_rejected", f"expected turnover_rejected, got {resp.get('screen_stage')}"
    assert resp.get("avg_daily_side_turnover") == 0.89, f"turnover={resp.get('avg_daily_side_turnover')}"
    # 只调了 lite + turnover，没调全量
    assert call_count["n"] == 2, f"expected 2 engine calls, got {call_count['n']}"
    print("PASS: high turnover short-circuited, full eval skipped")


def test_low_turnover_proceeds_to_full():
    """lite 过线 + 换手率 <= 0.5 → 跑全量。"""
    os.environ["ALPHA_EVAL_TURNOVER_PRESCREEN"] = "1"
    service, session = _make_service()

    call_count = {"n": 0}

    def fake_evaluate(_session, *, profile_id, **kwargs):
        call_count["n"] += 1
        if profile_id == "train_screen_lite":
            return _lite_pass_result()
        elif profile_id == "train_screen_turnover":
            return _turnover_result(0.3)  # 低换手
        elif profile_id == "train_screen":
            return _full_result()
        return {"ok": False, "error": f"unexpected profile {profile_id}"}

    with patch.object(service.evaluation_engine, "evaluate", side_effect=fake_evaluate):
        req = EvalTrainRequest(
            session_id=session.session_id,
            multi_line_expr="f = TEST\nf",
            factor_name="low_to",
            include_detail_tables=False,
            label_quantile_n=10,
        )
        resp = service.eval_train(req)

    assert resp.get("ok"), resp.get("error")
    assert resp.get("screen_stage") == "full", f"expected full, got {resp.get('screen_stage')}"
    # 调了 lite + turnover + full
    assert call_count["n"] == 3, f"expected 3 engine calls, got {call_count['n']}"
    print("PASS: low turnover proceeded to full eval")


def test_prescreen_disabled():
    """ALPHA_EVAL_TURNOVER_PRESCREEN=0 → 不跑换手预筛，lite 过线直接全量。"""
    os.environ["ALPHA_EVAL_TURNOVER_PRESCREEN"] = "0"
    service, session = _make_service()

    call_count = {"n": 0}

    def fake_evaluate(_session, *, profile_id, **kwargs):
        call_count["n"] += 1
        if profile_id == "train_screen_lite":
            return _lite_pass_result()
        elif profile_id == "train_screen_turnover":
            assert False, "turnover prescreen should not run when disabled"
        elif profile_id == "train_screen":
            return _full_result()
        return {"ok": False, "error": f"unexpected profile {profile_id}"}

    with patch.object(service.evaluation_engine, "evaluate", side_effect=fake_evaluate):
        req = EvalTrainRequest(
            session_id=session.session_id,
            multi_line_expr="f = TEST\nf",
            factor_name="no_prescreen",
            include_detail_tables=False,
            label_quantile_n=10,
        )
        resp = service.eval_train(req)

    assert resp.get("ok"), resp.get("error")
    assert resp.get("screen_stage") == "full", f"expected full, got {resp.get('screen_stage')}"
    # 只调了 lite + full，没调 turnover
    assert call_count["n"] == 2, f"expected 2 engine calls, got {call_count['n']}"
    print("PASS: prescreen disabled, lite → full directly")


def test_turnover_boundary_050():
    """换手率 = 0.5（边界，不 > 0.5）→ 不短路，跑全量。"""
    os.environ["ALPHA_EVAL_TURNOVER_PRESCREEN"] = "1"
    service, session = _make_service()

    def fake_evaluate(_session, *, profile_id, **kwargs):
        if profile_id == "train_screen_lite":
            return _lite_pass_result()
        elif profile_id == "train_screen_turnover":
            return _turnover_result(0.5)  # 边界值
        elif profile_id == "train_screen":
            return _full_result()
        return {"ok": False, "error": f"unexpected profile {profile_id}"}

    with patch.object(service.evaluation_engine, "evaluate", side_effect=fake_evaluate):
        req = EvalTrainRequest(
            session_id=session.session_id,
            multi_line_expr="f = TEST\nf",
            factor_name="boundary",
            include_detail_tables=False,
            label_quantile_n=10,
        )
        resp = service.eval_train(req)

    assert resp.get("ok"), resp.get("error")
    assert resp.get("screen_stage") == "full", f"turnover=0.5 should not short-circuit, got {resp.get('screen_stage')}"
    print("PASS: turnover=0.5 boundary not short-circuited")


def test_turnover_rejected_triggers_diagnostic():
    """turnover_rejected 响应应触发换手率诊断器 block_submit（ctx.passed=True）。"""
    from alphaagent.factor.mining.diagnostics import (
        DiagnosticContext,
        TurnoverDiagnostic,
    )

    # 模拟 turnover_rejected 响应
    result = {
        "ok": True,
        "screen_stage": "turnover_rejected",
        "avg_daily_side_turnover": 0.89,
        "summary": {"ic": 0.030, "icir": 0.35, "factor_coverage": 0.93},
    }
    ctx = DiagnosticContext.from_eval_result(
        result, expr="f = TS_RANK($ret, 5)\nNEG(f)", arguments={}
    )
    assert ctx.passed is True, f"turnover_rejected should set passed=True, got {ctx.passed}"
    assert ctx.turnover == 0.89, f"turnover should be 0.89, got {ctx.turnover}"
    assert ctx.split == "train", f"split should be train, got {ctx.split}"

    diag = TurnoverDiagnostic()
    opinion = diag.evaluate(ctx)
    assert opinion is not None, "TurnoverDiagnostic should trigger for turnover_rejected"
    assert opinion.opinion_type == "block_submit", f"expected block_submit, got {opinion.opinion_type}"
    assert "请勿提交" in opinion.message or "请勿直接提交" in opinion.message, f"message should warn against submit: {opinion.message[:80]}"
    print("PASS: turnover_rejected triggers TurnoverDiagnostic block_submit")


if __name__ == "__main__":
    test_high_turnover_short_circuited()
    test_low_turnover_proceeds_to_full()
    test_prescreen_disabled()
    test_turnover_boundary_050()
    test_turnover_rejected_triggers_diagnostic()
    print("\nALL TESTS PASSED")
