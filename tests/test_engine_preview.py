# -*- coding: utf-8 -*-
"""引擎预演（engine_preview）回归：train 过线因子自动附 val 窗口可交易口径。

用户设计意图（2026-09-11）：统计口径（分位组前瞻收益）与实盘口径（core.engine
完整约束）必须对齐——此前引擎只在 submit 时首次出现，LLM 整个迭代期都在用统计
口径选方向。方案 A：train 过线因子在两段式全量诊断后自动跑 val 窗口引擎预演，
直接复用 run_engine_gate（单一真源）。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _tiny_panel(n_days: int = 40, n_inst: int = 8) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2022-01-03", periods=n_days, freq="B"), [f"{i:06d}" for i in range(n_inst)]],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(3)
    cols = {c: rng.random(len(idx)).astype("float32") for c in
            ("open", "high", "low", "close", "adj_close", "adj_vwap", "volume", "amount",
             "turnover_rate", "float_cap")}
    df = pd.DataFrame(cols, index=idx)
    df["adj_open"] = df["open"]
    df["adj_high"] = df["high"]
    df["adj_low"] = df["low"]
    return df.sort_index()


def _service_with_session(panel, policy):
    from alphaagent.factor.mining.eval.service import StockEvalService

    svc = StockEvalService.__new__(StockEvalService)
    svc.sessions = SimpleNamespace()
    svc.sessions.get = lambda sid: _FakeSession(panel, policy)
    svc._eval_semaphore = __import__("threading").Semaphore(1)
    svc.evaluation_engine = None
    return svc


class _FakeSession:
    def __init__(self, panel, policy):
        self.panel = panel
        self.ctx = SimpleNamespace(
            engine_gate_policy=policy,
            val_start="2022-02-01",
            val_end="2022-03-15",
            asset_type="stock",
        )
        self.session_id = "s1"


_POLICY = {
    "enabled": True, "freq": "monthly", "selection_mode": "top_n", "top_n": 3,
    "capital": 100000.0, "slippage_bps": 0.0, "max_participation": 0.1,
    "min_am20_yuan": 0.0, "min_excess_annual": 0.03, "min_excess_sharpe": 0.5,
    "max_drawdown": 0.4, "min_daily_overlap": 0.5, "min_invested_ratio": 0.8,
}


class TestEnginePreview:
    def _run(self, svc, expr, result, policy=_POLICY):
        session = svc.sessions.get("s1")
        session.ctx.engine_gate_policy = policy
        svc._maybe_engine_preview(session, expr, result)
        return result

    def test_attaches_preview_for_passing_factor(self):
        svc = _service_with_session(_tiny_panel(), _POLICY)
        result = {"ok": True, "passed": True, "metrics": {"cross_sectional_core": {"ic": 0.03}}}
        svc._maybe_engine_preview(_FakeSession(_tiny_panel(), _POLICY), "RANK($adj_close)", result)
        prev = result.get("engine_preview")
        assert prev is not None and "error" not in prev
        assert prev["freq"] == "monthly"
        assert prev["window"]["start"] == "2022-02-01"
        assert prev["passed"] in (True, False)  # 真实引擎输出
        assert "annual_return" in prev and "excess_annual" in prev
        assert "可交易口径" in prev["note"]

    def test_sign_from_train_ic(self, monkeypatch):
        """direction 取 train IC 符号：负 IC 因子以 -1 方向预演。"""
        seen = {}

        def fake_gate(panel, values, *, val_start, val_end, direction=1, policy=None, **kw):
            seen["direction"] = direction
            return {"passed": False, "fail_reasons": [], "metrics": {}}

        import alphaagent.factor.mining.engine_gate as eg
        monkeypatch.setattr(eg, "run_engine_gate", fake_gate)
        svc = _service_with_session(_tiny_panel(), _POLICY)
        result = {"ok": True, "metrics": {"cross_sectional_core": {"ic": -0.05}}}
        svc._maybe_engine_preview(_FakeSession(_tiny_panel(), _POLICY), "RANK($adj_close)", result)
        assert seen["direction"] == -1
        assert result["engine_preview"]["fail_reasons"] == []

    def test_skipped_when_policy_missing_or_disabled(self):
        svc = _service_with_session(_tiny_panel(), None)
        result = {"ok": True, "passed": True}
        svc._maybe_engine_preview(_FakeSession(_tiny_panel(), None), "RANK($adj_close)", result)
        assert "engine_preview" not in result
        result2 = {"ok": True}
        svc._maybe_engine_preview(
            _FakeSession(_tiny_panel(), {"enabled": False}), "RANK($adj_close)", result2
        )
        assert "engine_preview" not in result2

    def test_exception_becomes_error_field_not_crash(self, monkeypatch):
        import alphaagent.dsl as dsl_mod
        monkeypatch.setattr(dsl_mod, "eval_factor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        svc = _service_with_session(_tiny_panel(), _POLICY)
        result = {"ok": True}
        svc._maybe_engine_preview(_FakeSession(_tiny_panel(), _POLICY), "RANK($adj_close)", result)
        prev = result["engine_preview"]
        assert "error" in prev and "boom" in prev["error"]

    def test_two_stage_attaches_preview(self, monkeypatch):
        """集成：两段式 lite 过线 → 全量 → 预演附加到最终响应。"""
        import threading

        from alphaagent.factor.mining.eval.service import StockEvalService
        from alphaagent.factor.evaluation.profile import EvaluationProfile

        panel = _tiny_panel()
        session = _FakeSession(panel, _POLICY)

        engine = SimpleNamespace()
        calls = []

        def fake_evaluate(session, *, profile_id, multi_line_expr, factor_name,
                          label_quantile_n, include_detail_tables, include_charts):
            calls.append(profile_id)
            ic = 0.03
            return {
                "ok": True,
                "metrics": {"cross_sectional_core": {"ic": ic, "icir": 0.4, "factor_coverage": 0.9}},
                "profile": {"profile_id": profile_id},
            }

        engine.evaluate = fake_evaluate
        engine.profile = lambda pid: EvaluationProfile(
            profile_id=pid, split="train", transforms=[], metrics=[],
            rules=[{"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": 0.02}],
        )

        svc = StockEvalService.__new__(StockEvalService)
        svc.sessions = SimpleNamespace(get=lambda sid: session)
        svc._eval_semaphore = threading.Semaphore(1)
        svc.evaluation_engine = engine

        from alphaagent.factor.mining.eval.schemas import EvalTrainRequest

        resp = svc.eval_train(EvalTrainRequest(session_id="s1", multi_line_expr="RANK($adj_close)"))
        assert calls == ["train_screen_lite", "train_screen"]
        assert resp["screen_stage"] == "full"
        assert "engine_preview" in resp and "error" not in resp["engine_preview"]
