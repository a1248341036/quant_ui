# -*- coding: utf-8 -*-
"""反馈层一致性与动态归因测试 (Feedback Consistency & Attribution Tests)."""

from __future__ import annotations

import json
import pytest

from alphaagent.factor.mining.diagnostics import (
    DiagnosticContext,
    DiagnosticMediator,
    PitSuspicionDiagnostic,
    TurnoverDiagnostic,
    NearMissDiagnostic,
    IcirDiagnostic,
)


def test_turnover_block_suppresses_submit_decision():
    """P0 验证：换手率超标时，必须压制普通 submit 提示，并标记 submit_suppressed=True。"""
    ctx = DiagnosticContext(
        expr="RANK(TS_PCTCHANGE($close, 1))",
        result={"ok": True, "split": "train"},
        arguments={},
        passed=True,
        ic=0.03,
        abs_ic=0.03,
        icir=0.35,
        abs_icir=0.35,
        coverage=0.95,
        turnover=0.72,  # 严重超标
    )
    mediator = DiagnosticMediator()
    fields = mediator.arbitrate(ctx)

    assert fields.get("submit_suppressed") is True
    assert "日单边换手 0.72 超标" in fields.get("submit_decision_required", "")
    assert "请勿提交" in fields.get("submit_decision_required", "")
    assert fields.get("diagnostic_verdict") == "RESTRUCTURE_REQUIRED"
    assert fields.get("bottleneck") == "HIGH_TURNOVER"


def test_contradicted_prediction_has_highest_veto():
    """P0-5 验证：预测对账证伪拥有最高否决权，强制覆盖 near_miss 和 submit 提示，禁止调参。"""
    ctx = DiagnosticContext(
        expr="RANK(TS_MEAN($close, 20))",
        result={"ok": False, "split": "train"},
        arguments={},
        passed=False,
        ic=0.018,  # 符合 near_miss 区间
        abs_ic=0.018,
        icir=0.25,
        abs_icir=0.25,
        coverage=0.95,
        turnover=0.25,
        prediction_check={"verdict": "contradicted", "message": "预期单调递增，实测倒U型"},
    )
    mediator = DiagnosticMediator()
    fields = mediator.arbitrate(ctx)

    # 验证 near_miss 被彻底压制
    assert "near_miss_hint" not in fields
    assert "严重证伪" in fields.get("submit_decision_required", "")
    assert "禁止对被证伪的机制结构做微调参数" in fields.get("submit_decision_required", "")
    assert fields.get("diagnostic_verdict") == "REJECT_MECHANISM"
    assert fields.get("bottleneck") == "SHAPE_CONTRADICTED"


def test_pit_warning_only_for_fundamentals():
    """P1-3 验证：|IC| >= 0.045 时，价量因子不提示 PIT 伪影，只有基本面因子才提示。"""
    # 1. 价量因子（高 IC 正常，不应扣 PIT 帽子）
    ctx_price = DiagnosticContext(
        expr="RANK(DIVIDE(SUBTRACT($close, $vwap), $vwap))",
        result={"ok": False, "split": "train", "research_mode": "technical"},
        arguments={},
        passed=False,
        ic=0.048,
        abs_ic=0.048,
        icir=0.40,
        abs_icir=0.40,
        coverage=0.95,
    )
    diag = PitSuspicionDiagnostic()
    assert diag.evaluate(ctx_price) is None

    # 2. 基本面因子（高 IC 提示核查阶梯函数或披露日）
    ctx_funda = DiagnosticContext(
        expr="RANK($funda_roe_ttm)",
        result={"ok": False, "split": "train", "research_mode": "fundamental"},
        arguments={},
        passed=False,
        ic=0.048,
        abs_ic=0.048,
        icir=0.40,
        abs_icir=0.40,
        coverage=0.95,
    )
    op = diag.evaluate(ctx_funda)
    assert op is not None
    assert "PIT 泄漏" in op.message


def test_turnover_attribution_with_session_cache():
    """P1-1 & P1-6 验证：利用会话级自相关基线缓存快速归因换手主要来源。"""
    class FakeSession:
        def get_column_autocorr(self, col: str) -> float:
            # 模拟 volume 抖动剧烈 (0.15)，close 稳定 (0.95)
            if col == "volume":
                return 0.15
            return 0.95

    ctx = DiagnosticContext(
        expr="DIVIDE($close, $volume)",
        result={"ok": True, "split": "train"},
        arguments={},
        passed=True,
        ic=0.025,
        abs_ic=0.025,
        icir=0.32,
        abs_icir=0.32,
        coverage=0.95,
        turnover=0.88,
        session=FakeSession(),
    )
    diag = TurnoverDiagnostic()
    op = diag.evaluate(ctx)
    assert op is not None
    assert "$volume" in op.message
    assert "自相关偏低 0.15" in op.message


def test_feedback_synthesizer_fields():
    """P1-8 验证：统一反馈收口器规范产出 3 个标准收敛字段。"""
    ctx_healthy = DiagnosticContext(
        expr="CS_ZSCORE(TS_MEAN($close, 20))",
        result={"ok": True, "split": "train"},
        arguments={},
        passed=True,
        ic=0.026,
        abs_ic=0.026,
        icir=0.32,
        abs_icir=0.32,
        coverage=0.95,
        turnover=0.20,
    )
    mediator = DiagnosticMediator()
    fields = mediator.arbitrate(ctx_healthy)

    assert fields["diagnostic_verdict"] == "PROCEED_TO_SUBMIT"
    assert fields["bottleneck"] == "NONE"
    assert "建议直接调用 submit_factor" in fields["actionable_guidance"]
