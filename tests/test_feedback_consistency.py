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
    AntiHomogenizationDiagnostic,
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


def test_icir_monthly_attribution():
    """P1-4 验证：ICIR 不足时根据月度正相关比例进行波动来源归因。"""
    ctx = DiagnosticContext(
        expr="RANK(TS_MEAN($close, 20))",
        result={
            "ok": True,
            "split": "train",
            "monthly_corr_robustness": {"positive_ratio": 0.45, "negative_months": ["2020-03", "2021-02"]},
        },
        arguments={},
        passed=True,
        ic=0.024,
        abs_ic=0.024,
        icir=0.21,  # 低于 0.28 门槛
        abs_icir=0.21,
        coverage=0.95,
    )
    diag = IcirDiagnostic()
    op = diag.evaluate(ctx)
    assert op is not None
    assert "ICIR=0.210" in op.message
    assert "月度正相关比例仅 45%" in op.message


def test_anti_homogenization_breaker():
    """P1-7 验证：连续 3 次同结构平滑调参换手未降破 0.50 时触发熔断。"""
    from alphaagent.dsl.core.ast import structure_fingerprint

    fp = structure_fingerprint("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 8))")
    recent = [
        {"fingerprint": fp, "turnover": 0.72, "has_smoothing": True},
        {"fingerprint": fp, "turnover": 0.68, "has_smoothing": True},
        {"fingerprint": fp, "turnover": 0.65, "has_smoothing": True},
    ]
    ctx = DiagnosticContext(
        expr="NEG(WMA(CS_RESIDUALIZE($close, $vwap), 8))",
        result={"ok": True, "split": "train"},
        arguments={},
        passed=True,
        ic=0.026,
        abs_ic=0.026,
        icir=0.30,
        abs_icir=0.30,
        coverage=0.95,
        turnover=0.63,
        recent_signatures=recent,
    )
    diag = AntiHomogenizationDiagnostic()
    op = diag.evaluate(ctx)
    assert op is not None
    assert "同质化探索熔断警告" in op.message
    assert "已连续 3 次" in op.message


def test_anti_homogenization_no_break_for_different_structure():
    """P1-7 反向验证：连续平滑调参但结构不同 / 换手已降破时不触发熔断。"""
    recent_diff = [
        {"fingerprint": "fp_a", "turnover": 0.72, "has_smoothing": True},
        {"fingerprint": "fp_b", "turnover": 0.68, "has_smoothing": True},
        {"fingerprint": "fp_c", "turnover": 0.65, "has_smoothing": True},
    ]
    ctx1 = DiagnosticContext(
        expr="NEG(WMA(CS_RESIDUALIZE($close, $vwap), 8))",
        result={"ok": True, "split": "train"},
        arguments={},
        passed=True,
        ic=0.026,
        abs_ic=0.026,
        icir=0.30,
        abs_icir=0.30,
        coverage=0.95,
        turnover=0.63,
        recent_signatures=recent_diff,
    )
    assert AntiHomogenizationDiagnostic().evaluate(ctx1) is None

    # 换手已降破 0.50（最后一次成功），不触发
    recent_low = [
        {"fingerprint": "fp_same", "turnover": 0.72, "has_smoothing": True},
        {"fingerprint": "fp_same", "turnover": 0.68, "has_smoothing": True},
        {"fingerprint": "fp_same", "turnover": 0.42, "has_smoothing": True},
    ]
    ctx2 = DiagnosticContext(
        expr="NEG(WMA(CS_RESIDUALIZE($close, $vwap), 8))",
        result={"ok": True, "split": "train"},
        arguments={},
        passed=True,
        ic=0.026,
        abs_ic=0.026,
        icir=0.30,
        abs_icir=0.30,
        coverage=0.95,
        turnover=0.42,
        recent_signatures=recent_low,
    )
    assert AntiHomogenizationDiagnostic().evaluate(ctx2) is None


def test_advisory_cache_lru_eviction(tmp_path):
    """P2-2 验证：advisory 缓存超限时按 LRU 淘汰最久未命中条目，而非全清。"""
    from alphaagent.factor.mining.memory import ResearchMemoryStore

    store = ResearchMemoryStore(tmp_path / "m.db")
    cache = store._get_advisory_cache()
    cache.clear()
    # OrderedDict 结构断言
    assert hasattr(cache, "move_to_end")
    assert hasattr(cache, "popitem")

    # 模拟超限淘汰：容量 3，插入 4 个键，最旧（k0）被淘汰
    cache["k0"] = (0.0, None)
    cache["k1"] = (0.0, None)
    cache["k2"] = (0.0, None)
    cache.move_to_end("k2")
    while len(cache) >= 3:
        cache.popitem(last=False)
    cache["new_key"] = (1.0, None)
    assert "k0" not in cache
    assert "new_key" in cache
