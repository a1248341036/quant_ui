# -*- coding: utf-8 -*-
"""分箱塌缩防御三阶段验证测试 (Decile Collapse Guard Tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.dsl.core.operators import GATED_SIGNAL, PIECEWISE_STATE
from alphaagent.factor.mining.delivery.delivery_checker import DeliveryChecker
from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria
from scripts.diagnose_decile_collapse import diagnose_entry


def test_stage_one_rejects_decile_collapse_bins():
    """阶段二测试：decile_mean_label 组数 < 8 时被 stage_one_stats 硬拦截。"""
    criteria = DeliveryCriteria.defaults()
    checker = DeliveryChecker(criteria)

    # 模拟只有 5 个分箱（塌缩）的因子
    metrics_collapsed = {
        "ic": 0.030,
        "icir": 0.40,
        "factor_coverage": 0.95,
        "cs_pearson_autocorr": 0.50,
        "decile_mean_label": [{"decile": i + 1, "mean_label": 0.001 * (i + 1)} for i in range(5)],
        "quantile_portfolio": {"avg_daily_side_turnover": 0.25, "collapse_ratio": 0.0},
    }

    res = checker.stage_one_stats(metrics_collapsed)
    assert not res.passed
    assert any("decile_bins=5 < 8" in r for r in res.fail_reasons)


def test_stage_one_rejects_high_collapse_ratio():
    """阶段二测试：collapse_ratio > 0.30 时被 stage_one_stats 硬拦截。"""
    criteria = DeliveryCriteria.defaults()
    checker = DeliveryChecker(criteria)

    # 组数虽然有 10 个，但 collapse_ratio 达到 0.60
    metrics_high_collapse = {
        "ic": 0.030,
        "icir": 0.40,
        "factor_coverage": 0.95,
        "cs_pearson_autocorr": 0.50,
        "decile_mean_label": [{"decile": i + 1, "mean_label": 0.001 * (i + 1)} for i in range(10)],
        "quantile_portfolio": {"avg_daily_side_turnover": 0.25, "collapse_ratio": 0.60},
    }

    res = checker.stage_one_stats(metrics_high_collapse)
    assert not res.passed
    assert any("decile_collapse_ratio=0.60 > 0.30" in r for r in res.fail_reasons)


def test_stage_one_passes_healthy_deciles():
    """阶段二测试：组数完整 (10) 且 collapse_ratio <= 0.30 时分箱检查通过。"""
    criteria = DeliveryCriteria.defaults()
    checker = DeliveryChecker(criteria)

    metrics_healthy = {
        "ic": 0.030,
        "icir": 0.40,
        "factor_coverage": 0.95,
        "cs_pearson_autocorr": 0.50,
        "decile_mean_label": [{"decile": i + 1, "mean_label": 0.001 * (i + 1)} for i in range(10)],
        "quantile_portfolio": {"avg_daily_side_turnover": 0.25, "collapse_ratio": 0.05},
    }

    res = checker.stage_one_stats(metrics_healthy)
    assert res.passed
    assert len(res.fail_reasons) == 0


def test_gated_signal_threshold_bounds():
    """阶段三测试：GATED_SIGNAL 的 threshold > 0.85 抛出异常。"""
    idx = pd.MultiIndex.from_tuples(
        [("2026-01-05", "000001.SZ"), ("2026-01-05", "000002.SZ")],
        names=["datetime", "instrument"],
    )
    df_sig = pd.DataFrame({"signal": [1.0, 2.0]}, index=idx)
    df_state = pd.DataFrame({"state": [0.1, 0.9]}, index=idx)

    # 极端 threshold=0.90 抛异常
    with pytest.raises(ValueError, match="GATED_SIGNAL threshold 须在 \\[0.5, 0.85\\]"):
        GATED_SIGNAL(df_sig, df_state, threshold=0.90)

    # 合规 threshold=0.75 正常执行
    out = GATED_SIGNAL(df_sig, df_state, threshold=0.75)
    assert len(out) == 2


def test_piecewise_state_mid_interval_bounds():
    """阶段三测试：PIECEWISE_STATE 中间常数区间 > 0.70 抛出异常。"""
    idx = pd.MultiIndex.from_tuples(
        [("2026-01-05", "000001.SZ"), ("2026-01-05", "000002.SZ")],
        names=["datetime", "instrument"],
    )
    df_sig = pd.DataFrame({"signal": [1.0, 2.0]}, index=idx)
    df_state = pd.DataFrame({"state": [0.1, 0.9]}, index=idx)

    # 中间常数区 0.9 - 0.1 = 0.80 > 0.70，抛异常
    with pytest.raises(ValueError, match="PIECEWISE_STATE 中间常数区占比过大"):
        PIECEWISE_STATE(df_sig, df_state, low_q=0.1, high_q=0.9)

    # 合规区间 0.75 - 0.25 = 0.50，正常执行
    out = PIECEWISE_STATE(df_sig, df_state, low_q=0.25, high_q=0.75)
    assert len(out) == 2


def test_diagnose_script_detects_collapse():
    """阶段一测试：诊断脚本函数对塌缩因子的判定准确性。"""
    entry_bad = {
        "name": "bad_gate",
        "expr": "GATED_SIGNAL(RANK($close), RANK($volume), 0.9)",
        "metrics": {
            "decile_mean_label": [{"decile": 1}, {"decile": 2}],
            "quantile_portfolio": {"collapse_ratio": 1.0},
        },
    }
    diag = diagnose_entry("bad_gate", entry_bad, min_bins=8, max_collapse_ratio=0.30)
    assert diag["is_collapsed"] is True
    assert diag["n_bins"] == 2
    assert "GATED_SIGNAL" in diag["operators"]

    entry_good = {
        "name": "good_factor",
        "expr": "CS_ZSCORE(TS_MEAN($close, 20))",
        "metrics": {
            "decile_mean_label": [{"decile": i + 1} for i in range(10)],
            "quantile_portfolio": {"collapse_ratio": 0.0},
        },
    }
    diag_good = diagnose_entry("good_factor", entry_good, min_bins=8, max_collapse_ratio=0.30)
    assert diag_good["is_collapsed"] is False
    assert diag_good["n_bins"] == 10


# ─────────────────────────────────────────────────────────────────────────────
# 2026-09-27 根因治理：常数簇标定 / SOFT_GATE / 预检硬拦 / 探索期短路
# 详见 docs/specs/alphaagent_decile_collapse_root_fix_spec.md
# ─────────────────────────────────────────────────────────────────────────────


def _cs_panel(n: int = 3000, seed: int = 0):
    """构造标准截面面板（信号 ~N(0,1)，state ~U(0,1)）。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_tuples(
        [("2026-01-05", f"{i:06d}.SZ") for i in range(n)],
        names=["datetime", "instrument"],
    )
    return (
        pd.DataFrame({"s": rng.standard_normal(n)}, index=idx),
        pd.DataFrame({"t": rng.random(n)}, index=idx),
    )


def _qcut_bins(df: pd.DataFrame, n_groups: int = 10) -> int:
    codes = pd.qcut(df.iloc[:, 0], n_groups, labels=False, duplicates="drop")
    return int(len(pd.unique(codes.dropna())))


def test_hard_gate_collapses_at_every_legal_threshold():
    """合法 threshold 区间 [0.5, 0.85] 内，GATED_SIGNAL 组数全部 < 8（定理级证据）。"""
    sig, state = _cs_panel()
    for thr, expected_max in ((0.5, 6), (0.6, 5), (0.8, 3), (0.85, 2)):
        k = _qcut_bins(GATED_SIGNAL(sig, state, threshold=thr, high_state=True, neutral=0.0))
        assert k == expected_max, f"threshold={thr} 期望 {expected_max} 组，实测 {k}"
        assert k < 8, f"threshold={thr} 组数 {k} 竟 >= 8，标定失效"


def test_zscore_cannot_rescue_constant_cluster():
    """CS_ZSCORE 是单调变换，无法打散常数簇（旧版 rule 14 的核心错误）。"""
    from alphaagent.dsl.core.operators import CS_ZSCORE

    sig, state = _cs_panel()
    raw = GATED_SIGNAL(sig, state, threshold=0.8, high_state=True, neutral=0.0)
    wrapped = CS_ZSCORE(raw)
    assert _qcut_bins(raw) == _qcut_bins(wrapped) < 8


def test_soft_gate_never_collapses():
    """SOFT_GATE 在全部 strength 下均保持 10 组（连续加权，零常数簇）。"""
    from alphaagent.dsl.core.operators import SOFT_GATE

    sig, state = _cs_panel()
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert _qcut_bins(SOFT_GATE(sig, state, w)) == 10, f"strength={w} 发生塌缩"
    assert _qcut_bins(SOFT_GATE(sig, state, 0.5, high_state=False)) == 10


def test_soft_gate_strength_bounds():
    """strength 越界抛错，边界值可用。"""
    from alphaagent.dsl.core.operators import SOFT_GATE

    sig, state = _cs_panel(n=10)
    with pytest.raises(ValueError, match="SOFT_GATE strength 须在"):
        SOFT_GATE(sig, state, 1.5)
    with pytest.raises(ValueError, match="SOFT_GATE strength 须在"):
        SOFT_GATE(sig, state, -0.1)
    assert len(SOFT_GATE(sig, state, 0.0)) == 10
    assert len(SOFT_GATE(sig, state, 1.0)) == 10


def test_soft_gate_semantics_high_low_state():
    """high_state=True 时高 state 端权重大；False 时相反。"""
    import numpy as np
    from alphaagent.dsl.core.operators import SOFT_GATE

    n = 200
    idx = pd.MultiIndex.from_tuples(
        [("2026-01-05", f"{i:06d}.SZ") for i in range(n)],
        names=["datetime", "instrument"],
    )
    sig = pd.DataFrame({"s": np.ones(n)}, index=idx)
    state = pd.DataFrame({"t": np.linspace(0.0, 1.0, n)}, index=idx)

    hi = SOFT_GATE(sig, state, 0.5, high_state=True).iloc[:, 0].to_numpy()
    lo = SOFT_GATE(sig, state, 0.5, high_state=False).iloc[:, 0].to_numpy()
    assert hi[0] < hi[-1] and lo[0] > lo[-1]
    # strength=0.5 → 权重理论区间 [0.5, 1.5]；RANK(pct) 取 (1/n, 1] → 实测 [0.505, 1.5]
    assert hi.min() == pytest.approx(0.505, abs=1e-6)
    assert hi.max() == pytest.approx(1.5, abs=1e-6)
    # 低状态门控是中位镜像：同一 state 逐元素 lo = 2 - hi
    assert lo[-1] == pytest.approx(2.0 - hi[-1], abs=1e-6)
    assert lo[0] == pytest.approx(2.0 - hi[0], abs=1e-6)
    assert lo[-1] == pytest.approx(0.5, abs=1e-6)


def test_precheck_blocks_any_hard_gate():
    """预检对任意合法 threshold 的硬门控均硬拦（旧版只在 >=0.8 报警）。"""
    from alphaagent.factor.mining.tools._precheck import precheck_expression

    for thr in (0.5, 0.6, 0.8, 0.85):
        res = precheck_expression(f"CS_ZSCORE(GATED_SIGNAL($ret, RANK($volume), {thr}))")["result"]
        assert res["blocked"] is True, f"threshold={thr} 未被拦截"
        assert any(r["kind"] == "gate_collapse" and r["blocked"] for r in res["risks"])


def test_precheck_passes_soft_gate_and_group_rank():
    """SOFT_GATE / CS_GROUP_RANK 不被拦截。"""
    from alphaagent.factor.mining.tools._precheck import precheck_expression

    for expr in (
        "CS_ZSCORE(SOFT_GATE($ret, RANK($volume), 0.5))",
        "CS_GROUP_RANK($ret, CS_BUCKET(RANK($volume), 5))",
    ):
        res = precheck_expression(expr)["result"]
        assert res["blocked"] is False, f"{expr} 被误拦"
        assert res["risk_count"] == 0


def test_precheck_piecewise_boundary():
    """PIECEWISE_STATE 中间区 > 0.3 硬拦，= 0.3 仅警告（临界点）。"""
    from alphaagent.factor.mining.tools._precheck import precheck_expression

    over = precheck_expression("PIECEWISE_STATE($ret, $volume, 0.3, 0.7, -1, 1, 0.0)")["result"]
    assert over["blocked"] is True
    edge = precheck_expression("PIECEWISE_STATE($ret, $volume, 0.35, 0.65, -1, 1, 0.0)")["result"]
    assert edge["blocked"] is False


def test_precheck_block_switch_disables_hard_block(monkeypatch):
    """逃生阀：ALPHA_PRECHECK_BLOCK_COLLAPSE=0 时降级为仅警告。"""
    monkeypatch.setenv("ALPHA_PRECHECK_BLOCK_COLLAPSE", "0")
    from alphaagent.factor.mining.tools import _precheck

    res = _precheck.precheck_expression("GATED_SIGNAL($ret, RANK($volume), 0.8)")["result"]
    assert res["blocked"] is False
    assert res["max_risk"] == "high"  # 风险仍然报告，只是不硬拦


def test_evaluate_path_auto_blocks_collapse_structure():
    """evaluate_factor 路径内联预检（不再依赖 LLM 主动调用 precheck_expression）。"""
    from alphaagent.factor.mining.agent.agentscope_tools import _preflight_check

    blocked = _preflight_check("CS_ZSCORE(GATED_SIGNAL($ret, RANK($volume), 0.8))", "f1")
    assert blocked is not None and blocked.get("blocked") is True
    assert "塌缩" in blocked["warning"]
    assert "SOFT_GATE" in blocked["suggestion"]

    # 连续结构不被拦
    assert _preflight_check("CS_ZSCORE(SOFT_GATE($ret, RANK($volume), 0.5))", "f2") is None


def test_diagnostic_attributes_collapse_to_gate_not_smoothing():
    """诊断层归因修正：含门控时指向门控，而非误导性地说"去掉末位平滑"。"""
    from alphaagent.factor.mining.diagnostics import (
        DecileCollapseDiagnostic,
        DiagnosticContext,
    )

    expr = "b = RANK($volume)\ng = GATED_SIGNAL($ret, b, 0.8)\nCS_ZSCORE(g)"
    diag = DecileCollapseDiagnostic()
    opinion = diag.evaluate(DiagnosticContext(expr=expr, result={}, arguments={}))
    assert opinion is not None
    assert opinion.extra_fields.get("collapse_risk_source") == "gate_collapse"
    assert "SOFT_GATE" in opinion.message
    assert "改用 CS_ZSCORE" not in opinion.message


def test_collapse_prescreen_short_circuits_before_full_eval():
    """探索期短路：train_screen_turnover 已带 collapse_ratio，超阈值即短路（零额外算力）。"""
    from alphaagent.factor.mining.eval import service as svc

    assert svc._COLLAPSE_GATE_LIMIT == 0.30  # 与 delivery_checker 同源
    assert svc._collapse_prescreen_enabled() is True


def test_collapse_prescreen_switch(monkeypatch):
    monkeypatch.setenv("ALPHA_EVAL_COLLAPSE_PRESCREEN", "0")
    from alphaagent.factor.mining.eval import service as svc

    assert svc._collapse_prescreen_enabled() is False
