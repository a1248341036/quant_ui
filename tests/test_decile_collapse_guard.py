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
