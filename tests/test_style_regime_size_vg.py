# -*- coding: utf-8 -*-
"""状态层运行时（style_regime）确定性/无前视/缺失语义测试（Phase 1，spec §3.3）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.rotation.compiler import compile_design
from core.rotation.style_regime import (
    boost_for_state,
    canonical_design,
    compute_states,
)


def _signal_frame(n: int = 400) -> pd.DataFrame:
    """单调递增双维信号：暖机后状态稳定为 LARGE_VALUE（size H + vg H）。"""
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    size = np.cumsum(np.linspace(0.02, 0.04, n))
    vg = np.cumsum(np.linspace(0.01, 0.03, n))
    return pd.DataFrame({"size_proxy": size, "value_growth_spread": vg}, index=idx)


def _compiled() -> object:
    return compile_design(canonical_design(), mode="technical")


class TestNoLookahead:
    def test_last_day_signal_does_not_affect_state_series(self) -> None:
        c = _compiled()
        sig = _signal_frame()
        full = compute_states(c, sig)
        alt = sig.copy()
        alt.iloc[-1] = alt.iloc[-1] * 100 + 999  # 最后一日信号突变
        alt_states = compute_states(c, alt)
        pd.testing.assert_series_equal(full.states, alt_states.states)

    def test_t_signal_change_does_not_change_t_state(self) -> None:
        c = _compiled()
        sig = _signal_frame()
        full = compute_states(c, sig)
        alt = sig.copy()
        i = 200
        alt.iloc[i] = alt.iloc[i] * 50 + 12345
        alt_states = compute_states(c, alt)
        # T 日状态只用 (T-window, T-1] 数据 => 改 T 日信号不影响 T 日状态
        assert full.states.iloc[i] == alt_states.states.iloc[i]


class TestMissingRetention:
    def test_tail_nan_retains_prior_state(self) -> None:
        c = _compiled()
        sig = _signal_frame()
        sig.iloc[200:] = np.nan
        sts = compute_states(c, sig)
        prev_state = sts.states.iloc[199]
        assert prev_state != "UNKNOWN"
        assert (sts.states.iloc[200:] == prev_state).all()

    def test_head_unknown_until_warmup(self) -> None:
        c = _compiled()
        sts = compute_states(c, _signal_frame())
        assert sts.states.iloc[0] == "UNKNOWN"
        assert sts.status.iloc[0] == "unknown"
        assert (sts.states != "UNKNOWN").any()

    def test_missing_dim_column_all_unknown(self) -> None:
        c = _compiled()
        sig = _signal_frame().drop(columns=["value_growth_spread"])
        sts = compute_states(c, sig)
        assert (sts.states == "UNKNOWN").all()
        assert (sts.status == "unknown").all()


class TestDegraded:
    def test_short_runs_marked_degraded(self) -> None:
        d = canonical_design()
        d["guardrails"]["min_state_duration_days"] = 500
        c = compile_design(d, mode="technical")
        sts = compute_states(c, _signal_frame(n=400))
        mask = sts.states != "UNKNOWN"
        assert mask.any()
        assert (sts.status[mask] == "degraded").all()


class TestBoostForState:
    def test_unknown_returns_empty_baseline(self) -> None:
        c = _compiled()
        assert boost_for_state(c, "UNKNOWN") == {}

    def test_mapping_state_returns_canonical_boosts(self) -> None:
        c = _compiled()
        assert boost_for_state(c, "SMALL_VALUE") == {
            "liquidity": 1.15,
            "reversal": 1.10,
            "momentum": 0.90,
        }

    def test_unmapped_state_returns_empty(self) -> None:
        d = canonical_design()
        d["mapping"] = {"SMALL_VALUE": {"reversal": 1.1}}
        c = compile_design(d, mode="technical")
        assert boost_for_state(c, "LARGE_GROWTH") == {}
