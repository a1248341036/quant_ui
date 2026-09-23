# -*- coding: utf-8 -*-
"""SIGNAL_BLEND 信号积分算子：语义、边界与目录可见性。"""
import numpy as np
import pandas as pd
import pytest

from alphaagent.dsl.catalog import operator_catalog_markdown
from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.dsl.eval import eval_factor
from alphaagent.dsl.registry import build_operator_namespace


def _panel() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            ["000001", "000002"],
        ],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"close": [10.0, 20.0, 11.0, 22.0, 12.0, 24.0]}, index=idx)


def test_signal_blend_in_namespace():
    ns = build_operator_namespace()
    assert "SIGNAL_BLEND" in ns


def test_signal_blend_equals_manual_form():
    """λ∈(0,1) 时与手写 ADD/MULTIPLY/DELAY 等价式逐位一致。"""
    panel = _panel()
    out = eval_factor("SIGNAL_BLEND($close, 0.6)", panel)
    manual = eval_factor("ADD(0.6*$close, 0.4*DELAY($close, 1))", panel)
    np.testing.assert_allclose(
        out.to_numpy(), manual.to_numpy(), rtol=1e-6, equal_nan=True
    )


def test_signal_blend_boundary_lambda():
    """λ=1 恒等（首日不因 0.0*NaN 污染）；λ=0 纯滞后。"""
    panel = _panel()
    out1 = eval_factor("SIGNAL_BLEND($close, 1.0)", panel)
    np.testing.assert_allclose(out1.to_numpy(), panel["close"].to_numpy(), rtol=1e-6)
    out0 = eval_factor("SIGNAL_BLEND($close, 0.0)", panel)
    lag = eval_factor("DELAY($close, 1)", panel)
    np.testing.assert_allclose(out0.to_numpy(), lag.to_numpy(), rtol=1e-6, equal_nan=True)


def test_signal_blend_rejects_out_of_range_lambda():
    panel = _panel()
    with pytest.raises(MultiLineFactorEvalError, match="必须在 \\[0, 1\\] 内"):
        eval_factor("SIGNAL_BLEND($close, 1.5)", panel)
    with pytest.raises(MultiLineFactorEvalError, match="必须在 \\[0, 1\\] 内"):
        eval_factor("SIGNAL_BLEND($close, -0.1)", panel)


def test_signal_blend_in_catalog():
    md = operator_catalog_markdown(tier="full")
    assert "SIGNAL_BLEND(df, lam=0.5)" in md