"""防未来函数守卫测试：$label_* 列黑名单 + 时序算子负整数参数拦截。

覆盖：
- 源码扫描：label_ 前缀列被拒、字符串字面量中的名字不误报、env 覆盖前缀集
- 运行时守卫：TS_DELTA/DELAY 负窗报错（直接传参与中间变量两种形态）、正窗不受影响
- 豁免集：MULTIPLY(x, -1) / LT(x, -0.5) 等算术比较负参数合法
- 总开关：ALPHA_DSL_GUARD=0 时守卫完全不介入
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.dsl.core.guard import (
    blocked_column_prefixes,
    find_blocked_columns,
    guard_enabled,
    wrap_lookahead_guard,
)
from alphaagent.dsl.eval import eval_factor, eval_multi_line_factor


def _panel(n_days: int = 40) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-01", periods=n_days, freq="min", name="datetime"),
            pd.Index(["A", "B"], name="instrument"),
        ],
    )
    rng = np.random.default_rng(11)
    n = len(idx)
    return pd.DataFrame(
        {
            "close": 10.0 + np.cumsum(rng.normal(0, 0.2, n)),
            "volume": rng.uniform(1e4, 1e5, n),
            "label_1d_open_to_open": rng.normal(0, 0.02, n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# 列黑名单（源码扫描）
# ---------------------------------------------------------------------------


def test_find_blocked_columns_label_prefix():
    expr = "TS_MEAN($label_1d_open_to_open, 5) + $close"
    assert find_blocked_columns(expr) == ["label_1d_open_to_open"]


def test_find_blocked_columns_ignores_string_literals_and_non_prefix():
    expr = "FILLNA($close, 'label_1d_open_to_open') + $labelx"
    assert find_blocked_columns(expr) == []


def test_find_blocked_columns_strips_freq_suffix():
    assert find_blocked_columns("TS_MEAN($label_10d_close_to_close@1w, 4)") == [
        "label_10d_close_to_close"
    ]


def test_blocked_prefixes_env_replaces_default(monkeypatch):
    monkeypatch.setenv("ALPHA_DSL_BLOCKED_COL_PREFIXES", "fwd_,label_")
    assert blocked_column_prefixes() == ("fwd_", "label_")
    assert find_blocked_columns("$fwd_ret + $close") == ["fwd_ret"]


def test_blocked_prefixes_env_empty_falls_back(monkeypatch):
    monkeypatch.setenv("ALPHA_DSL_BLOCKED_COL_PREFIXES", " , ")
    assert blocked_column_prefixes() == ("label_",)


def test_eval_label_column_blocked(monkeypatch):
    monkeypatch.delenv("ALPHA_DSL_GUARD", raising=False)
    with pytest.raises(MultiLineFactorEvalError) as exc_info:
        eval_multi_line_factor(
            "x = TS_MEAN($label_1d_open_to_open, 5)\nCS_ZSCORE(x)", _panel()
        )
    err = exc_info.value
    assert err.phase == "guard"
    assert err.problem == "blocked_columns:label_1d_open_to_open"


# ---------------------------------------------------------------------------
# 负参数守卫（运行时包装）
# ---------------------------------------------------------------------------


def test_ts_delta_negative_window_blocked():
    with pytest.raises(MultiLineFactorEvalError, match="防未来函数"):
        eval_multi_line_factor("DELTA($close, -1)", _panel())


def test_delay_negative_window_blocked():
    with pytest.raises(MultiLineFactorEvalError, match="防未来函数"):
        eval_multi_line_factor("DELAY($close, -1)", _panel())


def test_negative_window_via_intermediate_variable_blocked():
    expr = "w = -1\nDELTA($close, w)"
    with pytest.raises(MultiLineFactorEvalError, match="防未来函数"):
        eval_multi_line_factor(expr, _panel())


def test_ts_pctchange_negative_window_blocked():
    with pytest.raises(MultiLineFactorEvalError, match="防未来函数"):
        eval_multi_line_factor("TS_PCTCHANGE($close, -2)", _panel())


def test_positive_windows_still_work():
    out = eval_factor("DELTA($close, 1)", _panel(), operator_monitor=False)
    assert isinstance(out, pd.Series)
    assert out.index.equals(_panel().index)


def test_arithmetic_negative_scalar_exempt():
    out = eval_factor("MULTIPLY($close, -1)", _panel(), operator_monitor=False)
    assert isinstance(out, pd.Series)


def test_comparison_negative_threshold_exempt():
    out = eval_factor("LT(DELTA($close, 1), -0.5)", _panel(), operator_monitor=False)
    assert isinstance(out, pd.Series)


def test_wrap_lookahead_guard_unit():
    calls = []

    def fake_ts(x, w):
        calls.append(w)
        return x

    def fake_arith(x, a):
        return x

    ns = wrap_lookahead_guard({"TS_FAKE": fake_ts, "MULTIPLY": fake_arith})
    ns["TS_FAKE"](1, 5)
    assert calls == [5]
    with pytest.raises(ValueError, match="TS_FAKE"):
        ns["TS_FAKE"](1, -3)
    # 负整数 np.int64 同样拦截
    with pytest.raises(ValueError, match="TS_FAKE"):
        ns["TS_FAKE"](1, np.int64(-3))
    # 豁免算子负参不受影响
    assert ns["MULTIPLY"](1, -1) == 1
    # 负浮点不拦截（阈值类参数合法）
    ns["TS_FAKE"](1, -0.5)
    assert calls == [5, -0.5]


# ---------------------------------------------------------------------------
# 总开关
# ---------------------------------------------------------------------------


def test_guard_disabled_bypasses_everything(monkeypatch):
    monkeypatch.setenv("ALPHA_DSL_GUARD", "0")
    assert guard_enabled() is False
    try:
        out = eval_factor("DELTA($close, 1)", _panel(), operator_monitor=False)
        assert isinstance(out, pd.Series)
    except MultiLineFactorEvalError as exc:
        # 守卫关闭后底层算子自行处理负窗；但绝不出现守卫报错标记
        assert "防未来函数" not in str(exc)


def test_guard_disabled_negative_window_no_guard_error(monkeypatch):
    monkeypatch.setenv("ALPHA_DSL_GUARD", "0")
    try:
        eval_multi_line_factor("DELTA($close, -1)", _panel())
    except MultiLineFactorEvalError as exc:
        assert "防未来函数" not in str(exc)
        assert exc.phase != "guard"


def test_guard_enabled_default(monkeypatch):
    monkeypatch.delenv("ALPHA_DSL_GUARD", raising=False)
    assert guard_enabled() is True
