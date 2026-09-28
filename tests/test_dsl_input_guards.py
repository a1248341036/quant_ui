import re

import pandas as pd
import pytest

from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.dsl.eval import compile_multi_line_factor
from alphaagent.dsl.stock.resample import broadcast_timeframe_to_main_freq


def test_unknown_dollar_field_fails_before_parse():
    with pytest.raises(MultiLineFactorEvalError, match=r"\$missing"):
        compile_multi_line_factor("$close + $missing", columns=["close"])


# ── DSL 字段别名（2026-09-28）：提示词/题库写 $turnover，面板列名是 turnover_rate ──


def test_turnover_alias_resolves_to_real_column():
    code = compile_multi_line_factor("CS_ZSCORE($turnover)", columns=["turnover_rate"])
    assert "$turnover_rate" in code
    assert re.search(r"\$turnover(?![A-Za-z0-9_])", code) is None


def test_turnover_alias_keeps_canonical_name_intact():
    code = compile_multi_line_factor("CS_ZSCORE($turnover_rate)", columns=["turnover_rate"])
    assert "$turnover_rate" in code
    assert "turnover_raterate" not in code  # 右边界保护：别名不误伤规范列名


def test_turnover_alias_inactive_when_column_absent():
    """目标列不存在时不静默指向不存在列 → 仍按「不可用字段」报错。"""
    with pytest.raises(MultiLineFactorEvalError, match=r"\$turnover"):
        compile_multi_line_factor("CS_ZSCORE($turnover)", columns=["close"])


def test_turnover_alias_applies_to_all_fields_in_multiline():
    expr = "turn = DIVIDE($amount, $float_cap)\nCS_ZSCORE($turnover)"
    code = compile_multi_line_factor(expr, columns=["amount", "float_cap", "turnover_rate"])
    assert "$turnover_rate" in code
    assert re.search(r"\$turnover(?![A-Za-z0-9_])", code) is None


def test_weekly_broadcast_normalizes_datetime_units():
    src_index = pd.MultiIndex.from_product(
        [
            pd.date_range("2026-01-02", periods=2, freq="W-FRI").astype("datetime64[ns]"),
            pd.Index(["A"], name="instrument"),
        ],
        names=["datetime", "instrument"],
    )
    values = pd.DataFrame({"value": [1.0, 2.0]}, index=src_index)
    target_index = pd.MultiIndex.from_product(
        [
            pd.date_range("2026-01-05", periods=14, freq="D").astype("datetime64[ms]"),
            pd.Index(["A"], name="instrument"),
        ],
        names=["datetime", "instrument"],
    )

    out = broadcast_timeframe_to_main_freq(values, target_index, "1w")
    assert len(out) == len(target_index)
    assert out.iloc[-1, 0] == 2.0
