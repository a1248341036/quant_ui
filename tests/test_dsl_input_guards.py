import pandas as pd
import pytest

from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.dsl.eval import compile_multi_line_factor
from alphaagent.dsl.stock.resample import broadcast_timeframe_to_main_freq


def test_unknown_dollar_field_fails_before_parse():
    with pytest.raises(MultiLineFactorEvalError, match=r"\$missing"):
        compile_multi_line_factor("$close + $missing", columns=["close"])


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
