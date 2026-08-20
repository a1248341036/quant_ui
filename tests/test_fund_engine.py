import numpy as np
import pandas as pd

from core.engine import build_factor_frames
from core.execution import FundNavExecutionAdapter


def test_fund_share_class_mapping_uses_code_not_column_position():
    dates = pd.date_range("2024-01-01", periods=1)
    ones = np.ones((1, 2))
    adapter = FundNavExecutionAdapter(
        codes=["000002", "000001"],
        open_mat=ones,
        valid_open=ones.astype(bool),
        am20_mat=ones,
        turnover_mat=ones,
        limit_up=None,
        limit_down=None,
        dates=dates,
        buy_cost=0.01,
        sell_cost=0.02,
        lot_size=1,
        slippage_bps=0.0,
        max_participation=0.0,
        share_classes={"000001": "A", "000002": "C"},
    )

    assert adapter._buy_fee_rate(0) == 0.0  # 000002 is C class
    assert adapter._buy_fee_rate(1) == 0.01  # 000001 is A class


def test_fund_drawdown_and_composite_prefer_smaller_drawdown():
    dates = pd.date_range("2024-01-01", periods=80)
    base = np.linspace(1.0, 1.2, len(dates))
    close = pd.DataFrame({"stable": base, "drawn_down": base}, index=dates)
    close.loc[dates[70:], "drawn_down"] *= 0.8
    ones = pd.DataFrame(1.0, index=dates, columns=close.columns)

    factors = build_factor_frames(close, ones, ones, asset_type="fund_nav")

    assert factors["mdd20"].loc[dates[-1], "stable"] > factors["mdd20"].loc[
        dates[-1], "drawn_down"]
    assert factors["composite"].loc[dates[-1], "stable"] < factors[
        "composite"
    ].loc[dates[-1], "drawn_down"]
