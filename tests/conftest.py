from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


CODES = ["000001", "000002", "000003", "000004"]
START = "2024-06-03"
END = "2024-12-31"


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    """小型合成日线面板：4 只股票、180 个交易日，含 turn20/am20 滚动因子。"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=180)
    frames = []
    for code in CODES:
        base = 10.0 + int(code[-1])
        close = base * np.cumprod(1.0 + rng.normal(0, 0.01, len(dates)))
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.003, len(dates))))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.003, len(dates))))
        amount = rng.uniform(5e7, 2e8, len(dates))
        turnover = rng.uniform(0.5, 5.0, len(dates))
        volume = amount / close
        frames.append(pd.DataFrame({
            "date": dates,
            "code": code,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "turnover": turnover,
            "amount": amount,
            "turn20": pd.Series(turnover).rolling(20, min_periods=15).mean().values,
            "am20": pd.Series(amount).rolling(20, min_periods=15).mean().values,
            "volume": volume,
        }))
    return pd.concat(frames, ignore_index=True)
