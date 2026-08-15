from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(nav: pd.Series, periods_per_year: int = 244) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {"总收益": np.nan, "年化收益": np.nan, "年化波动": np.nan,
                "夏普": np.nan, "最大回撤": np.nan, "卡玛": np.nan,
                "胜率": np.nan}

    rets = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    n = len(rets)
    ann = (1 + total) ** (periods_per_year / n) - 1 if total > -1 else -1.0
    vol = rets.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (ann - 0.0) / vol if vol and not np.isnan(vol) else np.nan

    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    max_dd = float(drawdown.min())
    calmar = ann / abs(max_dd) if max_dd < 0 else np.nan

    win = float((rets > 0).mean()) if len(rets) else np.nan
    return {
        "总收益": float(total),
        "年化收益": float(ann),
        "年化波动": float(vol),
        "夏普": float(sharpe),
        "最大回撤": float(max_dd),
        "卡玛": float(calmar),
        "胜率": win,
    }


def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1
