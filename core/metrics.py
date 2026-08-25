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
    # 算术年化夏普（行业惯例，与旧 backtest_5w 口径一致）
    sharpe = (rets.mean() * periods_per_year) / vol if vol and not np.isnan(vol) else np.nan

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


def compute_excess_metrics(nav: pd.Series, bench: pd.Series) -> dict:
    """Compute active performance from aligned strategy and benchmark NAVs."""
    aligned = pd.concat([nav.rename("nav"), bench.rename("bench")], axis=1).dropna()
    if len(aligned) < 2:
        return {"超额年化": np.nan, "超额夏普": np.nan}

    active_nav = aligned["nav"] / aligned["bench"]
    active_metrics = compute_metrics(active_nav)
    return {
        "超额年化": active_metrics["年化收益"],
        "超额夏普": active_metrics["夏普"],
    }


def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1
