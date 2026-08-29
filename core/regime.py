"""确定性市场制度（Regime）检测。

用规则化指标替代 AlphaCrafter Screener 的 LLM regime 诊断，
保证同样输入永远产出同样输出，适合实盘部署。

三类制度：
  - TREND_UP   (1)  趋势上行：ADX 强 + 指数在均线上方
  - TREND_DOWN (-1)  趋势下行：ADX 强 + 指数在均线下方
  - RANGE      (0)   震荡：ADX 弱

指标组合：
  - ADX（趋向指标）：>25 认为有趋势
  - 指数收盘 vs MA60：判断趋势方向
  - 波动率分位数：辅助判断（高波环境下趋势可靠性降低）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 制度枚举
TREND_UP = 1
RANGE = 0
TREND_DOWN = -1

REGIME_LABELS = {
    TREND_UP: "趋势上行",
    RANGE: "震荡",
    TREND_DOWN: "趋势下行",
}

# 默认参数
DEFAULT_ADX_PERIOD = 14
DEFAULT_ADX_THRESHOLD = 25.0
DEFAULT_MA_PERIOD = 60
DEFAULT_VOL_LOOKBACK = 60


def compute_adx_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = DEFAULT_ADX_PERIOD,
) -> pd.Series:
    """计算单序列的 ADX（Wilder 简化版）。

    Parameters
    ----------
    high, low, close : pd.Series
        指数/个股的高、低、收序列（同索引）。
    period : int
        ADX 计算周期，默认 14。
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100.0 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100.0 * (minus_dm.rolling(period).mean() / atr)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


def detect_regime_series(
    index_close: pd.Series,
    index_high: pd.Series | None = None,
    index_low: pd.Series | None = None,
    adx_period: int = DEFAULT_ADX_PERIOD,
    adx_threshold: float = DEFAULT_ADX_THRESHOLD,
    ma_period: int = DEFAULT_MA_PERIOD,
) -> pd.Series:
    """检测指数每日的 regime，返回与 index_close 同长的整型 Series。

    Parameters
    ----------
    index_close : pd.Series
        指数收盘价序列。
    index_high, index_low : pd.Series, optional
        指数高/低价，用来算 ADX。为 None 时用 close 的滚动 high/low 近似。
    adx_threshold : float
        ADX > 此值认为有趋势，否则为震荡。默认 25.
    ma_period : int
        均线周期，默认 60 交易日。

    Returns
    -------
    pd.Series
        每日 regime：1=趋势上行, 0=震荡, -1=趋势下行。
    """
    if index_high is None:
        index_high = index_close.rolling(adx_period, min_periods=1).max()
    if index_low is None:
        index_low = index_close.rolling(adx_period, min_periods=1).min()

    adx = compute_adx_series(index_high, index_low, index_close, adx_period)
    ma = index_close.rolling(ma_period).mean()

    regime = pd.Series(RANGE, index=index_close.index, dtype=int)
    has_trend = adx >= adx_threshold
    above_ma = index_close > ma
    below_ma = index_close < ma

    regime[has_trend & above_ma] = TREND_UP
    regime[has_trend & below_ma] = TREND_DOWN
    # ADX 或 MA 为 NaN 的行保持 RANGE
    regime[adx.isna() | ma.isna()] = RANGE
    return regime


def regime_summary(regime: pd.Series) -> dict:
    """统计 regime 序列的分布概况，供前端展示。"""
    counts = regime.value_counts().to_dict()
    total = max(len(regime), 1)
    return {
        "trend_up_pct": float(counts.get(TREND_UP, 0) / total),
        "range_pct": float(counts.get(RANGE, 0) / total),
        "trend_down_pct": float(counts.get(TREND_DOWN, 0) / total),
        "current_regime": int(regime.iloc[-1]) if len(regime) else RANGE,
        "current_label": REGIME_LABELS.get(
            int(regime.iloc[-1]) if len(regime) else RANGE, "未知"),
    }
