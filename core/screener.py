"""Screener：regime 感知因子选择 + 动态权重分配。

借鉴 AlphaCrafter Screener 的核心流程，但用确定性计算替代 LLM 判断：

1. Regime 检测 → 确定性规则（ADX + 均线）
2. 因子适用性评分 → 最近 N 日 Rank IC 均值
3. 语义去重 → 因子族分类 + 相关性矩阵（|corr| > 阈值 视为冗余）
4. 权重分配 → |IC| 归一化
5. 方向 → IC 正负号

与 engine.py 的关系：
  - 开关关闭时：build_composite_factor 用固定 weights/directions（原逻辑不变）
  - 开关打开时：Screener 每个调仓信号日重新算 IC → 产出动态 weights/directions
    → 传给 build_composite_factor 合成得分
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .regime import (
    detect_regime_series,
    regime_summary,
    REGIME_LABELS,
    TREND_UP,
    RANGE,
    TREND_DOWN,
)


# ── 默认参数 ────────────────────────────────────────────────

DEFAULT_LOOKBACK = 10          # Rank IC 回看天数
DEFAULT_MIN_IC = 0.02          # |IC| 低于此值的因子淘汰
DEFAULT_MAX_CORR = 0.7         # 因子间 |corr| 高于此值视为冗余
DEFAULT_IC_EMA_SPAN = 5        # IC 的 EMA 平滑窗口（0 = 不平滑）


# ── 因子族 → regime 偏好映射 ────────────────────────────────
# 这是规则化的"先验"：不同 regime 下哪些因子族更可能有效。
# 不做硬过滤，只做软偏好（boost factor）——偏好族的 IC 乘以 boost。
# 数值 > 1 表示偏好，< 1 表示谨慎，= 1 中性。

FACTOR_FAMILY_BOOST: dict[int, dict[str, float]] = {
    TREND_UP: {
        "momentum": 1.2,       # 趋势市动量因子更强
        "reversal": 0.7,       # 趋势市反转因子容易亏
        "vwap": 1.1,
        "volatility": 0.9,
        "volume": 1.1,
        "fundamental": 1.0,
        "liquidity": 1.0,
        "breadth": 1.0,
        "chip": 1.0,
        "correlation": 0.9,
        "gap_overnight": 1.0,
    },
    RANGE: {
        "momentum": 0.8,       # 震荡市动量衰减
        "reversal": 1.2,       # 震荡市反转因子更强
        "vwap": 1.0,
        "volatility": 1.1,     # 震荡市低波动有优势
        "volume": 1.0,
        "fundamental": 1.0,
        "liquidity": 1.0,
        "breadth": 1.0,
        "chip": 1.0,
        "correlation": 1.1,
        "gap_overnight": 1.1,
    },
    TREND_DOWN: {
        "momentum": 0.7,       # 下行市追涨容易亏
        "reversal": 1.0,       # 反转有一定防御性
        "vwap": 0.9,
        "volatility": 1.2,     # 下行市低波动防御性强
        "volume": 0.9,
        "fundamental": 1.1,    # 下行市回归基本面
        "liquidity": 1.1,      # 下行市流动性更重要
        "breadth": 0.9,
        "chip": 1.0,
        "correlation": 1.0,
        "gap_overnight": 0.9,
    },
}


# ── 因子族分类规则 ──────────────────────────────────────────
# 与 research_memory.py _FAMILY_RULES 对齐（简化版）
# 根据 factor name 中的关键词推断所属族。

def infer_family(factor_name: str) -> str:
    """从因子名推断所属族。"""
    name = factor_name.lower()
    rules: list[tuple[str, str]] = [
        ("mom", "momentum"),
        ("reversal", "reversal"),
        ("brk", "momentum"),       # 突破归入动量族
        ("ma_cross", "momentum"),  # 均线交叉归入动量族
        ("vol", "volatility"),
        ("stdev", "volatility"),
        ("turn", "volume"),
        ("am20", "volume"),
        ("vratio", "volume"),
        ("vwap", "vwap"),
        ("mdd", "volatility"),
        ("sharpe", "volatility"),
        ("sortino", "volatility"),
        ("nav_stability", "volatility"),
        ("mom_accel", "momentum"),
        ("roe", "fundamental"),
        ("pb", "fundamental"),
        ("ep", "fundamental"),
        ("rev", "fundamental"),
        ("growth", "fundamental"),
        ("funda", "fundamental"),
        ("pred", "fundamental"),
        ("holder", "correlation"),
        ("dt_", "correlation"),
        ("bt_", "correlation"),
        ("ff_", "volume"),
        ("composite", "volatility"),
    ]
    for keyword, family in rules:
        if keyword in name:
            return family
    return "breadth"


# ── Screener 配置 ───────────────────────────────────────────

@dataclass
class ScreenerConfig:
    """Screener 运行配置。"""
    lookback: int = DEFAULT_LOOKBACK
    min_ic: float = DEFAULT_MIN_IC
    max_corr: float = DEFAULT_MAX_CORR
    ic_ema_span: int = DEFAULT_IC_EMA_SPAN
    use_family_boost: bool = True       # 是否启用因子族 regime 偏好
    adx_threshold: float = 25.0
    ma_period: int = 60
    min_cross_section: int = 30         # 某天有效股票不足此数则跳过该天 IC


@dataclass
class ScreenerResult:
    """Screener 在某个信号日的输出。"""
    signal_date: pd.Timestamp
    regime: int
    regime_label: str
    factor_ic: dict[str, float]         # 各因子原始 lookback IC
    factor_scores: dict[str, float]     # boost 后的得分
    weights: dict[str, float]           # 归一化权重
    directions: dict[str, bool]         # True=ascending(买低), False=买高
    rejected: dict[str, str]            # 被拒因子名 → 原因
    selected: list[str]                 # 选中因子名（按权重降序）
    regime_dist: dict                   # regime 分布统计


# ── 核心：单个信号日的因子评分 ──────────────────────────────

def _daily_rank_ic(factor_mat: pd.DataFrame, fwd_ret: pd.DataFrame,
                   min_cross_section: int) -> float | None:
    """计算截面 Rank IC 均值。

    factor_mat, fwd_ret : 同 shape (T, K) 的 DataFrame。
    返回所有有效截面日的 Spearman 秩相关系数均值。
    """
    from scipy.stats import spearmanr

    ic_list: list[float] = []
    for t in range(len(factor_mat)):
        fv = factor_mat.iloc[t].values
        rv = fwd_ret.iloc[t].values
        mask = np.isfinite(fv) & np.isfinite(rv)
        if mask.sum() < min_cross_section:
            continue
        rho, _ = spearmanr(fv[mask], rv[mask])
        if np.isfinite(rho):
            ic_list.append(float(rho))
    return np.mean(ic_list) if ic_list else None


def _compute_fwd_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """计算 horizon 日前瞻收益（shift(-horizon) 的 pct_change）。"""
    return close.pct_change(horizon).shift(-horizon)


def screen_factors(
    factor_frames: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    signal_idx: int,
    config: ScreenerConfig,
    index_close: pd.Series | None = None,
    index_high: pd.Series | None = None,
    index_low: pd.Series | None = None,
    all_dates: pd.DatetimeIndex | None = None,
) -> ScreenerResult:
    """在某个信号日评估所有因子，输出动态权重和方向。

    Parameters
    ----------
    factor_frames : dict[str, pd.DataFrame]
        因子名 → 因子值矩阵 (T, K)，与 close 同索引。
    close : pd.DataFrame
        收盘价矩阵 (T, K)。
    signal_idx : int
        信号日在 close.index 中的行号。
    config : ScreenerConfig
        Screener 参数。
    index_close : pd.Series, optional
        指数收盘价序列（用于 regime 检测）。为 None 时用 close 的等权均值近似。
    all_dates : pd.DatetimeIndex, optional
        完整日期索引（regime 检测用）。为 None 时用 close.index。
    """
    lookback = config.lookback
    min_cs = config.min_cross_section

    # ── 1. Regime 检测 ─────────────────────────────────────
    if index_close is None:
        # 用股票池等权均值近似指数
        index_close = close.mean(axis=1)
    if all_dates is not None:
        dates = all_dates
    else:
        dates = close.index
    regime = detect_regime_series(
        index_close, index_high, index_low,
        adx_threshold=config.adx_threshold,
        ma_period=config.ma_period,
    )
    # 截止信号日的 regime
    sig_date = dates[signal_idx]
    regime_up_to = regime.loc[:sig_date]
    current_regime = int(regime_up_to.iloc[-1]) if len(regime_up_to) else RANGE
    regime_dist = regime_summary(regime_up_to)

    # ── 2. 计算各因子近期 Rank IC ──────────────────────────
    # IC 窗口：信号日往前 lookback 天的截面 IC
    start_i = max(0, signal_idx - lookback)
    end_i = signal_idx
    fwd_ret = _compute_fwd_returns(close, horizon=1)

    factor_ic: dict[str, float] = {}
    for fname, fmat in factor_frames.items():
        sub_factor = fmat.iloc[start_i:end_i + 1]
        sub_ret = fwd_ret.iloc[start_i:end_i + 1]
        ic = _daily_rank_ic(sub_factor, sub_ret, min_cs)
        if ic is not None and np.isfinite(ic):
            factor_ic[fname] = float(ic)

    # ── 3. 阈值过滤 + 族偏好 boost ─────────────────────────
    boost_map = FACTOR_FAMILY_BOOST.get(current_regime, {}) if config.use_family_boost else {}
    factor_scores: dict[str, float] = {}
    rejected: dict[str, str] = {}

    for fname, ic in factor_ic.items():
        if abs(ic) < config.min_ic:
            rejected[fname] = f"|IC|={abs(ic):.4f} < 阈值{config.min_ic}"
            continue
        family = infer_family(fname)
        boost = boost_map.get(family, 1.0)
        factor_scores[fname] = ic * boost

    if not factor_scores:
        # 没有因子通过阈值，兜底返回空集
        return ScreenerResult(
            signal_date=sig_date,
            regime=current_regime,
            regime_label=REGIME_LABELS.get(current_regime, "未知"),
            factor_ic=factor_ic,
            factor_scores={},
            weights={},
            directions={},
            rejected=rejected,
            selected=[],
            regime_dist=regime_dist,
        )

    # ── 4. 语义去重：按 |score| 排序，贪心剔相关 ──────────────
    ranked = sorted(factor_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    selected: list[str] = []
    selected_scores: dict[str, float] = {}

    for fname, score in ranked:
        is_redundant = False
        for sel_name in selected:
            # 计算两因子在 lookback 窗口的截面 rank 相关性
            sub_a = factor_frames[fname].iloc[start_i:end_i + 1]
            sub_b = factor_frames[sel_name].iloc[start_i:end_i + 1]
            # 每天做 rank corr 取均值
            corr_list: list[float] = []
            for t in range(len(sub_a)):
                a = sub_a.iloc[t].rank().values
                b = sub_b.iloc[t].rank().values
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() >= min_cs:
                    c = np.corrcoef(a[mask], b[mask])[0, 1]
                    if np.isfinite(c):
                        corr_list.append(c)
            mean_corr = np.mean(corr_list) if corr_list else 0.0
            if abs(mean_corr) > config.max_corr:
                is_redundant = True
                rejected[fname] = (
                    f"与 {sel_name} 相关性 {abs(mean_corr):.2f} > {config.max_corr}")
                break
        if not is_redundant:
            selected.append(fname)
            selected_scores[fname] = score

    # ── 5. 权重归一化 + 方向 ───────────────────────────────
    total_abs = sum(abs(v) for v in selected_scores.values())
    weights: dict[str, float] = {}
    directions: dict[str, bool] = {}
    if total_abs > 0:
        for fname, score in selected_scores.items():
            weights[fname] = abs(score) / total_abs
            # score > 0 → 因子值大对应收益高 → 买高 → ascending=False
            # score < 0 → 因子值大对应收益低 → 买低 → ascending=True
            directions[fname] = score < 0
    else:
        # 全部 score 为 0 的退化情况，等权
        n = len(selected)
        for fname in selected:
            weights[fname] = 1.0 / n
            directions[fname] = False

    # selected 按权重降序
    selected.sort(key=lambda f: weights[f], reverse=True)

    return ScreenerResult(
        signal_date=sig_date,
        regime=current_regime,
        regime_label=REGIME_LABELS.get(current_regime, "未知"),
        factor_ic=factor_ic,
        factor_scores=selected_scores,
        weights=weights,
        directions=directions,
        rejected=rejected,
        selected=selected,
        regime_dist=regime_dist,
    )
