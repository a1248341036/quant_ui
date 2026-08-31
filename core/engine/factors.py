"""因子构建函数。"""
from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd


def _compute_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
                 period: int = 14) -> pd.DataFrame:
    """Average True Range（简化滚动均值版），与 high 同索引/列。"""
    tr = pd.DataFrame(
        np.maximum.reduce([(high - low).values,
                           (high - close.shift()).abs().values,
                           (low - close.shift()).abs().values]),
        index=high.index, columns=high.columns)
    return tr.rolling(period).mean()


def _compute_adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
                 period: int = 14) -> pd.DataFrame:
    """Wilder 平均趋向指标（简化滚动均值版），返回与 close 同索引/列的 ADX 矩阵。"""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _compute_atr(high, low, close, period)
    plus_dm_df = pd.DataFrame(plus_dm, index=high.index, columns=high.columns)
    minus_dm_df = pd.DataFrame(minus_dm, index=high.index, columns=high.columns)
    plus_di = 100.0 * (plus_dm_df.rolling(period).mean() / atr)
    minus_di = 100.0 * (minus_dm_df.rolling(period).mean() / atr)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


def build_factor_frames(close: pd.DataFrame, am20: pd.DataFrame,
                        turn20: pd.DataFrame,
                        financial: dict[str, pd.DataFrame] | None = None,
                        asset_type: str = "stock",
                        volume: pd.DataFrame | None = None,
                        ) -> dict[str, pd.DataFrame]:
    """构建因子矩阵。

    asset_type 控制因子集：
    - stock/etf：全量因子（量价 + 财务）
    - fund_nav：跳过 turn20/am20（恒 1.0 无区分度），新增净值专属因子
    """
    mom20 = close.pct_change(20, fill_method=None)
    mom60 = close.pct_change(60, fill_method=None)
    vol20 = close.pct_change(fill_method=None).rolling(20).std().reindex_like(am20)
    # 双均线交叉：短期均线相对长期均线的乖离率。
    ma_cross5_10 = (close.rolling(5).mean() / close.rolling(10).mean() - 1.0)
    ma_cross5_20 = (close.rolling(5).mean() / close.rolling(20).mean() - 1.0)
    ma_cross10_30 = (close.rolling(10).mean() / close.rolling(30).mean() - 1.0)
    ma_cross20_30 = (close.rolling(20).mean() / close.rolling(30).mean() - 1.0)
    ma_cross20_60 = (close.rolling(20).mean() / close.rolling(60).mean() - 1.0)

    if asset_type == "fund_nav":
        # 基金净值专属因子（纯净值计算，无需额外数据源）
        # 滚动最大回撤：返回负回撤，越接近 0 越稳健，因此选更大值。
        def _rolling_mdd(s: pd.Series, w: int) -> pd.Series:
            roll_max = s.rolling(w, min_periods=1).max()
            return (s / roll_max - 1.0)
        mdd20 = close.apply(lambda c: _rolling_mdd(c, 20)).reindex_like(am20)
        mdd60 = close.apply(lambda c: _rolling_mdd(c, 60)).reindex_like(am20)

        # 滚动夏普：收益/波动，选最大
        rets = close.pct_change(fill_method=None)
        sharpe20 = (rets.rolling(20).mean() / rets.rolling(20).std()
                    ).reindex_like(am20)
        sharpe60 = (rets.rolling(60).mean() / rets.rolling(60).std()
                    ).reindex_like(am20)

        # 滚动 Sortino：只用下行波动
        downside = rets.clip(upper=0.0)
        sortino20 = (rets.rolling(20).mean()
                     / downside.rolling(20).std()
                     ).reindex_like(am20)

        # 动量加速度：短期动量 - 长期动量
        mom_accel = mom20 - mom60

        # 净值稳定性：滚动线性拟合 R²
        def _rolling_r2(s: pd.Series, w: int) -> pd.Series:
            x = np.arange(w)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            def _r2(window: pd.Series) -> float:
                if len(window) < w or window.isna().any():
                    return np.nan
                y = window.values
                y_mean = y.mean()
                ss_xy = ((x - x_mean) * (y - y_mean)).sum()
                ss_yy = ((y - y_mean) ** 2).sum()
                if ss_yy < 1e-12:
                    return np.nan
                return float((ss_xy ** 2) / (x_var * ss_yy))
            return s.rolling(w, min_periods=w).apply(_r2, raw=False)
        nav_stability = close.apply(lambda c: _rolling_r2(c, 60)).reindex_like(am20)

        # composite 用低波动 + 低回撤。mdd20 为负数，排名后取反，
        # 使回撤越小（越接近 0）得分越低。
        composite = vol20.rank(axis=1) - mdd20.rank(axis=1)

        frames = {
            "mom20": mom20,
            "mom60": mom60,
            "vol20": vol20,
            "composite": composite,
            "ma_cross5_10": ma_cross5_10,
            "ma_cross5_20": ma_cross5_20,
            "ma_cross10_30": ma_cross10_30,
            "ma_cross20_60": ma_cross20_60,
            "ma_cross20_30": ma_cross20_30,
            "mdd20": mdd20,
            "mdd60": mdd60,
            "sharpe20": sharpe20,
            "sharpe60": sharpe60,
            "sortino20": sortino20,
            "mom_accel": mom_accel,
            "nav_stability": nav_stability,
        }
    else:
        # 股票/ETF：全量因子
        composite = am20.rank(axis=1) + vol20.rank(axis=1)
        # 趋势突破强度：收盘相对前20日最高价的突破幅度，>0 表示创20日新高
        brk20 = close / close.shift(1).rolling(20).max() - 1.0
        frames = {
            "mom20": mom20,
            "mom60": mom60,
            "vol20": vol20,
            "am20": am20,
            "turn20": turn20,
            "composite": composite,
            "ma_cross5_10": ma_cross5_10,
            "ma_cross5_20": ma_cross5_20,
            "ma_cross10_30": ma_cross10_30,
            "ma_cross20_60": ma_cross20_60,
            "ma_cross20_30": ma_cross20_30,
            "brk20": brk20,
        }
        if volume is not None:
            # 放量确认突破：量比 = 当日成交量 / 20日均量。
            # 缩量突破（量比 < 1.5）视为无效信号置 NaN，被有效性过滤剔除。
            vratio = volume / volume.rolling(20, min_periods=15).mean()
            frames["brk20_vol"] = brk20.where(vratio >= 1.5)
        if financial:
            for name, mat in financial.items():
                if name not in frames:
                    frames[name] = mat.reindex(index=close.index,
                                               columns=close.columns)
    return frames


def _inject_pred_factor(factors: dict, close: pd.DataFrame,
                        factor: str, factor_weights: dict | None,
                        external_scores: pd.DataFrame | None = None) -> dict:
    """外部 ML 预测分数因子：读 data/pred_demo.parquet 注入 factors['pred']。"""
    needs_pred = factor == "pred" or (factor_weights and "pred" in factor_weights)
    if not needs_pred:
        return factors
    if external_scores is not None:
        pred = external_scores.reindex(index=close.index,
                                       columns=close.columns)
    else:
        from ..data import load_pred_scores
        pred = load_pred_scores(close.columns.tolist(), close.index)
    if pred is None:
        raise ValueError(
            "缺少外部预测分数 data/pred_demo.parquet，请先运行 "
            "scripts/qweave_research.py --train-model")
    factors["pred"] = pred.reindex(index=close.index, columns=close.columns)
    return factors


def build_composite_factor(
    close: pd.DataFrame,
    am20: pd.DataFrame,
    turn20: pd.DataFrame,
    weights: dict[str, float],
    directions: dict[str, bool] | None = None,
    factor_builder: Callable | None = None,
    extra_factors: dict | None = None,
) -> pd.DataFrame:
    """多因子加权合成：每个因子先做横截面百分位排名（0~1），
    按方向翻转后乘以权重求和，得到组合得分矩阵。

    weights: {因子名: 权重}，权重可正可负（负权重=反向暴露）。
    directions: {因子名: ascending}，True 表示该因子买低（取 1-rank），
    False 表示买高（取 rank）。缺省按 False 处理。
    """
    if not weights:
        raise ValueError("组合至少需要一个因子")
    builder = factor_builder or build_factor_frames
    factors = builder(close, am20, turn20)
    if extra_factors:
        factors.update(extra_factors)
    directions = directions or {}
    total: pd.DataFrame | None = None
    for name, w in weights.items():
        if name not in factors:
            raise ValueError(f"未知因子: {name}")
        mat = factors[name]
        rank = mat.rank(axis=1, pct=True)
        if directions.get(name, False):
            rank = 1.0 - rank
        term = w * rank
        total = term if total is None else total.add(term)
    if total is None:
        raise ValueError("组合因子为空")
    return total.reindex_like(close)


def _ensure_ma_cross_factor(factors: dict, close: pd.DataFrame,
                            factor: str) -> dict:
    """参数化双均线因子 ma_cross{fast}_{slow}（如 10/120）动态补齐。"""
    if factor in factors:
        return factors
    m = re.fullmatch(r"ma_cross(\d{1,3})_(\d{1,3})", factor)
    if m is not None:
        fast, slow = int(m.group(1)), int(m.group(2))
        if 1 <= fast < slow <= 500:
            factors[factor] = (close.rolling(fast).mean()
                               / close.rolling(slow).mean() - 1.0)
    return factors


def _selection_count(candidate_count: int, top_n: int,
                     selection_mode: str = "top_n",
                     selection_pct: float = 0.10,
                     min_positions: int = 1,
                     max_positions: int | None = None) -> int:
    """Resolve a fixed or dynamic number of holdings from today's eligible set."""
    if candidate_count <= 0:
        return 0
    if selection_mode == "top_pct":
        pct = min(max(float(selection_pct), 0.001), 1.0)
        count = int(np.ceil(candidate_count * pct))
    else:
        count = int(top_n)
    count = max(int(min_positions), count)
    if max_positions is not None and int(max_positions) > 0:
        count = min(count, int(max_positions))
    return min(candidate_count, count)
