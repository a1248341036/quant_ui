from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .limit import build_limit_flags
from .metrics import compute_excess_metrics, compute_metrics, drawdown_series
from .assets import AssetExecutionProfile, STOCK_PROFILE
from . import trading_config
from .execution import ETFExecutionAdapter, FundNavExecutionAdapter, StockExecutionAdapter
from .selection import PortfolioBuilder, SelectionPolicy
from .screener import screen_factors, ScreenerConfig


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
        from .data import load_pred_scores
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


@dataclass(frozen=True)
class BacktestConfig:
    """一次回测运行的完整配置（与 run_backtest 的形参一一对应）。

    拆分后的引擎以 BacktestConfig 作为唯一输入载体；run_backtest 作为
    兼容门面保持原有 50 个形参签名不变，内部构造本配置后交给
    run_backtest_config 执行四阶段管线。
    """

    panel: pd.DataFrame
    codes: list[str]
    factor: str
    ascending: bool
    start: str
    end: str
    capital: float
    top_n: int
    freq: str = "monthly"
    buy_cost: float = trading_config.BUY_COST
    sell_cost: float = trading_config.SELL_COST
    amount_q: float = 0.3
    affordable: bool = True
    lot_size: int = 100
    warmup_days: int | None = None
    cash_mode: bool = True
    limit_flags: bool = True
    slippage_bps: float = trading_config.SLIPPAGE_BPS
    max_participation: float = trading_config.MAX_PARTICIPATION
    max_weight: float | None = None
    industry_map: dict[str, str] | None = None
    industry_cap: int | None = None
    factor_builder: Callable | None = None
    external_scores: pd.DataFrame | None = None
    factor_weights: dict[str, float] | None = None
    factor_directions: dict[str, bool] | None = None
    analyze: bool = False
    long_short: bool = False
    short_n: int | None = None
    short_cost_rate: float = 0.0
    industry_neutral: bool = False
    use_financial: bool = False
    risk_neutral: bool = False
    adx_filter: float | None = None
    chandelier_mult: float = 0.0
    chandelier_period: int = 22
    regime_adx: float | None = None
    regime_scale: float = 0.5
    selection_mode: str = "top_n"
    selection_pct: float = 0.10
    min_positions: int = 1
    max_positions: int | None = None
    min_score: float | None = None
    execution_profile: AssetExecutionProfile | None = None
    share_classes: dict[str, str] | None = None
    spread_bps: float | None = None
    min_commission: float | None = None
    impact_coef: float = 0.0
    impact_vol: float = 0.02
    # Screener（regime 感知因子选择）
    use_screener: bool = False
    screener_lookback: int = 10
    screener_min_ic: float = 0.02
    screener_max_corr: float = 0.7
    screener_factors: list[str] | None = None


def run_backtest_config(cfg: BacktestConfig) -> dict:
    """按 BacktestConfig 执行完整回测管线（阶段 1-4）。"""
    prep = _prepare_backtest(cfg)
    fctx = _build_factor_matrix(cfg, prep)
    sim = _simulate(cfg, prep, fctx)
    return _finalize_result(cfg, prep, fctx, sim)


def run_backtest(
    panel: pd.DataFrame,
    codes: list[str],
    factor: str,
    ascending: bool,
    start: str,
    end: str,
    capital: float,
    top_n: int,
    freq: str = "monthly",
    buy_cost: float = trading_config.BUY_COST,
    sell_cost: float = trading_config.SELL_COST,
    amount_q: float = 0.3,
    affordable: bool = True,
    lot_size: int = 100,
    warmup_days: int | None = None,
    cash_mode: bool = True,
    limit_flags: bool = True,
    slippage_bps: float = trading_config.SLIPPAGE_BPS,
    max_participation: float = trading_config.MAX_PARTICIPATION,
    max_weight: float | None = None,
    industry_map: dict[str, str] | None = None,
    industry_cap: int | None = None,
    factor_builder: Callable | None = None,
    external_scores: pd.DataFrame | None = None,
    factor_weights: dict[str, float] | None = None,
    factor_directions: dict[str, bool] | None = None,
    analyze: bool = False,
    long_short: bool = False,
    short_n: int | None = None,
    short_cost_rate: float = 0.0,
    industry_neutral: bool = False,
    use_financial: bool = False,
    risk_neutral: bool = False,
    adx_filter: float | None = None,
    chandelier_mult: float = 0.0,
    chandelier_period: int = 22,
    regime_adx: float | None = None,
    regime_scale: float = 0.5,
    selection_mode: str = "top_n",
    selection_pct: float = 0.10,
    min_positions: int = 1,
    max_positions: int | None = None,
    min_score: float | None = None,
    execution_profile: AssetExecutionProfile | None = None,
    share_classes: dict[str, str] | None = None,
    spread_bps: float | None = None,
    min_commission: float | None = None,
    impact_coef: float = 0.0,
    impact_vol: float = 0.02,
    use_screener: bool = False,
    screener_lookback: int = 10,
    screener_min_ic: float = 0.02,
    screener_max_corr: float = 0.7,
    screener_factors: list[str] | None = None,
) -> dict:
    """事件驱动回测：T+1、一手 100 股、费用、可承载性过滤。

    策略只需提供「每个信号日的因子得分」，引擎负责月度/周度调仓、
    停牌继承、买卖成本与净值计算。

    warmup_days: 因子预热天数。短窗口回测时（如只看近半年），动量/波动类
    因子在窗口起点没有足够历史，会用 start 前 warmup_days 个自然日的数据
    计算因子，但净值仍从 start 开始输出。None 表示不预热（窗口即计算区间）。

    industry_map/industry_cap: 行业分散约束。industry_map 为 {code: 行业}，
    调仓时每个行业最多选 industry_cap 只，选不满 top_n 时按实际数量等权。

    industry_neutral: 行业中性化。选股前把因子得分按行业内截面去均值，
    消除行业/风格暴露，再按中性化后的得分排序。

    use_financial: 使用财务因子（PG fina_indicator/income），因子名在
    FINANCIAL_FACTORS 中或组合权重含财务因子时自动开启。

    risk_neutral: 完整风险中性化（风格+行业）。选股前把因子得分对
    风格/行业暴露回归取残差；同时返回 risk_attribution（期末持仓风险分解）。
    需提供 industry_map。

    long_short: 多空对冲模式。多头买 top_n 只（等权 +1），空头卖最弱
    short_n 只（等权 -1），净敞口为 0（名义多头 = 名义空头 = 1 倍资金）。
    short_cost_rate: 空头年化融券费率（占空头名义的比例/年），默认 0。

    factor_builder: 可选的自定义因子构建函数，签名与 build_factor_frames 一致
    (close, am20, turn20) -> {因子名: 得分矩阵}。默认使用本模块内置实现。

    analyze: 额外计算因子质量（20 日未来收益的 Spearman IC + 5 分组收益），
    返回 factor_quality（含 ic_series / group_table）。默认 False 保持轻量。

    cash_mode: 现金/整手执行模型（与模拟盘同口径，默认 True）。按实际资金、
    100 股整手、先卖后买、费用从现金扣、涨停拒买/跌停拒卖、拒单不补仓，
    每日收盘按 close 估值；False 时保留旧的权重连续模型。

    limit_flags: 是否启用涨跌停过滤（涨停不可买入、跌停不可卖出）。
    slippage_bps: 固定滑点（基点），买入价=开盘×(1+bps/1e4)，卖出反向。
    max_participation: 流动性约束，单笔买入金额 <= 20日均成交额 × 该比例。
    max_weight: 单票权重上限（占组合市值比例），None 表示不限制。
    """

    cfg = BacktestConfig(
        panel=panel, codes=codes, factor=factor, ascending=ascending,
        start=start, end=end, capital=capital, top_n=top_n,
        freq=freq, buy_cost=buy_cost, sell_cost=sell_cost,
        amount_q=amount_q, affordable=affordable, lot_size=lot_size,
        warmup_days=warmup_days, cash_mode=cash_mode, limit_flags=limit_flags,
        slippage_bps=slippage_bps, max_participation=max_participation,
        max_weight=max_weight, industry_map=industry_map,
        industry_cap=industry_cap, factor_builder=factor_builder,
        external_scores=external_scores, factor_weights=factor_weights,
        factor_directions=factor_directions, analyze=analyze,
        long_short=long_short, short_n=short_n,
        short_cost_rate=short_cost_rate, industry_neutral=industry_neutral,
        use_financial=use_financial, risk_neutral=risk_neutral,
        adx_filter=adx_filter, chandelier_mult=chandelier_mult,
        chandelier_period=chandelier_period, regime_adx=regime_adx,
        regime_scale=regime_scale, selection_mode=selection_mode,
        selection_pct=selection_pct, min_positions=min_positions,
        max_positions=max_positions, min_score=min_score,
        execution_profile=execution_profile, share_classes=share_classes,
        spread_bps=spread_bps, min_commission=min_commission,
        impact_coef=impact_coef, impact_vol=impact_vol,
        use_screener=use_screener,
        screener_lookback=screener_lookback,
        screener_min_ic=screener_min_ic,
        screener_max_corr=screener_max_corr,
        screener_factors=screener_factors,
    )
    return run_backtest_config(cfg)


def _load_st_mask_for(
    cal: pd.DatetimeIndex,
    codes_used: list[str],
    calc_start: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame | None:
    """读取回测窗口内逐日 ST 标记（6 位 code 宽表），失败时返回 None。

    返回与 build_limit_flags 契约一致的 st_mask（index=交易日, columns=code,
    True=ST）。CNE 读取不可用时（未启用/无数据/失败）静默降级为 None，
    由 limit 层按板块比例近似——保持旧行为。
    """
    try:
        from .cne_reader import load_st_mask, CneUnavailable
        mask = load_st_mask(
            codes_used,
            start=calc_start.date().isoformat(),
            end=end_ts.date().isoformat(),
        )
        if mask is None or mask.empty:
            return None
        return mask.reindex(index=cal, columns=codes_used)
    except (CneUnavailable, Exception):  # noqa: BLE001 - 降级不打断回测
        return None


def _prepare_backtest(cfg: BacktestConfig) -> dict:
    """阶段 1 准备:面板切片/pivot、ADX/止损矩阵、估值矩阵、o2o 与调仓计划。

    产出后续阶段所需全部矩阵/计划/构建器;因子矩阵构建在阶段 2。
    """
    panel = cfg.panel
    codes = cfg.codes
    factor = cfg.factor
    ascending = cfg.ascending
    start = cfg.start
    end = cfg.end
    capital = cfg.capital
    top_n = cfg.top_n
    freq = cfg.freq
    buy_cost = cfg.buy_cost
    sell_cost = cfg.sell_cost
    amount_q = cfg.amount_q
    affordable = cfg.affordable
    lot_size = cfg.lot_size
    warmup_days = cfg.warmup_days
    cash_mode = cfg.cash_mode
    limit_flags = cfg.limit_flags
    slippage_bps = cfg.slippage_bps
    max_participation = cfg.max_participation
    max_weight = cfg.max_weight
    industry_map = cfg.industry_map
    industry_cap = cfg.industry_cap
    factor_builder = cfg.factor_builder
    external_scores = cfg.external_scores
    factor_weights = cfg.factor_weights
    factor_directions = cfg.factor_directions
    analyze = cfg.analyze
    long_short = cfg.long_short
    short_n = cfg.short_n
    short_cost_rate = cfg.short_cost_rate
    industry_neutral = cfg.industry_neutral
    use_financial = cfg.use_financial
    risk_neutral = cfg.risk_neutral
    adx_filter = cfg.adx_filter
    chandelier_mult = cfg.chandelier_mult
    chandelier_period = cfg.chandelier_period
    regime_adx = cfg.regime_adx
    regime_scale = cfg.regime_scale
    selection_mode = cfg.selection_mode
    selection_pct = cfg.selection_pct
    min_positions = cfg.min_positions
    max_positions = cfg.max_positions
    min_score = cfg.min_score
    execution_profile = cfg.execution_profile
    share_classes = cfg.share_classes
    spread_bps = cfg.spread_bps
    min_commission = cfg.min_commission
    impact_coef = cfg.impact_coef
    impact_vol = cfg.impact_vol
    profile = execution_profile or STOCK_PROFILE
    spread_bps = profile.spread_bps if spread_bps is None else spread_bps
    min_commission = profile.min_commission if min_commission is None else min_commission
    if profile.asset_type in ("etf", "fund_nav"):
        # ETF/基金入口使用各自默认费率；显式传入非股票默认费率时保留调用方配置。
        if buy_cost == STOCK_PROFILE.buy_cost:
            buy_cost = profile.buy_cost
        if sell_cost == STOCK_PROFILE.sell_cost:
            sell_cost = profile.sell_cost
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    calc_start = (start_ts - pd.Timedelta(days=warmup_days)
                  if warmup_days and warmup_days > 0 else start_ts)
    sub = panel[panel["code"].isin(codes)].copy()
    sub = sub[(sub["date"] >= calc_start) & (sub["date"] <= end_ts)]
    if sub.empty:
        raise ValueError("所选区间/股票池内没有数据")

    cal = pd.DatetimeIndex(sorted(sub["date"].unique()))

    # 排序/去重/建索引只做一次，8 列 pivot 共享（原先每列重复做是回测的主要耗时）。
    # 日频面板 (date, code) 唯一时 set_index+unstack 与 pivot_table(aggfunc="last")
    # 逐位等价；出现重复键时保留最后一条，与 aggfunc="last" 语义一致。
    dup = sub.duplicated(["date", "code"], keep="last")
    if dup.any():
        sub = sub[~dup]
    sub = sub.sort_values(["date", "code"], kind="stable").set_index(["date", "code"])

    def pivot(col: str) -> pd.DataFrame:
        # unstack 会保留全 NaN 行/列（pivot_table 会丢弃），dropna+reindex 恢复其语义：
        # 全 NaN 列（如新上市代码的 am20）丢弃后由下方统一对齐回 close 列；
        # 全 NaN 行（如 am20 前 20 个交易日）经 reindex(cal) 回到交易日历。
        out = sub[col].unstack()
        out = out.dropna(axis=1, how="all")
        return out.reindex(cal).sort_index()

    close = pivot("close")
    open_ = pivot("open")
    turnover = pivot("turnover")
    am20 = pivot("am20")
    turn20 = pivot("turn20")
    high = pivot("high") if "high" in sub.columns else None
    low = pivot("low") if "low" in sub.columns else None
    volume_w = pivot("volume") if "volume" in sub.columns else None
    # 新上市/新发代码可能在部分因子矩阵中无列（如 am20 全 NaN 被 pivot 丢弃），
    # 统一对齐到 close 列，缺失列补 NaN 后由 valid 掩码过滤为不可交易。
    open_ = open_.reindex(columns=close.columns)
    turnover = turnover.reindex(columns=close.columns)
    am20 = am20.reindex(columns=close.columns)
    turn20 = turn20.reindex(columns=close.columns)
    if high is not None:
        high = high.reindex(columns=close.columns)
    if low is not None:
        low = low.reindex(columns=close.columns)
    codes_used = close.columns.tolist()

    adx_mat = None
    if adx_filter is not None and adx_filter > 0:
        if high is None or low is None:
            print("警告: 面板无 high/low，ADX 过滤被跳过", file=sys.stderr)
        else:
            adx_mat = _compute_adx(high, low, close).values

    stop_mat = None
    if chandelier_mult > 0:
        if high is None or low is None:
            print("警告: 面板无 high/low，Chandelier 出场被跳过", file=sys.stderr)
        else:
            stop_mat = (high.rolling(chandelier_period).max()
                        - chandelier_mult * _compute_atr(high, low, close, chandelier_period)).values

    close_mat = close.values
    # 停牌日 close 为 NaN，持仓按最后一笔有效收盘价（每股 ffill）继续估值，
    # 避免停牌期间市值被当成 0、复牌日净值跳变。
    close_fill_mat = close.ffill().values
    open_mat = open_.values
    turn_mat = turnover.values
    am20_mat = am20.values
    valid_close = ~np.isnan(close_mat)
    valid_open = ~np.isnan(open_mat) & (open_mat > 0)

    dates = close.index
    T, K = close.shape
    if T < 5:
        raise ValueError("数据区间太短")

    open_ff = open_.where(open_ > 0).ffill()
    o2o = open_ff.pct_change().values  # t>=1
    # 停牌/缺失日 open 为 0 时 pct_change 会产生 inf，与 NaN 一起归零，避免污染等权基准
    o2o = np.nan_to_num(o2o, nan=0.0, posinf=0.0, neginf=0.0)

    start_idx = int(np.argmax(dates >= start_ts)) if (dates >= start_ts).any() else 0
    if freq == "daily":
        signal_idx = list(range(max(1, start_idx)))
    elif freq == "monthly":
        signal_idx = [i - 1 for i in range(1, T) if dates[i].month != dates[i - 1].month]
    elif freq == "semiannual":
        # 每年 3 月/9 月换仓：信号取前一交易日收盘，次日开盘成交
        signal_idx = [i - 1 for i in range(1, T)
                      if dates[i].month != dates[i - 1].month
                      and dates[i].month in (3, 9)]
    else:
        signal_idx = [i for i, d in enumerate(dates) if d.weekday() == 4]
    exec_dates = [i + 1 for i in signal_idx if i + 1 < T]
    # 模拟盘不支持空头；空头回测继续走权重模型
    use_cash = bool(cash_mode) and not long_short
    if freq == "daily":
        # 每日收盘信号 -> 次日开盘成交，现金/权重模式统一每日调仓
        exec_dates = list(range(max(1, start_idx), T))
    if use_cash:
        # 与模拟盘一致：窗口起点即首次调仓日（立即建仓）
        exec_dates = sorted(set(exec_dates) | {start_idx})
    limit_up = limit_down = None
    if use_cash and limit_flags:
        st_mask = _load_st_mask_for(cal, codes_used, calc_start, end_ts)
        limit_up, limit_down, _, _ = build_limit_flags(close, open_, st_mask=st_mask)
    # 预热模式：窗口起点不继承预热段持仓，也不在窗口起点当天调仓，
    # 与"全量算因子、窗口起点从零开始"的旧脚本语义一致。
    exec_set = ({i for i in exec_dates if i != start_idx}
                if (warmup_days and not use_cash) else set(exec_dates))

    portfolio_builder = PortfolioBuilder(codes_used, industry_map, industry_cap)
    # 选股策略对象：把散装选股参数收拢，选股逻辑统一走 build_targets
    selection_policy = SelectionPolicy(
        count_mode=selection_mode, top_n=top_n, pct=selection_pct,
        min_positions=min_positions, max_positions=max_positions,
        ascending=ascending, min_score=min_score,
        industry_cap=industry_cap,
        regime_adx=regime_adx, regime_scale=regime_scale,
    )

    return {
        "profile": profile, "spread_bps": spread_bps,
        "min_commission": min_commission, "buy_cost": buy_cost,
        "sell_cost": sell_cost, "start_ts": start_ts, "end_ts": end_ts,
        "cal": cal, "close": close, "open_": open_, "turnover": turnover,
        "am20": am20, "turn20": turn20, "high": high, "low": low,
        "volume_w": volume_w, "codes_used": codes_used,
        "adx_mat": adx_mat, "stop_mat": stop_mat,
        "close_mat": close_mat, "close_fill_mat": close_fill_mat,
        "open_mat": open_mat, "turn_mat": turn_mat, "am20_mat": am20_mat,
        "valid_close": valid_close, "valid_open": valid_open,
        "dates": dates, "T": T, "K": K, "o2o": o2o, "start_idx": start_idx,
        "exec_dates": exec_dates, "use_cash": use_cash,
        "limit_up": limit_up, "limit_down": limit_down, "exec_set": exec_set,
        "portfolio_builder": portfolio_builder,
        "selection_policy": selection_policy,
        "capital": capital, "top_n": top_n, "freq": freq,
        "amount_q": amount_q, "affordable": affordable, "lot_size": lot_size,
        "warmup_days": warmup_days, "cash_mode": cash_mode,
        "limit_flags": limit_flags, "slippage_bps": slippage_bps,
        "max_participation": max_participation, "max_weight": max_weight,
        "industry_map": industry_map, "industry_cap": industry_cap,
        "factor_builder": factor_builder, "external_scores": external_scores,
        "factor_weights": factor_weights,
        "factor_directions": factor_directions, "analyze": analyze,
        "long_short": long_short, "short_n": short_n,
        "short_cost_rate": short_cost_rate,
        "industry_neutral": industry_neutral, "use_financial": use_financial,
        "risk_neutral": risk_neutral, "adx_filter": adx_filter,
        "chandelier_mult": chandelier_mult,
        "chandelier_period": chandelier_period,
        "regime_adx": regime_adx, "regime_scale": regime_scale,
        "selection_mode": selection_mode, "selection_pct": selection_pct,
        "min_positions": min_positions, "max_positions": max_positions,
        "min_score": min_score, "execution_profile": execution_profile,
        "share_classes": share_classes,
        "impact_coef": impact_coef, "impact_vol": impact_vol,
        "use_screener": cfg.use_screener,
        "screener_lookback": cfg.screener_lookback,
        "screener_min_ic": cfg.screener_min_ic,
        "screener_max_corr": cfg.screener_max_corr,
        "screener_factors": cfg.screener_factors,
        "signal_indices": sorted(set(i for i in
            (e - 1 for e in exec_dates if e > 0)
            if 0 <= i < T)),
    }


def _build_factor_matrix(cfg: BacktestConfig, prep: dict) -> dict:
    """阶段 2 因子矩阵:财务帧、因子构建、pred 注入、composite 与中性化。"""
    close = prep["close"]
    am20 = prep["am20"]
    turn20 = prep["turn20"]
    volume_w = prep["volume_w"]
    cal = prep["cal"]
    codes_used = prep["codes_used"]
    profile = prep["profile"]
    factor = cfg.factor
    factor_weights = cfg.factor_weights
    factor_directions = cfg.factor_directions
    external_scores = cfg.external_scores
    analyze = cfg.analyze
    industry_map = cfg.industry_map
    risk_neutral = cfg.risk_neutral
    industry_neutral = cfg.industry_neutral
    factor_builder = cfg.factor_builder
    use_financial = cfg.use_financial

    from .financial import FINANCIAL_FACTORS, financial_factor_frames
    need_financial = use_financial or factor in FINANCIAL_FACTORS
    if factor_weights:
        need_financial = need_financial or any(
            n in FINANCIAL_FACTORS for n in factor_weights)
    financial_frames = None
    if need_financial:
        try:
            financial_frames = financial_factor_frames(codes_used, cal, close)
        except Exception:
            financial_frames = None

    _asset_type = profile.asset_type

    def _default_builder(c: pd.DataFrame, a: pd.DataFrame, t: pd.DataFrame):
        return build_factor_frames(c, a, t, financial=financial_frames,
                                   asset_type=_asset_type, volume=volume_w)

    builder = factor_builder or _default_builder
    factors = builder(close, am20, turn20)
    _inject_pred_factor(factors, close, factor, factor_weights, external_scores)
    _ensure_ma_cross_factor(factors, close, factor)
    if factor_weights:
        # 多因子自由组合：权重合成后的得分矩阵
        combo = build_composite_factor(
            close, am20, turn20, factor_weights, factor_directions,
            factor_builder=builder,
            extra_factors={"pred": factors["pred"]} if "pred" in factors else None,
        )
        fmat = combo.values
        quality = None
        if analyze:
            from .performance import factor_quality
            quality = factor_quality(combo, close, horizon=20, groups=5, min_n=10)
    else:
        fmat = factors[factor].values
        quality = None
        if analyze:
            from .performance import factor_quality
            quality = factor_quality(factors[factor], close, horizon=20, groups=5, min_n=10)

    use_screener = prep.get("use_screener", False)

    # Screener 模式：保留全部因子帧供信号日动态评分
    if use_screener:
        screener_cfg = ScreenerConfig(
            lookback=prep.get("screener_lookback", 10),
            min_ic=prep.get("screener_min_ic", 0.02),
            max_corr=prep.get("screener_max_corr", 0.7),
        )
        # 筛选要参与 Screener 的因子（用户可指定子集，否则取 factor_weights 的键）
        screener_factor_names = prep.get("screener_factors") or list(factors.keys())
        screener_frames = {n: factors[n] for n in screener_factor_names
                           if n in factors and n != "composite"}
        signal_indices = prep.get("signal_indices", [])
        close_df = prep["close"]
        high_df = prep.get("high")
        low_df = prep.get("low")
        # 用等权均值近似指数
        index_close_s = close_df.mean(axis=1)
        index_high_s = high_df.mean(axis=1) if high_df is not None else None
        index_low_s = low_df.mean(axis=1) if low_df is not None else None

        # 逐信号日算 Screener → 动态权重 → 合成得分
        combo_arr = np.full_like(close_df.values, np.nan)
        screener_log: list[dict] = []
        for sig_i in signal_indices:
            result = screen_factors(
                screener_frames, close_df, sig_i, screener_cfg,
                index_close=index_close_s,
                index_high=index_high_s,
                index_low=index_low_s,
                all_dates=close_df.index,
            )
            if not result.weights:
                # 没有因子通过，该信号日得分全 NaN（选不出股 → 空仓）
                screener_log.append({
                    "date": str(result.signal_date.date()),
                    "regime": result.regime_label,
                    "selected": [],
                    "rejected": dict(list(result.rejected.items())[:5]),
                    "weights": {},
                })
                continue
            # 用动态权重合成该信号日的截面得分
            row = np.zeros(len(codes_used))
            for fname, w in result.weights.items():
                fmat_row = factors[fname].iloc[sig_i].values
                ascending = result.directions.get(fname, False)
                rank = pd.Series(fmat_row).rank(pct=True).values
                if ascending:
                    rank = 1.0 - rank
                row += w * rank
            combo_arr[sig_i] = row
            screener_log.append({
                "date": str(result.signal_date.date()),
                "regime": result.regime_label,
                "selected": result.selected,
                "factor_ic": {k: round(v, 4) for k, v in result.factor_ic.items()},
                "weights": {k: round(v, 4) for k, v in result.weights.items()},
                "directions": {k: "买低" if v else "买高"
                               for k, v in result.directions.items()},
                "rejected": dict(list(result.rejected.items())[:5]),
            })

        # 填充非信号日的值（用前一个信号日的得分延持到次日执行）
        fmat_frame = pd.DataFrame(combo_arr, index=close_df.index,
                                  columns=codes_used)
        fmat = fmat_frame.values
        quality = None
        if analyze:
            from .performance import factor_quality
            quality = factor_quality(fmat_frame, close, horizon=20, groups=5, min_n=10)
        return {
            "fmat": fmat,
            "quality": quality,
            "_X_risk": None,
            "_risk_names": [],
            "screener_log": screener_log,
        }

    if factor_weights:
        factor = "composite"

    _X_risk: np.ndarray | None = None
    _risk_names: list[str] = []
    if risk_neutral and industry_map:
        from .risk_model import (build_exposures, neutralize)
        _X_risk, _risk_names = build_exposures(
            close.values, am20.values, turn20.values,
            mom20=factors.get("mom20").values if "mom20" in factors else None,
            vol20=factors.get("vol20").values if "vol20" in factors else None,
            pb=factors.get("pb").values if "pb" in factors else None,
            roe=factors.get("roe").values if "roe" in factors else None,
            growth=factors.get("rev_yoy").values if "rev_yoy" in factors else None,
            industry_map=industry_map, codes=codes_used,
        )
        fmat = neutralize(np.array(fmat, dtype=float, copy=True), _X_risk)
        fmat_frame = pd.DataFrame(fmat, index=close.index, columns=codes_used)
        if analyze:
            from .performance import factor_quality
            quality = factor_quality(fmat_frame, close, horizon=20, groups=5, min_n=10)
    elif industry_neutral and industry_map:
        ind_arr = np.array([industry_map.get(str(c), "?") for c in codes_used])
        raw_fmat = np.array(fmat, dtype=float, copy=True)
        for ind in np.unique(ind_arr):
            mask = ind_arr == ind
            if mask.sum() == 0:
                continue
            sub = raw_fmat[:, mask]
            valid_cnt = np.sum(~np.isnan(sub), axis=1, keepdims=True)
            valid_sum = np.nansum(np.where(np.isnan(sub), 0.0, sub), axis=1, keepdims=True)
            row_means = np.divide(valid_sum, valid_cnt,
                                  out=np.full_like(valid_sum, np.nan),
                                  where=valid_cnt > 0)
            raw_fmat[:, mask] = sub - row_means
        fmat = raw_fmat

    return {
        "fmat": fmat,
        "quality": quality,
        "_X_risk": _X_risk,
        "_risk_names": _risk_names,
        "screener_log": [],
    }


def _simulate(cfg: BacktestConfig, prep: dict, fctx: dict) -> dict:
    """阶段 3 主循环:现金/整手模型与旧权重连续模型(含多空)。"""
    codes_used = prep["codes_used"]
    profile = prep["profile"]
    close = prep["close"]
    open_ = prep["open_"]
    turnover = prep["turnover"]
    am20 = prep["am20"]
    high = prep["high"]
    low = prep["low"]
    dates = prep["dates"]
    close_mat = prep["close_mat"]
    close_fill_mat = prep["close_fill_mat"]
    open_mat = prep["open_mat"]
    turn_mat = prep["turn_mat"]
    am20_mat = prep["am20_mat"]
    valid_close = prep["valid_close"]
    valid_open = prep["valid_open"]
    o2o = prep["o2o"]
    T = prep["T"]
    K = prep["K"]
    start_idx = prep["start_idx"]
    exec_set = prep["exec_set"]
    limit_up = prep["limit_up"]
    limit_down = prep["limit_down"]
    use_cash = prep["use_cash"]
    stop_mat = prep["stop_mat"]
    adx_mat = prep["adx_mat"]
    selection_policy = prep["selection_policy"]
    portfolio_builder = prep["portfolio_builder"]
    capital = prep["capital"]
    top_n = prep["top_n"]
    warmup_days = prep["warmup_days"]
    amount_q = prep["amount_q"]
    affordable = prep["affordable"]
    lot_size = prep["lot_size"]
    buy_cost = prep["buy_cost"]
    sell_cost = prep["sell_cost"]
    slippage_bps = prep["slippage_bps"]
    max_participation = prep["max_participation"]
    spread_bps = prep["spread_bps"]
    min_commission = prep["min_commission"]
    impact_coef = prep["impact_coef"]
    impact_vol = prep["impact_vol"]
    max_weight = prep["max_weight"]
    share_classes = prep["share_classes"]
    long_short = prep["long_short"]
    short_n = prep["short_n"]
    short_cost_rate = prep["short_cost_rate"]
    selection_mode = prep["selection_mode"]
    selection_pct = prep["selection_pct"]
    min_positions = prep["min_positions"]
    max_positions = prep["max_positions"]
    ascending = cfg.ascending
    adx_filter = prep["adx_filter"]
    fmat = fctx["fmat"]

    nav = np.ones(T)
    bench = np.ones(T)
    hold = np.zeros(K)
    holdings_history = []
    trades: list[dict] = []
    trades_detail: list[dict] = []
    cash_history: list[float] = []
    positions_history: list[dict[str, float]] = []
    rejections: list[dict] = []
    last_chosen = []

    def _record_holdings(day: int, cash_: float, pos: dict[int, float]) -> float:
        eq = float(cash_)
        for k, sh in pos.items():
            if np.isfinite(close_fill_mat[day, k]):
                eq += sh * float(close_fill_mat[day, k])
        w = np.zeros(K)
        for k, sh in pos.items():
            px = float(close_fill_mat[day, k]) if np.isfinite(close_fill_mat[day, k]) else 0.0
            w[k] = sh * px / eq if eq > 0 else 0.0
        holdings_history.append(w)
        cash_history.append(float(cash_))
        positions_history.append({codes_used[k]: float(sh)
                                  for k, sh in pos.items() if abs(sh) > 1e-9})
        return eq

    if use_cash:
        cash = float(capital)
        positions: dict[int, float] = {}
        if profile.asset_type == "etf":
            adapter_cls = ETFExecutionAdapter
        elif profile.asset_type == "fund_nav":
            adapter_cls = FundNavExecutionAdapter
        else:
            adapter_cls = StockExecutionAdapter
        adapter_kwargs = dict(
            codes=codes_used, open_mat=open_mat, valid_open=valid_open,
            am20_mat=am20_mat, turnover_mat=turn_mat,
            limit_up=limit_up, limit_down=limit_down, dates=dates,
            buy_cost=buy_cost, sell_cost=sell_cost, lot_size=lot_size,
            slippage_bps=slippage_bps,
            max_participation=max_participation,
            spread_bps=spread_bps,
            min_commission=min_commission,
            impact_coef=impact_coef, impact_vol=impact_vol,
        )
        if adapter_cls is FundNavExecutionAdapter:
            # 传入 A/C 类信息，供申购费率判断
            if share_classes:
                adapter_kwargs["share_classes"] = share_classes
        execution_adapter = adapter_cls(**adapter_kwargs)

        def _portfolio_value(day: int) -> float:
            v = cash
            for k, sh in positions.items():
                if np.isfinite(close_fill_mat[day, k]):
                    v += sh * float(close_fill_mat[day, k])
            return v

        for t in range(1, T):
            prev = t - 1
            if use_cash and stop_mat is not None:
                # Chandelier/ATR 追踪止损：收盘跌破止损线，次日开盘卖出
                stopped = execution_adapter.execute_stop_losses(
                    cash, positions, close_mat, valid_close, stop_mat, prev, t,
                )
                cash, positions = stopped.cash, stopped.positions
                trades_detail.extend(stopped.trades_detail)
            if t == start_idx:
                # 窗口起点：清空预热段持仓，与模拟盘/旧口径一致从零开始
                positions.clear()
                cash = float(capital)
                bench[t] = 1.0
            elig = valid_close[prev] & valid_open[prev] & valid_open[t]
            bench_ret = float(np.nanmean(np.where(elig, o2o[t], np.nan))) if elig.any() else 0.0

            if t in exec_set:
                sig = prev
                am_vals = am20_mat[sig]
                finite = am_vals[~np.isnan(am_vals)]
                am_thr = np.nanquantile(am_vals, amount_q) if finite.size else np.nan
                adx_ok = (adx_mat[sig] >= adx_filter) if adx_mat is not None else None
                valid = (valid_close[sig] & valid_open[t]
                         & ~np.isnan(fmat[sig])
                         & ~np.isnan(turn_mat[sig])
                         & ~np.isnan(am20_mat[sig])
                         & (am20_mat[sig] >= am_thr)
                         & (turn_mat[sig] > 0))
                if adx_ok is not None:
                    valid = valid & adx_ok
                cand = np.where(valid)[0]
                if limit_up is not None and len(cand):
                    cand = cand[~limit_up[t, cand]]
                targets: dict[int, float] = {}
                chosen_list: list[int] = []
                if len(cand) > 0:
                    scores = fmat[sig, cand]
                    market_adx = (float(np.nanmedian(adx_mat[sig]))
                                  if (selection_policy.regime_adx is not None
                                      and adx_mat is not None) else None)
                    chosen_list, sel_targets = portfolio_builder.build_targets(
                        selection_policy, cand, scores, market_adx)
                    targets.update(sel_targets)
                    if chosen_list:
                        last_chosen = [codes_used[k] for k in chosen_list]

                pv = _portfolio_value(sig)
                signal_d = dates[sig].date().isoformat()
                exec_d = dates[t].date().isoformat()
                bought_codes: list[str] = []
                sold_codes: list[str] = []
                buy_amt = 0.0
                sell_amt = 0.0

                executed = execution_adapter.execute_targets(
                    cash, positions, targets, chosen_list, pv, am_thr, sig, t,
                    max_weight=max_weight,
                )
                cash, positions = executed.cash, executed.positions
                buy_amt, sell_amt = executed.buy_amount, executed.sell_amount
                bought_codes = executed.bought_codes
                sold_codes = executed.sold_codes
                trades_detail.extend(executed.trades_detail)
                rejections.extend(executed.rejections)

                turn = (buy_amt + sell_amt) / 2.0 / pv if pv else 0.0
                trades.append({
                    "date": dates[t],
                    "signal_date": dates[sig],
                    "num_hold": int(sum(1 for v in positions.values() if v > 0)),
                    "turnover": float(turn),
                    "bought": ",".join(bought_codes[:12]),
                    "sold": ",".join(sold_codes[:12]),
                })

            eq = _record_holdings(t, cash, positions)
            nav[t] = eq / capital
            if t != start_idx:
                bench[t] = bench[t - 1] * (1.0 + bench_ret)

        hold = np.zeros(K)
        final_eq = float(cash)
        for k, sh in positions.items():
            px = float(close_fill_mat[-1, k]) if np.isfinite(close_fill_mat[-1, k]) else 0.0
            final_eq += sh * px
        for k, sh in positions.items():
            px = float(close_fill_mat[-1, k]) if np.isfinite(close_fill_mat[-1, k]) else 0.0
            hold[k] = sh * px / final_eq if final_eq > 0 else 0.0
    else:
        for t in range(1, T):
            prev = t - 1
            if t == start_idx and warmup_days:
                # 窗口起点：持仓清零、净值归 1，只输出预热段之后的净值
                hold = np.zeros(K)
                nav[t] = 1.0
                bench[t] = 1.0
                holdings_history.append(hold.copy())
                cash_history.append(float(capital))
                positions_history.append({})
                continue
            rr = o2o[t]
            if long_short:
                # 多空：long/short 两腿按各自的加权收益分别再平衡，
                # 保持多头名义=1、空头名义=1、净敞口=0，避免共用分母导致敞口漂移。
                long_mask = hold > 0
                short_mask = hold < 0
                long_gross = float(hold[long_mask].sum()) if long_mask.any() else 0.0
                short_notional = float(-hold[short_mask].sum()) if short_mask.any() else 0.0
                long_ret = (float((rr[long_mask] * hold[long_mask]).sum()) / long_gross
                            if long_gross > 0 else 0.0)
                # 空头腿：标的加权收益 u_short，空头腿 P&L = -u_short。
                # 再平衡时多头按 (1+long_ret) 缩放，空头按 (1+u_short) 缩放，
                # 保持两腿名义各 1、净敞口为 0。
                short_base = -hold[short_mask] if short_mask.any() else np.zeros(0)
                u_short = (float((rr[short_mask] * short_base).sum()) / short_notional
                           if short_notional > 0 else 0.0)
                raw = long_gross * long_ret - short_notional * u_short
                new_hold = np.zeros(K)
                if long_gross > 0 and (1.0 + long_ret) > 0:
                    new_hold[long_mask] = (hold[long_mask] * (1.0 + rr[long_mask])
                                           / (1.0 + long_ret))
                if short_notional > 0 and (1.0 + u_short) > 0:
                    new_hold[short_mask] = (hold[short_mask] * (1.0 + rr[short_mask])
                                            / (1.0 + u_short))
                hold = new_hold
                raw -= short_notional * short_cost_rate / 252.0
            else:
                gross = float(np.abs(hold).sum())
                if gross > 0:
                    raw = float((rr * hold).sum()) / hold.sum()
                    hold = hold * (1.0 + rr) / (1.0 + raw)
                else:
                    raw = 0.0

            elig = valid_close[prev] & valid_open[prev] & valid_open[t]
            bench_ret = float(np.nanmean(np.where(elig, rr, np.nan))) if elig.any() else 0.0

            if t in exec_set:
                sig = prev
                am_vals = am20_mat[sig]
                finite = am_vals[~np.isnan(am_vals)]
                am_thr = np.nanquantile(am_vals, amount_q) if finite.size else np.nan
                adx_ok = (adx_mat[sig] >= adx_filter) if adx_mat is not None else None
                valid = (valid_close[sig] & valid_open[t]
                         & ~np.isnan(fmat[sig])
                         & ~np.isnan(turn_mat[sig])
                         & ~np.isnan(am20_mat[sig])
                         & (am20_mat[sig] >= am_thr)
                         & (turn_mat[sig] > 0))
                if adx_ok is not None:
                    valid = valid & adx_ok
                cand = np.where(valid)[0]
                cant_sell = np.where((hold != 0) & ~valid_open[t])[0]
                new_hold = np.zeros(K)
                if len(cand) > 0:
                    scores = fmat[sig, cand]
                    if affordable:
                        per_budget = capital / top_n
                        prices = close_mat[sig, cand]
                        afford = (prices * lot_size) <= per_budget
                        if afford.any():
                            cand = cand[afford]
                            scores = scores[afford]
                        else:
                            cand = np.array([], dtype=int)
                            scores = np.array([])
                    if len(cand) > 0:
                        gated = selection_policy.gate(cand, scores)
                        if len(gated) > 0:
                            scores = scores[np.isin(cand, gated)]
                            cand = gated
                    if len(cand) > 0:
                        long_n = PortfolioBuilder.selection_count(
                            len(cand), top_n, selection_mode, selection_pct,
                            min_positions, max_positions)
                        short_count = short_n or long_n
                        min_cand = long_n + short_count if long_short else long_n
                        ordered = portfolio_builder.rank_select(
                            cand, scores, ascending, min_cand, "top_n", 0.10,
                            1, None, limit_count=min_cand,
                        )
                        ordered_sel = np.asarray(ordered, dtype=int)
                        if len(ordered_sel) >= min_cand:
                            chosen = ordered_sel[:long_n]
                            long_stuck = float(np.maximum(hold[cant_sell], 0).sum()) if len(cant_sell) else 0.0
                            remain_long = 1.0 - long_stuck
                            new_hold[chosen] = remain_long / len(chosen)
                            last_chosen = [codes_used[int(c)] for c in chosen]
                            if long_short:
                                shorts = ordered_sel[-short_count:]
                                short_stuck = float(np.maximum(-hold[cant_sell], 0).sum()) if len(cant_sell) else 0.0
                                remain_short = 1.0 - short_stuck
                                new_hold[shorts] = -remain_short / len(shorts)
                new_hold[cant_sell] = hold[cant_sell]

                buy = float(np.maximum(new_hold - hold, 0).sum())
                sell = float(np.maximum(hold - new_hold, 0).sum())
                turn = float(np.abs(new_hold - hold).sum() / 2.0)
                raw = raw - buy * buy_cost - sell * sell_cost
                sold_codes = [codes_used[k] for k in np.where((hold > 0) & (new_hold <= hold))[0] if hold[k] > 0]
                bought_codes = [codes_used[k] for k in np.where(new_hold > hold)[0]]
                trades.append({
                    "date": dates[t],
                    "signal_date": dates[sig],
                    "num_hold": int((new_hold != 0).sum() if long_short
                                    else (new_hold > 0).sum()),
                    "turnover": turn,
                    "bought": ",".join(bought_codes[:12]),
                    "sold": ",".join(sold_codes[:12]),
                })
                hold = new_hold

            nav[t] = nav[t - 1] * (1.0 + raw)
            bench[t] = bench[t - 1] * (1.0 + bench_ret)
            holdings_history.append(hold.copy())
            cash_history.append(float(capital * nav[t]))
            positions_history.append({codes_used[k]: float(v)
                                      for k, v in enumerate(hold)
                                      if abs(v) > 1e-9})

    return {
        "nav": nav, "bench": bench, "trades": trades,
        "trades_detail": trades_detail, "holdings_history": holdings_history,
        "cash_history": cash_history, "positions_history": positions_history,
        "rejections": rejections, "last_chosen": last_chosen, "hold": hold,
    }


def _finalize_result(cfg: BacktestConfig, prep: dict, fctx: dict, sim: dict) -> dict:
    """阶段 4 收尾:预热段切片、指标、持仓表、风险归因与结果字典。"""
    profile = prep["profile"]
    close = prep["close"]
    codes_used = prep["codes_used"]
    dates = prep["dates"]
    K = prep["K"]
    o2o = prep["o2o"]
    start_ts = prep["start_ts"]
    start_idx = prep["start_idx"]
    exec_set = prep["exec_set"]
    capital = prep["capital"]
    nav = sim["nav"]
    bench = sim["bench"]
    trades = sim["trades"]
    trades_detail = sim["trades_detail"]
    holdings_history = sim["holdings_history"]
    cash_history = sim["cash_history"]
    positions_history = sim["positions_history"]
    rejections = sim["rejections"]
    last_chosen = sim["last_chosen"]
    hold = sim["hold"]
    _X_risk = fctx.get("_X_risk")
    _risk_names = fctx.get("_risk_names") or []
    quality = fctx.get("quality")
    screener_log = fctx.get("screener_log", [])

    exec_in_out = sorted(e for e in exec_set if e > 0)
    last_signal_date = dates[exec_in_out[-1] - 1] if exec_in_out else None
    if start_idx > 0:
        # 预热段只用于因子计算，净值/成交从 start 开始输出
        nav = nav[start_idx:]
        bench = bench[start_idx:]
        # 兜底：基准必须从窗口起点归 1，避免前端 capital*bench 起点偏差
        if len(bench) and not np.isclose(bench[0], 1.0):
            bench = bench / bench[0]
        dates_out = dates[start_idx:]
        trades = [t for t in trades if t["date"] >= start_ts]
        holdings_history = holdings_history[start_idx - 1:] if start_idx > 0 else holdings_history
        cash_history = cash_history[start_idx - 1:] if start_idx > 0 else cash_history
        positions_history = positions_history[start_idx - 1:] if start_idx > 0 else positions_history
        trades_detail = [t for t in trades_detail if t["date"] >= start_ts]
        rejections = [r for r in rejections if r["date"] >= start_ts.date().isoformat()]
    else:
        dates_out = dates
        # start_idx == 0：nav 有 T 个元素（含初始 nav[0]=1.0），
        # 但 cash_history/positions_history 只有 T-1 个（循环从 t=1 开始记录）。
        # 补齐初始状态，使长度与 nav/weight_history 一致。
        cash_history = [float(capital)] + cash_history
        positions_history = [{}] + positions_history

    if quality is not None:
        from .performance import slice_quality
        quality = slice_quality(quality, dates_out)

    # 每日市值权重历史（供 Brinson/风险归因等下游使用）
    wh_list = holdings_history if start_idx > 0 else [np.zeros(K)] + holdings_history
    weight_history = [{codes_used[k]: float(v) for k, v in enumerate(h)
                       if abs(v) > 1e-9} for h in wh_list]

    nav_s = pd.Series(nav, index=dates_out, name="nav")
    bench_s = pd.Series(bench, index=dates_out, name="bench")
    trades_df = pd.DataFrame(trades)

    last_hold = pd.Series(hold, index=codes_used)
    last_holdings = last_hold[last_hold != 0].sort_values(ascending=False)
    last_price = close.iloc[-1]
    holdings_df = pd.DataFrame({
        "code": last_holdings.index,
        "weight": last_holdings.values,
        "price": [last_price.get(c, np.nan) for c in last_holdings.index],
        "direction": ["空" if v < 0 else "多" for v in last_holdings.values],
    })
    holdings_df["market_value"] = holdings_df["weight"] * nav_s.iloc[-1] * capital
    holdings_df["weight_pct"] = holdings_df["weight"] * 100

    risk_attribution = None
    if _X_risk is not None:
        from .risk_model import (covariance_from_exposures,
                                 portfolio_risk_attribution)
        _, factor_cov, spec_var = covariance_from_exposures(_X_risk, o2o)
        last_w = np.zeros(K)
        for c, v in last_holdings.items():
            last_w[codes_used.index(str(c))] = v
        w_norm = last_w / (np.abs(last_w).sum() or 1.0)
        risk_attribution = portfolio_risk_attribution(
            w_norm, _X_risk, factor_cov, spec_var, _risk_names)

    metrics = compute_metrics(nav_s)
    metrics.update(compute_excess_metrics(nav_s, bench_s))

    return {
        "nav": nav_s,
        "bench": bench_s,
        "drawdown": drawdown_series(nav_s),
        "metrics": metrics,
        "bench_metrics": compute_metrics(bench_s),
        "trades": trades_df,
        "holdings": holdings_df,
        "last_signal_date": last_signal_date,
        "capital": capital,
        "dates": dates_out,
        "last_chosen": last_chosen,
        "factor_quality": quality,
        "weight_history": weight_history,
        "trades_detail": trades_detail,
        "cash_history": cash_history,
        "positions_history": positions_history,
        "rejections": rejections,
        "risk_attribution": risk_attribution,
        "asset_type": profile.asset_type,
        "execution_profile": profile,
        "screener_log": screener_log,
    }

def latest_signals(panel: pd.DataFrame, codes: list[str], factor: str,
                   ascending: bool, top_n: int = 10,
                   factor_weights: dict[str, float] | None = None,
                   factor_directions: dict[str, bool] | None = None,
                   long_short: bool = False,
                   short_n: int | None = None,
                   use_financial: bool = False,
                   adx_filter: float | None = None,
                   asset_type: str = "stock") -> pd.DataFrame:
    sub = panel[panel["code"].isin(codes)].copy()

    cal = pd.DatetimeIndex(sorted(sub["date"].unique()))

    # 与 _prepare_backtest.pivot 相同的快速路径：排序/去重/建索引一次共享。
    dup = sub.duplicated(["date", "code"], keep="last")
    if dup.any():
        sub = sub[~dup]
    sub = sub.sort_values(["date", "code"], kind="stable").set_index(["date", "code"])

    def pivot(col: str) -> pd.DataFrame:
        # 全 NaN 列丢弃（pivot_table 语义），全 NaN 行由 reindex(cal) 恢复交易日历
        return sub[col].unstack().dropna(axis=1, how="all").reindex(cal).sort_index()

    close = pivot("close")
    am20 = pivot("am20")
    turn20 = pivot("turn20")
    turnover = pivot("turnover")
    high = pivot("high") if "high" in sub.columns else None
    low = pivot("low") if "low" in sub.columns else None
    volume_sig = pivot("volume") if "volume" in sub.columns else None
    adx_row = None
    if adx_filter is not None and high is not None and low is not None:
        high = high.reindex(columns=close.columns)
        low = low.reindex(columns=close.columns)
        adx_row = _compute_adx(high, low, close).iloc[-1]
    from .financial import FINANCIAL_FACTORS, financial_factor_frames
    need_financial = use_financial or factor in FINANCIAL_FACTORS
    if factor_weights:
        need_financial = need_financial or any(
            n in FINANCIAL_FACTORS for n in factor_weights)
    financial_frames = None
    if need_financial:
        try:
            financial_frames = financial_factor_frames(
                close.columns.tolist(), close.index, close)
        except Exception:
            financial_frames = None
    factors = build_factor_frames(close, am20, turn20,
                                  financial=financial_frames,
                                  asset_type=asset_type,
                                  volume=volume_sig)
    _inject_pred_factor(factors, close, factor, factor_weights)
    _ensure_ma_cross_factor(factors, close, factor)
    last_date = close.index[-1]
    if factor_weights:
        combo = build_composite_factor(
            close, am20, turn20, factor_weights, factor_directions,
            factor_builder=(lambda c, a, t: build_factor_frames(
                c, a, t, financial=financial_frames,
                asset_type=asset_type)),
            extra_factors={"pred": factors["pred"]} if "pred" in factors else None)
        row = combo.iloc[-1]
        factor = "composite"
    else:
        row = factors[factor].iloc[-1]
    am_row = am20.iloc[-1]
    turn_row = turnover.iloc[-1]
    close_row = close.iloc[-1]

    cand = row.dropna()
    valid = (am_row[cand.index].notna() & (turn_row[cand.index] > 0)
             & close_row[cand.index].notna())
    if adx_row is not None:
        valid = valid & (adx_row[cand.index] >= adx_filter)
    cand = cand[valid].sort_values(ascending=ascending)
    top = cand.tail(top_n) if not ascending else cand.head(top_n)
    if long_short:
        short_count = short_n or top_n
        bottom = cand.head(short_count) if not ascending else cand.tail(short_count)
        bottom = bottom[~bottom.index.isin(top.index)]
        rows = []
        for c, s in top.items():
            rows.append({"code": c, "score": s, "side": "多"})
        for c, s in bottom.items():
            rows.append({"code": c, "score": s, "side": "空"})
        top_idx = pd.Index([r["code"] for r in rows])
        out = pd.DataFrame({
            "code": top_idx,
            "side": [r["side"] for r in rows],
            "score": [r["score"] for r in rows],
            "close": [close_row.get(c, np.nan) for c in top_idx],
            "turnover": [turn_row.get(c, np.nan) for c in top_idx],
        }).reset_index(drop=True)
        return out, last_date

    out = pd.DataFrame({
        "code": top.index,
        "score": top.values,
        "close": [close_row.get(c, np.nan) for c in top.index],
        "turnover": [turn_row.get(c, np.nan) for c in top.index],
    }).reset_index(drop=True)
    return out, last_date


