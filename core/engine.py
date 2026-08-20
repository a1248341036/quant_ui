from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from .limit import build_limit_flags
from .metrics import compute_metrics, drawdown_series
from .assets import AssetExecutionProfile, STOCK_PROFILE
from .execution import ETFExecutionAdapter, FundNavExecutionAdapter, StockExecutionAdapter
from .selection import PortfolioBuilder


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
        }
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
    buy_cost: float = 0.0008,
    sell_cost: float = 0.0013,
    amount_q: float = 0.3,
    affordable: bool = True,
    lot_size: int = 100,
    warmup_days: int | None = None,
    cash_mode: bool = True,
    limit_flags: bool = True,
    slippage_bps: float = 0.0,
    max_participation: float = 0.0,
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
    execution_profile: AssetExecutionProfile | None = None,
    share_classes: dict[str, str] | None = None,
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
    profile = execution_profile or STOCK_PROFILE
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

    def pivot(col: str) -> pd.DataFrame:
        # pivot_table 会丢弃全 NaN 行（如 am20 前 20 个交易日），需统一回交易日历
        return sub.pivot_table(index="date", columns="code", values=col,
                               aggfunc="last", observed=True).reindex(cal).sort_index()

    close = pivot("close")
    open_ = pivot("open")
    turnover = pivot("turnover")
    am20 = pivot("am20")
    turn20 = pivot("turn20")
    high = pivot("high") if "high" in sub.columns else None
    low = pivot("low") if "low" in sub.columns else None
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
                                   asset_type=_asset_type)

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
            from .performance import factor_quality, slice_quality
            quality = factor_quality(combo, close, horizon=20, groups=5, min_n=10)
    else:
        fmat = factors[factor].values
        quality = None
        if analyze:
            from .performance import factor_quality, slice_quality
            quality = factor_quality(factors[factor], close, horizon=20, groups=5, min_n=10)

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
        limit_up, limit_down, _, _ = build_limit_flags(close, open_)
    # 预热模式：窗口起点不继承预热段持仓，也不在窗口起点当天调仓，
    # 与"全量算因子、窗口起点从零开始"的旧脚本语义一致。
    exec_set = ({i for i in exec_dates if i != start_idx}
                if (warmup_days and not use_cash) else set(exec_dates))

    nav = np.ones(T)
    bench = np.ones(T)
    portfolio_builder = PortfolioBuilder(codes_used, industry_map, industry_cap)
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
                    chosen_list = portfolio_builder.rank_select(
                        cand, scores, ascending, top_n, selection_mode,
                        selection_pct, min_positions, max_positions,
                    )
                    if chosen_list:
                        targets.update(portfolio_builder.equal_weights(chosen_list))
                        last_chosen = [codes_used[k] for k in chosen_list]
                    if regime_adx is not None and adx_mat is not None:
                        ma = float(np.nanmedian(adx_mat[sig]))
                        if np.isfinite(ma) and ma < regime_adx:
                            targets = {k: v * regime_scale for k, v in targets.items()}

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

    if quality is not None:
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

    return {
        "nav": nav_s,
        "bench": bench_s,
        "drawdown": drawdown_series(nav_s),
        "metrics": compute_metrics(nav_s),
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

    def pivot(col: str) -> pd.DataFrame:
        return sub.pivot_table(index="date", columns="code", values=col,
                               aggfunc="last", observed=True).reindex(cal).sort_index()

    close = pivot("close")
    am20 = pivot("am20")
    turn20 = pivot("turn20")
    turnover = pivot("turnover")
    high = pivot("high") if "high" in sub.columns else None
    low = pivot("low") if "low" in sub.columns else None
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
                                  asset_type=asset_type)
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


