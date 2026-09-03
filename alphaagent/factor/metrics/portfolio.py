"""metrics 子模块：分位组合评估与选股重合率。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ._core import spearman_ic


def _rebalance_dates(ts_list: list, rebalance: str) -> set:
    """按调仓周期标记调仓日：daily=每日；weekly=每周最后交易日；monthly=每月最后交易日。"""
    if rebalance == "daily":
        return set(ts_list)
    idx = pd.DatetimeIndex(ts_list)
    s = pd.Series(idx, index=idx)
    if rebalance == "weekly":
        groups = [s.groupby([idx.year, idx.isocalendar().week.astype(int)]).max()]
    elif rebalance == "monthly":
        groups = [s.groupby([idx.year, idx.month]).max()]
    else:
        raise ValueError(f"unknown rebalance freq: {rebalance!r}")
    out: set = set()
    for g in groups:
        out.update(g.tolist())
    return out


def topn_selection_overlap(
    factor: pd.Series,
    *,
    time_level: str = "datetime",
    top_n: int | None = None,
    selection_pct: float | None = None,
    rebalance: str = "daily",
) -> float:
    """TopN 选股名单的相邻调仓重合率（纯排名统计，不涉及价格/收益）。

    支持两种选股模式：
    - 固定 N：传 top_n=30 每日选 30 只
    - 动态百分比：传 selection_pct=0.02 每日选前 2%（适配停牌/涨跌停导致的候选池缩放）
    两者都传时以 selection_pct 为准。
    """
    rb_dates = _rebalance_dates(list(factor.groupby(level=time_level, sort=False).groups.keys()), rebalance)
    prev: set[str] | None = None
    overlap_sum = 0.0
    count = 0
    for ts, f_sub in factor.groupby(level=time_level, sort=False):
        if ts not in rb_dates:
            continue
        xf = f_sub.to_numpy(dtype=np.float64, copy=False)
        inst = np.asarray(f_sub.index.get_level_values("instrument"))
        valid = np.isfinite(xf)
        n_valid = int(valid.sum())
        if selection_pct is not None:
            n_pick = max(1, int(np.ceil(n_valid * float(selection_pct))))
        else:
            n_pick = int(top_n or 30)
        if n_valid < max(n_pick, 10):
            continue
        order = np.argsort(-xf[valid], kind="stable")[:n_pick]
        current = set(inst[valid][order].tolist())
        if prev is not None:
            overlap_sum += len(current & prev) / max(len(current), 1)
            count += 1
        prev = current
    return round(overlap_sum / count, 6) if count else float("nan")


def quantile_portfolio_metrics(
    factor: pd.Series,
    label: pd.Series,
    *,
    time_level: str = "datetime",
    n_groups: int = 10,
    min_stocks: int = 30,
    cost_bps: float = 15.0,
    annualization_factor: float = 252.0,
    direction: int | None = None,
    holding_days: int = 1,
    _day_slices=None,
    _fast_equal_freq_codes=None,
) -> dict[str, Any]:
    """分位组合评估（纯多头，A 股口径，方向自适应 + 换手计成本）。

    与旧实现的差异：
    - 方向自适应：未显式传 ``direction`` 时按全样本截面 Rank IC 符号选多头侧
      （RankIC>=0 买最高组 Q_N；<0 买最低组 Q1）。``top_group_*`` 键恒指
      多头组合——负 IC 因子不再被系统性判负。
    - 成本按换手计：每日成本 = cost_bps × 单边换手比例（首日建仓计单边全额，
      之后双边），不再每天固定扣全额双边费用。cost_bps=0 时为纯毛值口径。
    - ``monotonicity`` 为方向调整后的单调性（多头侧沿因子方向收益递增为正）；
      未调向的原始高减低单调性见 ``raw_monotonicity``。

    参数 ``direction``：+1 买最高组，-1 买最低组；None 表示自动按 Rank IC 判定。

    返回键（供 profile rules 引用）:
      - top_group_annualized_return / top_group_annualized_excess_return
      - top_group_gross_excess_return（未扣成本的费前超额）
      - top_group_sharpe / top_group_excess_sharpe
      - top_group_max_drawdown / monotonicity / raw_monotonicity
      - direction / long_side / avg_daily_side_turnover / group_means
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("quantile_portfolio_metrics 需要 MultiIndex 面板 (datetime, instrument)")
    if time_level not in factor.index.names:
        raise ValueError(f"索引缺少 level={time_level!r}")

    if direction is None:
        from .ic import cross_sectional_rank_ic
        daily_ric = cross_sectional_rank_ic(factor, label, time_level=time_level, _day_slices=_day_slices)
        ric_vals = daily_ric[np.isfinite(daily_ric.to_numpy(dtype=float, copy=False))]
        direction = 1 if len(ric_vals) == 0 or float(ric_vals.mean()) >= 0 else -1
    direction = 1 if int(direction) >= 0 else -1

    net: list[float] = []
    excess: list[float] = []
    gross_excess: list[float] = []
    high_minus_low: list[float] = []
    grp_sum: dict[int, float] = {}
    grp_cnt: dict[int, int] = {}

    prev_members: set | None = None
    turnover_sum = 0.0
    n_days = 0
    daily_prev_members: set | None = None
    daily_turnover_sum = 0.0
    nav_prev_members: set | None = None
    nav_turnover_sum = 0.0
    nav_n_days = 0
    _ = prev_members, turnover_sum  # 已由 daily_*/nav_* 双轨累计承担（见下）

    f_arr_all = factor.to_numpy(dtype=np.float64, copy=False)
    l_arr_all = label.to_numpy(dtype=np.float64, copy=False)
    inst_all = np.asarray(factor.index.get_level_values("instrument"))
    slices = _day_slices(factor.index, time_level) if _day_slices else None

    def _quantile_portfolio_day(xf, yl, inst):
        mask = np.isfinite(xf) & np.isfinite(yl)
        if int(mask.sum()) < max(min_stocks, n_groups):
            return None
        fac, ret, names = xf[mask], yl[mask], inst[mask]
        bins = _fast_equal_freq_codes(fac, n_groups) if _fast_equal_freq_codes else None
        if bins is None:
            try:
                bins = pd.qcut(fac, n_groups, labels=False, duplicates="drop")
            except ValueError:
                bins = pd.qcut(pd.Series(fac).rank(method="first"), n_groups, labels=False, duplicates="drop")
        b = np.asarray(bins, dtype=float)
        return fac, ret, names, b

    if slices is not None:
        bounds, day_vals = slices
        day_iter = list(zip(day_vals.tolist(), bounds[:-1].tolist(), bounds[1:].tolist()))
    else:
        day_iter = None
    if day_iter is not None:
        grouped_days = (
            (ts, f_arr_all[st:en], l_arr_all[st:en], inst_all[st:en])
            for ts, st, en in day_iter
        )
    else:
        grouped_days = (
            (ts,
             f_sub.to_numpy(dtype=np.float64, copy=False),
             label.xs(ts, level=time_level).to_numpy(dtype=np.float64, copy=False),
             np.asarray(f_sub.index.get_level_values("instrument")))
            for ts, f_sub in factor.groupby(level=time_level, sort=False)
        )

    # label 名义持有期 > 1 时，组合净值按持有期节奏调仓复利（如 label_20d →
    # 每 20 个交易日调仓一次）。否则 20 日收益被逐日重叠计入 20 次：年化/回撤/
    # 夏普全部失真（实测 fundamental 档年化虚高 ~20 倍、回撤虚高至 99%）。
    # 注意：avg_daily_side_turnover 保持日频口径不变（stage_one 换手门的输入，
    # 语义与历史一致）；多日 label 的真实调仓成本由 engine_gate 按档位频率裁决。
    hold = max(1, int(holding_days))
    nav_annualization = annualization_factor / hold
    day_index = -1
    nav_prev_members: set | None = None
    nav_turnover_sum = 0.0
    nav_n_days = 0
    daily_prev_members: set | None = None
    daily_turnover_sum = 0.0

    for ts, xf, yl, inst in grouped_days:
        day_index += 1
        prepared = _quantile_portfolio_day(xf, yl, inst)
        if prepared is None:
            continue
        fac, ret, names, b = prepared
        if not np.isfinite(b).any():
            continue
        k = int(np.nanmax(b)) + 1
        long_bin = k - 1 if direction >= 0 else 0
        short_bin = 0 if direction >= 0 else k - 1
        long_mask_arr = b == long_bin
        if not long_mask_arr.any():
            continue

        members = set(names[long_mask_arr].tolist())

        def _bin_mean(idx: int) -> float:
            sel = b == idx
            return float(ret[sel].mean()) if sel.any() else float("nan")

        long_ret = float(ret[long_mask_arr].mean())
        universe_ret = float(ret.mean())
        high_ret, low_ret = _bin_mean(k - 1), _bin_mean(0)
        if np.isfinite(high_ret) and np.isfinite(low_ret):
            high_minus_low.append(high_ret - low_ret)

        # 组合净值：仅在调仓日（每 hold 个交易日）复利一次
        # 日度换手（信息性，口径与历史一致：stage_one 换手门继续按日换手解读）
        if daily_prev_members is None:
            daily_side = 1.0
        else:
            daily_side = 2.0 * len(members - daily_prev_members) / max(len(members), 1)
        daily_turnover_sum += daily_side
        daily_prev_members = members

        if day_index % hold != 0:
            for g in range(k):
                sel = b == g
                if sel.any():
                    key = g + 1
                    grp_sum[key] = grp_sum.get(key, 0.0) + float(ret[sel].sum())
                    grp_cnt[key] = grp_cnt.get(key, 0) + int(sel.sum())
            n_days += 1
            continue

        if nav_prev_members is None:
            nav_side = 1.0  # 建仓：单边买入
        else:
            changed = len(members - nav_prev_members) / max(len(members), 1)
            nav_side = 2.0 * changed  # 双边
        nav_turnover_sum += nav_side
        nav_prev_members = members
        day_cost = cost_bps / 10_000.0 * nav_side

        net.append(long_ret - day_cost)
        excess.append((long_ret - universe_ret) - day_cost)
        gross_excess.append(long_ret - universe_ret)

        for g in range(k):
            sel = b == g
            if sel.any():
                key = g + 1
                grp_sum[key] = grp_sum.get(key, 0.0) + float(ret[sel].sum())
                grp_cnt[key] = grp_cnt.get(key, 0) + int(sel.sum())
        n_days += 1
        nav_n_days += 1

    if n_days == 0:
        return {
            "n_groups": int(n_groups),
            "available": False,
            "error": "insufficient_data",
            "direction": direction,
        }

    grp_keys = sorted(grp_sum)
    group_means_vals = np.array([grp_sum[k2] / grp_cnt[k2] for k2 in grp_keys], dtype=np.float64)
    raw_monotonicity = float("nan")
    if len(group_means_vals) >= 3:
        ranks_g = np.arange(1, len(group_means_vals) + 1, dtype=np.float64)
        raw_monotonicity = float(spearman_ic(ranks_g, group_means_vals, min_pairs=3))

    def _compound_ann(series: list[float]) -> float:
        arr = np.asarray(series, dtype=np.float64)
        if np.any(arr <= -1.0):
            return float("nan")
        # nav_annualization 已按持有期缩放（label_20d → 每期 20 个交易日）
        return float(np.prod(1.0 + arr) ** (nav_annualization / len(arr)) - 1.0)

    def _sharpe(series: list[float]) -> float:
        arr = np.asarray(series, dtype=np.float64)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        if std > 0 and np.isfinite(std):
            return float(arr.mean() / std * math.sqrt(nav_annualization))
        return float("nan")

    def _max_drawdown(series: list[float]) -> float:
        arr = np.asarray(series, dtype=np.float64)
        nav = np.cumprod(1.0 + arr)
        running_max = np.maximum.accumulate(nav)
        drawdowns = 1.0 - nav / running_max
        return float(drawdowns.max()) if len(drawdowns) else float("nan")

    return {
        "n_groups": int(n_groups),
        "available": True,
        "direction": direction,
        "long_side": f"Q{n_groups}" if direction >= 0 else "Q1",
        "cost_model": "per_rebalance_turnover",
        "cost_bps_per_side": float(cost_bps),
        "holding_days": hold,
        # 日频换手：口径与历史一致（stage_one 换手门的输入）
        "avg_daily_side_turnover": round(daily_turnover_sum / max(n_days, 1), 4),
        # 调仓日换手（每 hold 个交易日一次的真实调仓成本）
        "avg_rebalance_side_turnover": round(nav_turnover_sum / max(nav_n_days, 1), 4),
        "n_rebalances": int(nav_n_days),
        "n_days": int(n_days),
        "top_group_annualized_return": _compound_ann(net),
        "top_group_annualized_excess_return": _compound_ann(excess),
        "top_group_gross_excess_return": _compound_ann(gross_excess),
        "top_group_sharpe": _sharpe(net),
        "top_group_excess_sharpe": _sharpe(excess),
        "top_group_max_drawdown": _max_drawdown(net),
        "top_group_daily_mean": float(np.mean(net)) if net else float("nan"),
        "bottom_group_gross_annualized_return": (
            _compound_ann([-v for v in high_minus_low])
            if direction >= 0 else _compound_ann(list(high_minus_low))
        ),
        "group_means": {int(k2): float(grp_sum[k2] / grp_cnt[k2]) for k2 in grp_keys},
        "monotonicity": (
            float(direction * raw_monotonicity) if np.isfinite(raw_monotonicity) else float("nan")
        ),
        "raw_monotonicity": raw_monotonicity,
        "spread_daily_mean": float(np.mean(high_minus_low)) if high_minus_low else float("nan"),
        "spread_annualized": _compound_ann(high_minus_low) if high_minus_low else float("nan"),
    }
