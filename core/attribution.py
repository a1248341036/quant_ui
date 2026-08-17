from __future__ import annotations

"""Brinson 归因：把组合相对基准的超额收益分解为行业配置与个股选择。

输入来自事件引擎回测的每日权重历史（weight_history），
基准取股票池每日有效股票等权。按自然月做一期归因：

  配置效应 = (组合行业权重 - 基准行业权重) × 基准行业收益
  选择效应 = 基准行业权重 × (组合行业收益 - 基准行业收益)
  交互效应 = (组合行业权重 - 基准行业权重) × (组合行业收益 - 基准行业收益)

输出逐月逐行业明细 + 全期行业汇总。
"""

import numpy as np
import pandas as pd


def _month_edges(dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, list[int]]]:
    """按自然月分组：返回 (month_start, 月内索引列表)。"""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(dates):
        groups.setdefault((d.year, d.month), []).append(i)
    return [(dates[idxs[0]], idxs) for idxs in groups.values()]


def brinson_attribution(
    panel: pd.DataFrame,
    codes: list[str],
    weight_history: list[dict[str, float]],
    dates: pd.DatetimeIndex,
    industry_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按月 Brinson 归因。

    weight_history 与 dates 对齐，每个元素是 {code: 市值权重}。
    industry_map: {code: 行业}。
    返回 (detail, summary)：
    - detail：逐月逐行业明细（month/industry/权重/收益/三效应）
    - summary：全期按行业汇总（效应加总 + 平均权重）
    """
    if len(weight_history) != len(dates):
        raise ValueError("weight_history 长度必须与 dates 一致")
    sub = panel[panel["code"].isin(codes)].copy()
    cal = pd.DatetimeIndex(sorted(sub["date"].unique()))
    close = (sub.pivot_table(index="date", columns="code", values="close",
                             aggfunc="last", observed=True)
             .reindex(cal).sort_index())
    dates = dates[dates.isin(cal)] if hasattr(dates, "isin") else dates
    # 只保留回测窗口内的日历，weight_history 按窗口对齐
    start_ts, end_ts = dates.min(), dates.max()
    close = close[(close.index >= start_ts) & (close.index <= end_ts)]
    close_mat = close.values
    cols = close.columns.tolist()
    col_idx = {c: i for i, c in enumerate(cols)}

    industries = sorted({industry_map.get(c, "其他") for c in cols})
    detail_rows: list[dict] = []

    for month_start, idxs in _month_edges(dates):
        t0 = idxs[0]
        t1 = idxs[-1]
        w_p = weight_history[t0]
        if not w_p:
            continue  # 月初空仓，跳过该月
        rets = close_mat[t1] / close_mat[t0] - 1.0
        valid = np.isfinite(rets) & (close_mat[t0] > 0)
        # 基准：池内有效股票等权
        bench_weights = np.zeros(len(cols))
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        bench_weights[valid] = 1.0 / n_valid

        for ind in industries:
            members = [col_idx[c] for c in cols
                       if industry_map.get(c, "其他") == ind]
            if not members:
                continue
            # 组合行业权重 / 收益
            cw = sum(w_p.get(cols[k], 0.0) for k in members)
            cw_num = sum(w_p.get(cols[k], 0.0) * (rets[k] if valid[k] else 0.0)
                         for k in members)
            combo_ret = (cw_num / cw) if cw > 0 else 0.0
            # 基准行业权重 / 收益（池内等权）
            bw = float(bench_weights[members].sum())
            bench_ret = float(np.nanmean(np.where(valid[members],
                                                  rets[members], np.nan)))
            if not np.isfinite(bench_ret):
                bench_ret = 0.0
            alloc = (cw - bw) * bench_ret
            select = bw * (combo_ret - bench_ret)
            inter = (cw - bw) * (combo_ret - bench_ret)
            detail_rows.append({
                "month": month_start.strftime("%Y-%m"),
                "industry": ind,
                "combo_weight": cw,
                "bench_weight": bw,
                "combo_ret": combo_ret,
                "bench_ret": bench_ret,
                "allocation": alloc,
                "selection": select,
                "interaction": inter,
                "total": alloc + select + inter,
            })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby("industry", as_index=False).agg(
        allocation=("allocation", "sum"),
        selection=("selection", "sum"),
        interaction=("interaction", "sum"),
        total=("total", "sum"),
        avg_combo_weight=("combo_weight", "mean"),
        avg_bench_weight=("bench_weight", "mean"),
    ).sort_values("total", ascending=False).reset_index(drop=True)
    summary["total_pct"] = summary["total"] * 100
    return detail, summary
