"""IC 衰减与 IC 分布诊断（因子实验室图表数据，纯函数）。

- ``forward_close_to_close_label``：与 ``alphaagent.data.panel`` 的
  ``label_{N}d_close_to_close`` 完全同口径的 close→close 前瞻收益
  （T+1 收盘 → T+(N+1) 收盘），供 IC 衰减曲线按任意 horizon 现算。
- ``decay_horizons_for``：按 label 名义持有期选择衰减曲线 horizon 集。
- ``rank_ic_decay_summary``：逐日 RankIC → 均值 / 去重叠 ICIR
  （与 ``cs_ic_summary`` 同口径：持有期 >1 时按持有期节奏重采样）。
- ``ic_histogram``：逐日 IC 序列直方图分箱。

仅因子实验室 include_charts=True 路径消费；挖掘批量评估不触发。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def forward_close_to_close_label(adj_close: pd.Series, hold_days: int) -> pd.Series:
    """close→close 前瞻收益：T+1 收盘进、T+(hold_days+1) 收盘出。

    须在按 instrument 分组、组内时间升序的 Series 上调用
    （panel 按 (datetime, instrument) 排序时 groupby 的组内序即时间序）。
    """
    entry = adj_close.shift(-1)
    exit_ = adj_close.shift(-(int(hold_days) + 1))
    denom = entry.replace(0, np.nan)
    return (exit_ - entry) / denom


def panel_forward_label(panel: pd.DataFrame, hold_days: int) -> pd.Series:
    """在 panel 上按 instrument 分组计算 close→close 前瞻收益，行序与 panel 一致。"""
    return panel.groupby(level="instrument", sort=False)["adj_close"].transform(
        forward_close_to_close_label, hold_days=int(hold_days)
    )


def decay_horizons_for(holding_days: int) -> tuple[int, ...]:
    """按 label 名义持有期选 horizon 集：曲线覆盖设计持有期两侧并外延。"""
    hold = max(1, int(holding_days))
    if hold <= 1:
        return (1, 2, 3, 5, 10, 20)
    if hold <= 10:
        return (1, 5, 10, 20, 40)
    return (5, 10, 20, 40, 60)


def rank_ic_decay_summary(
    daily_rank_ic: pd.Series, *, holding_days: int
) -> dict[str, float | int]:
    """逐日 RankIC → {mean_ic, ic_ir, n_days}。

    持有期 >1 时逐日 IC 来自重叠前瞻收益，std 被低估导致 ICIR 虚高；
    与 ``cs_ic_summary`` 同口径按持有期节奏重采样（每 hold 个交易日取一点）。
    """
    hold = max(1, int(holding_days))
    s = daily_rank_ic
    if hold > 1 and len(s) > hold:
        s = s.iloc[::hold]
    vals = s.to_numpy(dtype=float, copy=False)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / std if std and np.isfinite(std) and std > 0 else float("nan")
    return {"mean_ic": mean, "ic_ir": icir, "n_days": n}


def ic_histogram(daily_series: pd.Series | None, *, bins: int = 40) -> dict[str, list[Any]]:
    """逐日 IC 序列 → {edges, counts}；非有限值剔除，空序列返回空 bins。"""
    if daily_series is None or len(daily_series) == 0:
        return {"edges": [], "counts": []}
    vals = daily_series.to_numpy(dtype=float, copy=False)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(vals, bins=int(bins))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}
