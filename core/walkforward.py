from __future__ import annotations

"""Walk-forward 样本外验证 + 参数稳健性分析。

核心思路：把时间轴切成多个连续窗口，每个窗口独立从零回测，
看策略在「不同市场段落」下的指标分布，而不是只看一段的总收益。
参数稳健性：对参数网格逐组跑 walk-forward，按均值夏普/收益胜率/
最差窗口等排序，输出参数热力图，避免「单点最优 = 过拟合」。
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from .event_engine import EventStrategy, GoldenCrossStrategy, run_event_backtest


def split_windows(start: str, end: str, n_folds: int = 4) -> list[tuple[str, str]]:
    """把 [start, end] 均匀切成 n_folds 个连续窗口。"""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    edges = pd.date_range(start_ts, end_ts, periods=n_folds + 1)
    return [(edges[i].strftime("%Y-%m-%d"), edges[i + 1].strftime("%Y-%m-%d"))
            for i in range(n_folds)]


def make_golden_cross(short: int = 5, long: int = 20, top_n: int = 3,
                      max_weight: float = 0.5) -> type[GoldenCrossStrategy]:
    """按参数生成双均线金叉策略类（参数化工厂）。"""
    return type("GoldenCross", (GoldenCrossStrategy,), {
        "short": int(short), "long": int(long),
        "top_n": int(top_n), "max_weight": float(max_weight),
    })


def _metrics_row(fold: int, ws: str, we: str, res: dict) -> dict:
    m = res["metrics"]
    return {
        "fold": fold,
        "start": ws,
        "end": we,
        "n_days": int(len(res["nav"])),
        "total": float(m["总收益"]),
        "annual": float(m["年化收益"]),
        "sharpe": float(m["夏普"]),
        "mdd": float(m["最大回撤"]),
        "calmar": float(m["卡玛"]),
        "win_rate": float(m["胜率"]),
        "end_nav": float(res["nav"].iloc[-1]),
    }


def walk_forward_event(
    panel: pd.DataFrame,
    codes: list[str],
    strategy_factory: Callable[[], EventStrategy],
    start: str,
    end: str,
    capital: float,
    n_folds: int = 4,
    warmup_days: int = 400,
    limit_flags: bool = True,
    amount_q: float = 0.3,
) -> pd.DataFrame:
    """对事件策略做滚动样本外回测，返回逐窗口指标明细。"""
    rows: list[dict] = []
    for i, (ws, we) in enumerate(split_windows(start, end, n_folds), 1):
        strategy = strategy_factory()
        res = run_event_backtest(
            panel, codes, type(strategy), ws, we, capital,
            warmup_days=warmup_days, limit_flags=limit_flags, amount_q=amount_q,
        )
        rows.append(_metrics_row(i, ws, we, res))
    return pd.DataFrame(rows)


def walk_forward_factor(
    panel: pd.DataFrame,
    codes: list[str],
    factor: str,
    ascending: bool,
    start: str,
    end: str,
    capital: float,
    top_n: int = 3,
    freq: str = "monthly",
    n_folds: int = 4,
    warmup_days: int = 400,
    amount_q: float = 0.3,
    affordable: bool = True,
) -> pd.DataFrame:
    """对旧因子轮动策略做 walk-forward，返回逐窗口指标明细。"""
    from .engine import run_backtest

    rows: list[dict] = []
    for i, (ws, we) in enumerate(split_windows(start, end, n_folds), 1):
        res = run_backtest(
            panel, codes, factor, ascending, ws, we, capital,
            top_n=top_n, freq=freq, warmup_days=warmup_days,
            amount_q=amount_q, affordable=affordable,
        )
        rows.append(_metrics_row(i, ws, we, res))
    return pd.DataFrame(rows)


def _rolling_summary(windows: pd.DataFrame) -> dict:
    """滚动训练-测试的汇总行。"""
    return {
        "mode": "rolling",
        "n_windows": int(len(windows)),
        "trained_windows": int(windows["trained"].sum())
        if "trained" in windows else 0,
        "mean_sharpe": float(windows["sharpe"].mean()),
        "median_sharpe": float(windows["sharpe"].median()),
        "win_rate": float((windows["total"] > 0).mean()),
        "mean_total": float(windows["total"].mean()),
        "worst_total": float(windows["total"].min()),
        "best_total": float(windows["total"].max()),
    }


def _train_prefix(folds: list[tuple[str, str]], i: int, start: str) -> str:
    """第 i 个测试窗口之前的训练区间终点（前一天）。"""
    if i <= 0:
        return start
    tws = pd.Timestamp(folds[i][0])
    return (tws - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def rolling_train_test_factor(
    panel: pd.DataFrame,
    codes: list[str],
    factor: str,
    ascending: bool,
    start: str,
    end: str,
    capital: float,
    top_n_list: list[int] | tuple[int, ...] = (3, 5),
    freq_list: list[str] | tuple[str, ...] = ("monthly",),
    n_folds: int = 4,
    warmup_days: int = 400,
    amount_q: float = 0.3,
    affordable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """滚动训练-测试（因子策略）。

    每个测试窗口之前，用 [start, 窗口前一天] 的全部历史在参数网格
    （top_n × freq）上选最优（按夏普），再跑当前测试窗口做样本外验证。
    第一个窗口没有训练前缀，用默认参数，标记 trained=False。

    返回 (windows, summary, param_history)：
    - windows：逐测试窗口指标 + 训练区间 + 所选参数
    - summary：单行汇总
    - param_history：每窗口训练结果（train_sharpe / 所选参数）
    """
    from .engine import run_backtest

    folds = split_windows(start, end, n_folds)
    rows: list[dict] = []
    hist: list[dict] = []
    for i, (tws, twe) in enumerate(folds):
        train_end = _train_prefix(folds, i, start)
        params = {"top_n": int(top_n_list[0]), "freq": str(freq_list[0])}
        train_sharpe = np.nan
        if i > 0 and train_end >= start:
            best: tuple[tuple, float] | None = None
            for tn in top_n_list:
                for fr in freq_list:
                    tres = run_backtest(
                        panel, codes, factor, ascending, start, train_end,
                        capital, top_n=int(tn), freq=str(fr),
                        warmup_days=warmup_days, amount_q=amount_q,
                        affordable=affordable,
                    )
                    sharpe = float(tres["metrics"].get("夏普", np.nan))
                    if not np.isfinite(sharpe):
                        sharpe = -np.inf
                    if best is None or sharpe > best[1]:
                        best = ((int(tn), str(fr)), sharpe)
            if best is not None:
                params = {"top_n": best[0][0], "freq": best[0][1]}
                train_sharpe = best[1]
        res = run_backtest(
            panel, codes, factor, ascending, tws, twe, capital,
            top_n=params["top_n"], freq=params["freq"],
            warmup_days=warmup_days, amount_q=amount_q, affordable=affordable,
        )
        row = _metrics_row(i + 1, tws, twe, res)
        row.update({
            "train_start": start,
            "train_end": train_end,
            "chosen_top_n": params["top_n"],
            "chosen_freq": params["freq"],
            "trained": i > 0,
        })
        rows.append(row)
        hist.append({
            "fold": i + 1,
            "train_start": start,
            "train_end": train_end,
            "test_start": tws,
            "test_end": twe,
            "chosen_top_n": params["top_n"],
            "chosen_freq": params["freq"],
            "train_sharpe": train_sharpe,
        })
    windows = pd.DataFrame(rows)
    summary = pd.DataFrame([_rolling_summary(windows)])
    return windows, summary, pd.DataFrame(hist)


def rolling_train_test_event(
    panel: pd.DataFrame,
    codes: list[str],
    start: str,
    end: str,
    capital: float,
    short_list: list[int] | tuple[int, ...] = (3, 5),
    long_list: list[int] | tuple[int, ...] = (10, 20),
    n_folds: int = 4,
    top_n: int = 3,
    max_weight: float = 0.5,
    warmup_days: int = 400,
    limit_flags: bool = True,
    amount_q: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """滚动训练-测试（双均线金叉事件策略）。

    每个测试窗口之前，在 [start, 窗口前一天] 上跑 golden_cross_sweep
    选择 short/long（按跨训练窗均值夏普），再用所选参数跑当前测试窗口。
    """
    folds = split_windows(start, end, n_folds)
    rows: list[dict] = []
    hist: list[dict] = []
    for i, (tws, twe) in enumerate(folds):
        train_end = _train_prefix(folds, i, start)
        params = (5, 20)
        train_sharpe = np.nan
        if i > 0 and train_end >= start:
            s, _, _ = golden_cross_sweep(
                panel, codes, start, train_end, capital,
                short_list=list(short_list), long_list=list(long_list),
                n_folds=min(2, max(1, n_folds)), top_n=top_n,
                max_weight=max_weight, warmup_days=warmup_days,
                limit_flags=limit_flags, amount_q=amount_q,
            )
            if len(s):
                best = s.iloc[0]
                params = (int(best["short"]), int(best["long"]))
                train_sharpe = float(best["mean_sharpe"])
        strategy_cls = make_golden_cross(params[0], params[1], top_n=top_n,
                                         max_weight=max_weight)
        res = run_event_backtest(
            panel, codes, strategy_cls, tws, twe, capital,
            warmup_days=warmup_days, limit_flags=limit_flags, amount_q=amount_q,
        )
        row = _metrics_row(i + 1, tws, twe, res)
        row.update({
            "train_start": start,
            "train_end": train_end,
            "chosen_short": params[0],
            "chosen_long": params[1],
            "trained": i > 0,
        })
        rows.append(row)
        hist.append({
            "fold": i + 1,
            "train_start": start,
            "train_end": train_end,
            "test_start": tws,
            "test_end": twe,
            "chosen_short": params[0],
            "chosen_long": params[1],
            "train_sharpe": train_sharpe,
        })
    windows = pd.DataFrame(rows)
    summary = pd.DataFrame([_rolling_summary(windows)])
    return windows, summary, pd.DataFrame(hist)
def _aggregate(windows: pd.DataFrame, params: dict) -> dict:
    return {
        **params,
        "n_windows": int(len(windows)),
        "mean_total": float(windows["total"].mean()),
        "median_total": float(windows["total"].median()),
        "std_total": float(windows["total"].std(ddof=1)) if len(windows) > 1 else 0.0,
        "mean_sharpe": float(windows["sharpe"].mean()),
        "median_sharpe": float(windows["sharpe"].median()),
        "std_sharpe": float(windows["sharpe"].std(ddof=1)) if len(windows) > 1 else 0.0,
        "mean_mdd": float(windows["mdd"].mean()),
        "worst_total": float(windows["total"].min()),
        "best_total": float(windows["total"].max()),
        "win_rate": float((windows["total"] > 0).mean()),
    }


def golden_cross_sweep(
    panel: pd.DataFrame,
    codes: list[str],
    start: str,
    end: str,
    capital: float,
    short_list: list[int],
    long_list: list[int],
    n_folds: int = 4,
    top_n: int = 3,
    max_weight: float = 0.5,
    warmup_days: int = 400,
    limit_flags: bool = True,
    amount_q: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """双均线金叉参数网格 walk-forward 扫描。

    返回 (summary, heatmap, windows)：
    - summary：每个参数组合的跨窗口汇总指标，按 mean_sharpe 降序
    - heatmap：short x long 的均值夏普透视表
    - windows：逐参数逐窗口明细
    """
    summaries: list[dict] = []
    windows_list: list[pd.DataFrame] = []
    for s in short_list:
        for l in long_list:
            if s >= l:
                continue
            fac = make_golden_cross(s, l, top_n=top_n, max_weight=max_weight)
            wf = walk_forward_event(
                panel, codes, fac, start, end, capital,
                n_folds=n_folds, warmup_days=warmup_days,
                limit_flags=limit_flags, amount_q=amount_q,
            )
            wf.insert(0, "short", s)
            wf.insert(1, "long", l)
            windows_list.append(wf)
            summaries.append(_aggregate(wf, {"short": s, "long": l}))

    windows = pd.concat(windows_list, ignore_index=True) if windows_list else \
        pd.DataFrame(columns=["short", "long", "fold", "start", "end", "n_days",
                              "total", "annual", "sharpe", "mdd", "calmar",
                              "win_rate", "end_nav"])
    summary = pd.DataFrame(summaries).sort_values(
        ["mean_sharpe", "mean_total"], ascending=[False, False]).reset_index(drop=True)
    heatmap = (summary.pivot_table(index="short", columns="long",
                                   values="mean_sharpe")
               if len(summary) else pd.DataFrame())
    return summary, heatmap, windows
