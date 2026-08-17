# -*- coding: utf-8 -*-
"""绩效与因子质量分析：IC / 分组收益 / QuantStats 报告。

P0 轻量实现，直接消费现有回测/舆情回测的 NAV 与因子矩阵。
QuantStats 缺失时只降级跳过 HTML 报告，IC/分组分析不依赖它。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """截面未来 horizon 个交易日收益（不含当日，天然避免前视）。"""
    return close.shift(-horizon) / close - 1.0


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_n: int = 10) -> pd.Series:
    """逐日截面 Spearman Rank IC。"""
    out: dict[pd.Timestamp, float] = {}
    idx = factor.index.intersection(fwd.index)
    for d in idx:
        f = factor.loc[d]
        r = fwd.loc[d]
        mask = f.notna() & r.notna() & np.isfinite(f.astype(float)) & np.isfinite(r.astype(float))
        if int(mask.sum()) < min_n:
            continue
        fr, rr = f[mask].rank(), r[mask].rank()
        if fr.nunique() < 2 or rr.nunique() < 2:
            continue
        ic = fr.corr(rr)
        if np.isfinite(ic):
            out[d] = float(ic)
    return pd.Series(out, name="rank_ic")


def ic_summary(ic: pd.Series) -> dict:
    """IC 序列汇总：均值 / 标准差 / ICIR / t 值 / 胜率。"""
    ic = ic.dropna()
    n = int(len(ic))
    if n == 0:
        return {"n": 0, "mean_ic": None, "std_ic": None, "icir": None,
                "t_stat": None, "win_rate": None, "positive_days": 0}
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else np.nan
    ir = mean / std if std and np.isfinite(std) else np.nan
    t = mean / (std / np.sqrt(n)) if std and np.isfinite(std) else np.nan
    return {
        "n": n,
        "mean_ic": mean,
        "std_ic": None if not np.isfinite(std) else std,
        "icir": None if not np.isfinite(ir) else ir,
        "t_stat": None if not np.isfinite(t) else t,
        "win_rate": float((ic > 0).mean()),
        "positive_days": int((ic > 0).sum()),
    }


def group_analysis(factor: pd.DataFrame, fwd: pd.DataFrame,
                   groups: int = 5, min_n: int = 20) -> pd.DataFrame:
    """逐日按因子分位分组，统计各组未来收益均值（长表）。"""
    rows = []
    idx = factor.index.intersection(fwd.index)
    for d in idx:
        f = factor.loc[d].astype(float)
        r = fwd.loc[d].astype(float)
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if int(mask.sum()) < max(min_n, groups * 2):
            continue
        ff, rr = f[mask], r[mask]
        try:
            q = pd.qcut(ff.rank(method="first"), groups, labels=False) + 1
        except ValueError:
            continue
        g = rr.groupby(q).mean()
        rows.append(pd.DataFrame({"date": d, "group": g.index.astype(int),
                                  "fwd_ret": g.values}))
    if not rows:
        return pd.DataFrame(columns=["date", "group", "fwd_ret"])
    return pd.concat(rows, ignore_index=True)


def group_summary(gt: pd.DataFrame, horizon: int = 20) -> dict:
    """分组结果摘要：各组平均收益 + 多空价差（top-bottom）。"""
    if gt.empty:
        return {"groups": [], "spread": None, "spread_pa": None}
    means = gt.groupby("group")["fwd_ret"].mean()
    groups_out = [{"group": int(g), "mean_fwd_ret": float(v)}
                  for g, v in means.items()]
    spread = float(means.iloc[-1] - means.iloc[0]) if len(means) >= 2 else None
    # 未来 horizon 交易日收益折算年化（244 个交易日）
    spread_pa = ((1 + spread) ** (244 / horizon) - 1) if spread is not None else None
    return {"groups": groups_out, "spread": spread, "spread_pa": spread_pa}


def factor_quality(factor: pd.DataFrame, close: pd.DataFrame,
                   horizon: int = 20, groups: int = 5,
                   min_n: int = 10) -> dict:
    """一次算完 IC + 分组，返回汇总 dict（供引擎/API/报告复用）。"""
    fwd = forward_returns(close, horizon=horizon)
    ic = rank_ic(factor, fwd, min_n=min_n)
    gt = group_analysis(factor, fwd, groups=groups, min_n=min_n)
    return {
        "horizon": horizon,
        "ic": ic_summary(ic),
        "group": group_summary(gt, horizon=horizon),
        "ic_series": ic,
        "group_table": gt,
    }


def slice_quality(quality: dict, dates: pd.DatetimeIndex) -> dict:
    """把预热窗口算出的 IC/分组结果裁到实际输出区间。"""
    ic = quality["ic_series"].loc[quality["ic_series"].index.intersection(dates)]
    gt = quality["group_table"]
    if len(gt):
        gt = gt[gt["date"].isin(dates)].copy()
    return {
        "horizon": quality["horizon"],
        "ic": ic_summary(ic),
        "group": group_summary(gt, horizon=quality["horizon"]),
        "ic_series": ic,
        "group_table": gt,
    }


def quantstats_html(nav: pd.Series, bench: pd.Series | None = None,
                    title: str = "Backtest", out_path: str | Path = "quantstats.html",
                    periods_per_year: int = 244) -> Path | None:
    """生成 QuantStats HTML 报告；未安装 quantstats 时返回 None。"""
    try:
        import quantstats as qs
    except ImportError:
        return None
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        if "Arial" not in {f.name for f in fm.fontManager.ttflist}:
            # quantstats 默认请求 Arial；用系统字体替代并静默 matplotlib 的缺字体告警
            matplotlib.rcParams["font.family"] = "Arial"
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans",
                                                      "WenQuanYi Micro Hei", "sans-serif"]
            import logging
            logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    except Exception:
        pass
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    returns = nav.pct_change().dropna()
    bench_ret = bench.pct_change().dropna() if bench is not None else None
    qs.reports.html(returns, benchmark=bench_ret, output=str(out), title=title,
                    periods_per_year=periods_per_year)
    return out


def build_md_report(name: str, metrics: dict, bench_metrics: dict,
                    quality: dict | None, nav: pd.Series,
                    bench: pd.Series | None, ascending: bool | None = None) -> str:
    """把已有指标 + IC/分组结果拼成一份可读 Markdown。"""
    def f(v, nd=2):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    lines = [f"# 绩效报告：{name}", ""]
    lines += ["## 基础指标", ""]
    lines += ["| 指标 | 策略 | 基准 |", "|---|---|---|"]
    keys = ["总收益", "年化收益", "年化波动", "夏普", "最大回撤", "卡玛", "胜率"]
    for k in keys:
        lines.append(f"| {k} | {f(metrics.get(k))} | {f(bench_metrics.get(k) if bench_metrics else None)} |")
    lines += ["", f"- 起始: {nav.index[0].date()} · 结束: {nav.index[-1].date()}",
              f"- 样本交易日: {len(nav)}"]
    if bench is not None:
        m_total = metrics.get("总收益")
        b_total = bench_metrics.get("总收益") if bench_metrics else None
        lines += [f"- 策略区间收益: {f(m_total * 100 if isinstance(m_total, float) else None, 2)}% · "
                  f"基准: {f(b_total * 100 if isinstance(b_total, float) else None, 2)}%"]
    if quality:
        ic = quality["ic"]
        sign = -1.0 if ascending else 1.0
        ic_adj = ic["mean_ic"] * sign if isinstance(ic["mean_ic"], float) else None
        lines += ["", "## 因子 IC（Spearman）", ""]
        lines += ["| 指标 | 值 |", "|---|---|"]
        lines += [f"| 样本期数 | {ic['n']} |",
                  f"| 平均 IC | {f(ic['mean_ic'])} |",
                  (f"| 方向调整 IC | {f(ic_adj)} |" if ascending is not None else ""),
                  f"| IC 标准差 | {f(ic['std_ic'])} |",
                  f"| ICIR | {f(ic['icir'])} |",
                  f"| t 值 | {f(ic['t_stat'], 2)} |",
                  f"| IC>0 占比 | {f(ic['win_rate'] * 100 if isinstance(ic['win_rate'], float) else None, 1)}% |"]
        g = quality["group"]
        spread = g["spread"]
        spread_label = "G1-G5" if ascending else "G5-G1"
        spread_val = spread * sign if isinstance(spread, float) else None
        spread_pa = ((1 + spread_val) ** (244 / quality["horizon"]) - 1
                     if isinstance(spread_val, float) and spread_val > -1 else None)
        lines += ["", f"## 分组收益（{quality['horizon']} 日未来收益均值）", ""]
        lines += ["| 组 | 平均未来收益 |", "|---|---|"]
        for gr in g["groups"]:
            lines += [f"| G{gr['group']} | {f(gr['mean_fwd_ret'] * 100, 2)}% |"]
        lines += ["", f"- 多空价差 ({spread_label}): {f(spread_val * 100, 2)}% · "
                      f"年化折算: {f(spread_pa * 100, 2)}%"]
        if ascending is not None:
            lines += ["", f"> 策略方向: {'买入低因子组 (ascending)' if ascending else '买入高因子组 (descending)'}；"
                          "方向调整 IC/价差已按该方向换算"]
    lines += ["", "> 自动生成 · 数据来源：本地 panel.parquet"]
    return "\n".join(lines)
