# -*- coding: utf-8 -*-
"""用聚宽官方导出的两个回测 CSV 反推/校验指标口径:
把聚宽的日收益序列灌入候选公式, 对照聚宽面板显示值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")

SAMPLES = [
    {
        "csv": r"C:\Users\zhoubw\Downloads\result_1 (2).csv",
        "name": "post47523",
        "panel": {"策略收益": -4.34, "策略年化收益": -3.22, "超额收益": -20.98,
                  "基准收益": 21.06, "阿尔法": -0.156, "贝塔": 0.755,
                  "夏普比率": -0.269, "胜率": 0.403, "盈亏比": 0.976,
                  "最大回撤": 34.78, "索提诺比率": -0.356, "日均超额收益": -0.06,
                  "超额收益最大回撤": 42.20, "超额收益夏普比率": -0.800,
                  "日胜率": 0.478, "信息比率": -0.736, "策略波动率": 0.268,
                  "基准波动率": 0.138,
                  "最大回撤区间": ("2025/07/24", "2026/05/29")},
    },
    {
        "csv": r"C:\Users\zhoubw\Downloads\result_1 (1).csv",
        "name": "国九",
        "panel": {"策略收益": 11.98, "策略年化收益": 7.29, "超额收益": -15.22,
                  "基准收益": 32.08, "阿尔法": -0.036, "贝塔": 0.459,
                  "夏普比率": 0.111, "胜率": 0.603, "盈亏比": 1.051,
                  "最大回撤": 47.66, "索提诺比率": 0.157, "日均超额收益": -0.02,
                  "超额收益最大回撤": 50.31, "超额收益夏普比率": -0.455,
                  "日胜率": 0.475, "信息比率": -0.384, "策略波动率": 0.296,
                  "基准波动率": 0.212,
                  "最大回撤区间": ("2025/12/19", "2026/07/22")},
    },
]

RF = 0.04
ANN = 250


def daily_from_cum(cum_pct: pd.Series) -> pd.Series:
    """累计%序列 -> 日收益序列(首日=累计/100)。"""
    nav = (1 + cum_pct / 100.0)
    return nav.pct_change().fillna(nav.iloc[0] - 1)


def ann_total(total: float, n: int) -> float:
    return (1 + total) ** (ANN / n) - 1


def max_dd(series: pd.Series):
    running = series.cummax()
    dd = series / running - 1
    trough = dd.idxmin()
    peak = series.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def main() -> int:
    for s in SAMPLES:
        df = pd.read_csv(s["csv"], encoding="gbk")
        df["date"] = pd.to_datetime(df["时间"])
        df = df.set_index("date")
        rp = daily_from_cum(df["策略收益"])
        rb = daily_from_cum(df["基准收益"])
        p = s["panel"]
        n = len(rp)
        total_s = df["策略收益"].iloc[-1] / 100
        total_b = df["基准收益"].iloc[-1] / 100
        print(f"\n===== {s['name']} (n={n}) =====")
        ann_s = ann_total(total_s, n)
        ann_b = ann_total(total_b, n)
        print(f"年化收益  计算 {ann_s:+.4f}  聚宽 {p['策略年化收益']:+.2f}%")
        vol_s = rp.std(ddof=1) * np.sqrt(ANN)
        vol_b = rb.std(ddof=1) * np.sqrt(ANN)
        print(f"策略波动  计算 {vol_s:.4f}  聚宽 {p['策略波动率']:.3f}")
        print(f"基准波动  计算 {vol_b:.4f}  聚宽 {p['基准波动率']:.3f}")
        print(f"夏普      计算 {(ann_s - RF) / vol_s:+.4f}  聚宽 {p['夏普比率']:+.3f}")
        cov = rp.cov(rb)
        var = rb.var(ddof=1)
        beta = cov / var
        alpha = ann_s - (RF + beta * (ann_b - RF))
        print(f"贝塔      计算 {beta:.4f}  聚宽 {p['贝塔']:.3f}")
        print(f"阿尔法    计算 {alpha:+.4f}  聚宽 {p['阿尔法']:+.3f}")
        excess_geo = (1 + total_s) / (1 + total_b) - 1
        print(f"超额收益  计算 {excess_geo * 100:+.4f}%  聚宽 {p['超额收益']:+.2f}%")
        print(f"日均超额  计算 {excess_geo / n * 100:+.4f}%  聚宽 {p['日均超额收益']:+.2f}%")
        mdd, peak, trough = max_dd(1 + df["策略收益"] / 100)
        print(f"最大回撤  计算 {abs(mdd) * 100:.2f}%  聚宽 {p['最大回撤']:.2f}%")
        print(f"回撤区间  计算 {peak.date()} ~ {trough.date()}  "
              f"聚宽 {p['最大回撤区间'][0]} ~ {p['最大回撤区间'][1]}")
        # 超额曲线(几何)回撤
        ratio = (1 + df["策略收益"] / 100) / (1 + df["基准收益"] / 100)
        mdd_x, _, _ = max_dd(ratio)
        print(f"超额回撤  计算 {abs(mdd_x) * 100:.2f}%  聚宽 {p['超额收益最大回撤']:.2f}%")
        # 日胜率
        print(f"日胜率    计算 {(rp > 0).mean():.4f}  聚宽 {p['日胜率']:.3f}")
        # 信息比率候选
        te_arith = (rp - rb).std(ddof=1) * np.sqrt(ANN)
        print(f"信息比率  算术超额TE: {(ann_s - ann_b) / te_arith:+.4f}  "
              f"聚宽 {p['信息比率']:+.3f}")
        # 索提诺候选
        dn_std = rp[rp < 0].std(ddof=1) * np.sqrt(ANN)
        dn_lpm2 = np.sqrt(np.mean(np.minimum(rp, 0) ** 2)) * np.sqrt(ANN)
        print(f"索提诺    负收益std: {(ann_s - RF) / dn_std:+.4f}  "
              f"LPM2: {(ann_s - RF) / dn_lpm2:+.4f}  聚宽 {p['索提诺比率']:+.3f}")
        # 超额夏普候选
        ex_arith = (rp - rb)
        ex_geo_d = (1 + rp) / (1 + rb) - 1
        ex_ann = (1 + excess_geo) ** (ANN / n) - 1
        print(f"超额夏普  超额年化/算术TE: {ex_ann / ex_arith.std(ddof=1) / np.sqrt(ANN) * np.sqrt(ANN):+.4f}"
              f"  (ex_ann={ex_ann:+.4f}, TE算术年化={ex_arith.std(ddof=1) * np.sqrt(ANN):.4f})"
              f"  聚宽 {p['超额收益夏普比率']:+.3f}")
        print(f"          (ex_ann-Rf)/算术TE: {(ex_ann - RF) / (ex_arith.std(ddof=1) * np.sqrt(ANN)):+.4f}")
        print(f"          ex_ann/几何TE: {ex_ann / (ex_geo_d.std(ddof=1) * np.sqrt(ANN)):+.4f}")
        print(f"          (ex_ann-Rf)/几何TE: {(ex_ann - RF) / (ex_geo_d.std(ddof=1) * np.sqrt(ANN)):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
