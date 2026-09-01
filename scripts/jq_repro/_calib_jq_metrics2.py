# -*- coding: utf-8 -*-
"""日胜率/索提诺 剩余口径变体校验。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RF = 0.04
ANN = 250


def daily_from_cum(cum_pct: pd.Series) -> pd.Series:
    nav = 1 + cum_pct / 100.0
    return nav.pct_change().fillna(nav.iloc[0] - 1)


SAMPLES = [
    (r"C:\Users\zhoubw\Downloads\result_1 (2).csv", "post47523",
     {"日胜率": 0.478, "索提诺": -0.356, "年化": -0.0322}),
    (r"C:\Users\zhoubw\Downloads\result_1 (1).csv", "国九",
     {"日胜率": 0.475, "索提诺": 0.157, "年化": 0.0729}),
]

for path, name, p in SAMPLES:
    df = pd.read_csv(path, encoding="gbk")
    df["date"] = pd.to_datetime(df["时间"])
    df = df.set_index("date")
    rp = daily_from_cum(df["策略收益"])
    nav = 1 + df["策略收益"] / 100
    print(f"\n===== {name} =====")
    # 日胜率变体
    print("日胜率:  rp>0:", f"{(rp > 0).mean():.4f}",
          " rp>=0:", f"{(rp >= 0).mean():.4f}",
          " diff>0(nav):", f"{(nav.diff() > 0).mean():.4f}",
          " 盈利列>0:", f"{(df['当日盈利'] > 0).mean():.4f}",
          " (盈利-亏损)>0:", f"{((df['当日盈利'] + df['当日亏损']) > 0).mean():.4f}",
          " 目标:", p["日胜率"])
    # 索提诺变体
    ann = p["年化"]
    dn_negstd0 = rp[rp < 0].std(ddof=0) * np.sqrt(ANN)
    dn_lpm2_0 = np.sqrt(np.mean(np.minimum(rp, 0) ** 2)) * np.sqrt(ANN)
    rf_d = RF / ANN
    dn_vs_rf = np.sqrt(np.mean(np.minimum(rp - rf_d, 0) ** 2)) * np.sqrt(ANN)
    print(f"索提诺:  ann/dn_negstd0: {ann / dn_negstd0:+.4f}"
          f"  ann/dn_lpm2_0: {ann / dn_lpm2_0:+.4f}"
          f"  (ann-Rf)/dn_negstd0: {(ann - RF) / dn_negstd0:+.4f}"
          f"  (ann-Rf)/dn_vs_rf: {(ann - RF) / dn_vs_rf:+.4f}"
          f"  目标: {p['索提诺']:+.3f}")
