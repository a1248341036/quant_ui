# -*- coding: utf-8 -*-
"""日胜率/索提诺 口径第二轮校验。"""
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
    rb = daily_from_cum(df["基准收益"])
    print(f"\n===== {name} (n={len(rp)}) =====")
    n = len(rp)
    tgt = p["日胜率"]
    print("目标胜率对应天数:", tgt * n)
    print("rp>0:", int((rp > 0).sum()),
          " rp>rb:", int((rp > rb).sum()),
          " 盈利>0:", int((df["当日盈利"] > 0).sum()),
          " 亏损<0:", int((df["当日亏损"] < 0).sum()),
          " 双零:", int(((df["当日盈利"] == 0) & (df["当日亏损"] == 0)).sum()),
          " 净盈亏>0:", int(((df["当日盈利"] + df["当日亏损"]) > 0).sum()))
    # 条件口径: 分母排除双零日
    flat = (df["当日盈利"] == 0) & (df["当日亏损"] == 0)
    for w_name, w in (("盈利>0", (df["当日盈利"] > 0)),
                      ("净盈亏>0", (df["当日盈利"] + df["当日亏损"]) > 0),
                      ("rp>0", rp > 0)):
        print(f"  {w_name}/非双零: {w[~flat].mean():.4f}")
    # 索提诺: ddof=0 与 MAR=Rf 口径
    ann = p["年化"]
    rf_d = RF / ANN
    for ddof in (0, 1):
        dn = rp[rp < rf_d]
        s = dn.std(ddof=ddof) * np.sqrt(ANN) if len(dn) > 1 else np.nan
        lpm = np.sqrt(np.mean(np.minimum(rp - rf_d, 0) ** 2)) * np.sqrt(ANN)
        print(f"  ddof={ddof}: (ann-Rf)/neg_std({rf_d:.6f}以下): "
              f"{(ann - RF) / s:+.4f}  (ann-Rf)/lpm2: {(ann - RF) / lpm:+.4f}")
