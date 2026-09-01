# -*- coding: utf-8 -*-
"""compute_jq_panel 口径终验: 灌入聚宽官方 nav/bench 曲线, 应复现聚宽面板。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from core.metrics import compute_jq_panel  # noqa: E402

SAMPLES = [
    (r"C:\Users\zhoubw\Downloads\result_1 (2).csv", "post47523"),
    (r"C:\Users\zhoubw\Downloads\result_1 (1).csv", "国九"),
]

for path, name in SAMPLES:
    df = pd.read_csv(path, encoding="gbk")
    df["date"] = pd.to_datetime(df["时间"])
    df = df.set_index("date")
    nav = 1 + df["策略收益"] / 100        # 首点 = 1+首日收益(基点=初始资金)
    bench = 1 + df["基准收益"] / 100      # 基点 = 回测前收盘(首点=1+首日涨跌)
    panel = compute_jq_panel(nav=nav, bench=bench, fills=None)
    print(f"\n===== {name}: 计算 vs 聚宽 =====")
    jq = {"策略收益": None, "策略年化收益": None, "基准收益": None,
          "超额收益": None, "阿尔法": None, "贝塔": None, "夏普比率": None,
          "索提诺比率": None, "最大回撤": None, "策略波动率": None,
          "基准波动率": None, "日胜率": None, "信息比率": None,
          "日均超额收益": None, "超额收益最大回撤": None,
          "超额收益夏普比率": None}
    for k in panel:
        v = panel[k]
        if isinstance(v, float):
            print(f"  {k}: {v:+.4f}" if abs(v) < 10 else f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
