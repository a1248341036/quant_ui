#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用新平台引擎复现 quant_3stocks/scripts/backtest_5w.py，对照旧结果。

旧脚本口径：
- 股票池: tech(申万 电子/计算机/通信/传媒) ∩ CS800 panel，剔除 300/301/688/689
- 资金 50000, TopN 每只预算=资金/N, 一手 100 股可负担过滤
- 月度调仓, BUY_COST=0.0008, SELL_COST=0.0013, amount_q=0.2
- 因子在全量 2020-01-01~end 上计算后再截取窗口（即含预热数据）

本脚本:
- mode=platform : 新平台默认行为（窗口内直接算因子，无预热）
- mode=legacy   : 新引擎 + 预热数据（start=2020-01-01 算因子，再截取净值段），与旧脚本同口径
- data=new      : 新 data/ 面板
- data=old      : /tmp 旧面板（隔离数据差异）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine import run_backtest
from core.store import LEGACY_DATA_DIR
from strategies.registry import STRATEGIES

CAPITAL = 50_000
TOP_N = 5
AMOUNT_Q = 0.2

# 旧策略名 -> 新平台策略名（新平台无 cold_ind/cold_div）
STRAT_MAP = {
    "cold": "低成交冷门",
    "leader": "高成交领涨",
    "mom20": "动量 20 日",
    "mom60": "动量 60 日",
    "rev20": "反转 20 日",
    "lowvol": "低波动",
    "composite": "复合因子",
}
WINDOWS = {
    "长期(2020-2026)": ("2020-01-02", "2026-08-13"),
    "近半年": ("2026-02-02", "2026-08-13"),
}

# 旧结果参考 (window, strategy) -> (total%, annual%, sharpe, mdd%)
OLD_REF = {}


def load_old_ref() -> dict:
    ref = {}
    path = LEGACY_DATA_DIR / "backtest/quant_3stocks/outputs/backtest_5w.csv"
    if not path.exists():
        return ref
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        if r["top_n"] != TOP_N:
            continue
        ref[(r["window"], r["strategy"])] = {
            "total": r["total"], "annual": r["annual"],
            "sharpe": r["sharpe"], "mdd": r["mdd"],
        }
    return ref


def old_metrics(nav: pd.Series) -> dict:
    """旧脚本 calc_metrics 口径（算术夏普）。"""
    nav = nav.dropna()
    rets = nav.pct_change().dropna()
    n = len(rets)
    years = n / 244.0
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = nav.iloc[-1] / nav.iloc[0] ** (1 / years) - 1 if years > 0 else np.nan
    # 注意旧脚本实际: (nav_end)**(1/years)-1, 当 nav0=1 时与上式相同
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    vol = rets.std(ddof=1) * np.sqrt(244)
    sharpe = (rets.mean() * 244) / (rets.std(ddof=1) * np.sqrt(244)) if len(rets) else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    return {"total": total, "annual": ann, "sharpe": sharpe, "mdd": mdd}


def run_one(panel: pd.DataFrame, codes: list[str], strat_name: str,
            start: str, end: str, mode: str) -> dict:
    strat = STRATEGIES[strat_name]
    warmup = 9999 if mode == "legacy" else None  # legacy: 全量数据预热因子
    res = run_backtest(
        panel=panel, codes=codes, factor=strat["factor"], ascending=strat["ascending"],
        start=start, end=end, capital=CAPITAL, top_n=TOP_N,
        freq="monthly", buy_cost=0.0008, sell_cost=0.0013,
        amount_q=AMOUNT_Q, affordable=True, warmup_days=warmup,
    )
    return old_metrics(res["nav"])


def main() -> None:
    OLD_REF.update(load_old_ref())

    # 数据准备
    new_panel = pd.read_parquet(ROOT / "data/panel.parquet")
    new_panel["code"] = new_panel["code"].astype(str).str.zfill(6)
    new_panel["date"] = pd.to_datetime(new_panel["date"])
    old_panel = pd.read_parquet(
        LEGACY_DATA_DIR / "panel/turn20/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet"
    )
    old_panel["code"] = old_panel["code"].astype(str).str.zfill(6)
    old_panel["date"] = pd.to_datetime(old_panel["date"])

    def codes_from(panel, tech_path):
        tech = pd.read_csv(tech_path, dtype={"code": str})
        tech["code"] = tech["code"].astype(str).str.zfill(6)
        return sorted({c for c in set(tech["code"]) & set(panel["code"].unique())
                       if not c.startswith(("300", "301", "688", "689"))})

    codes_new = codes_from(new_panel, ROOT / "data/tech.csv")
    codes_old = codes_from(old_panel, LEGACY_DATA_DIR / "panel/tech_universe_sw.csv")
    print(f"codes: new={len(codes_new)} old={len(codes_old)}")

    rows = []
    for mode in ("platform", "legacy"):
        for wname, (ws, we) in WINDOWS.items():
            for old_name, new_name in STRAT_MAP.items():
                for data_label, panel, codes in (
                        ("new", new_panel, codes_new), ("old", old_panel, codes_old)):
                    m = run_one(panel, codes, new_name, ws, we, mode)
                    ref = OLD_REF.get((wname, old_name), {})
                    rows.append({
                        "mode": mode, "data": data_label, "window": wname,
                        "strategy": old_name,
                        "new_total": m["total"] * 100, "new_annual": m["annual"] * 100,
                        "new_sharpe": m["sharpe"], "new_mdd": m["mdd"] * 100,
                        "old_total": ref.get("total"), "old_annual": ref.get("annual"),
                        "old_sharpe": ref.get("sharpe"), "old_mdd": ref.get("mdd"),
                    })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    for mode in ("platform", "legacy"):
        for wname in WINDOWS:
            print(f"\n===== mode={mode}  {wname}  (新平台引擎 Top{TOP_N} 5万 amount_q={AMOUNT_Q}) =====")
            sub = df[(df.mode == mode) & (df.window == wname)].sort_values("strategy")
            for data_label in ("new", "old"):
                s = sub[sub.data == data_label]
                if s.empty:
                    continue
                print(f"--- 数据源: {data_label} ---")
                out = []
                for _, r in s.iterrows():
                    dt = "" if pd.isna(r.old_total) else f"{r.old_total:.2f}%"
                    da = "" if pd.isna(r.old_annual) else f"{r.old_annual:.2f}%"
                    ds = "" if pd.isna(r.old_sharpe) else f"{r.old_sharpe:.2f}"
                    dm = "" if pd.isna(r.old_mdd) else f"{r.old_mdd:.2f}%"
                    out.append({
                        "策略": r.strategy,
                        "新total%": f"{r.new_total:.2f}", "旧total%": dt,
                        "新annual%": f"{r.new_annual:.2f}", "旧annual%": da,
                        "新sharpe": f"{r.new_sharpe:.2f}", "旧sharpe": ds,
                        "新mdd%": f"{r.new_mdd:.2f}", "旧mdd%": dm,
                    })
                print(pd.DataFrame(out).to_string(index=False))
    out_path = ROOT / "results" / "compare_old.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
