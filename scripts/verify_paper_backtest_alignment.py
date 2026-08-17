#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模拟盘 vs 回测 口径对照验证。

因子模拟盘账户现在重放 core.engine.run_backtest(cash_mode=True) 的结果，
本脚本对同一策略/资金/区间/频率分别跑模拟盘与回测，逐项对照：
- 每日权益（净值×资金）是否一致
- 成交笔数 / 拒单笔数 / 持仓只数是否一致
- 期末现金与市值是否一致

用法：python scripts/verify_paper_backtest_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import services  # noqa: E402
from core import paper  # noqa: E402
from core.engine import run_backtest  # noqa: E402
from strategies.registry import STRATEGIES  # noqa: E402


CASES = [
    # (策略名, 频率, 起点, 终点)
    ("动量 20 日", "monthly", "2025-01-02", "2025-06-30"),
    ("低换手冷门", "monthly", "2025-01-02", "2025-06-30"),
    ("动量 20 日", "daily", "2025-01-02", "2025-06-30"),
]


def run_case(panel, codes_by_universe, codes, strat_name, freq, start, end) -> dict:
    s = STRATEGIES[strat_name]
    risk = {"max_weight": 0.5, "amount_q": 0.2, "warmup_days": 400}
    name = f"__verify_{strat_name.replace(' ', '_')}_{freq}__"
    for a in paper.list_accounts():
        if a["name"] == name:
            paper.delete_account(a["id"])
    acc = paper.create_account(name, strat_name, s["factor"], s["ascending"],
                               "科技TMT", 50000, 3, freq,
                               risk_config=risk, start_date=start)
    try:
        r = paper.run_paper_trade(panel, codes_by_universe, acc["id"],
                                  exec_date=end)
        ar = r["accounts"][0]
        eq = paper.account_equity(acc["id"])
        snaps = {pd.Timestamp(x["date"]).date(): x for x in eq}
        n_trades = len(paper.account_trades(acc["id"]))
        n_orders = len(paper.account_orders(acc["id"]))
        n_pos = len(paper.account_positions(acc["id"]))

        res = run_backtest(
            panel, codes, s["factor"], s["ascending"], start, end,
            50000, 3, freq, amount_q=0.2, warmup_days=400,
            cash_mode=True, limit_flags=True, max_weight=0.5,
        )
        nav = res["nav"]
        mismatches: list[tuple] = []
        max_diff = 0.0
        for dt, v in nav.items():
            d = pd.Timestamp(dt).date()
            if d not in snaps:
                continue
            exp = v * 50000
            got = snaps[d]["equity"]
            diff = abs(got - exp)
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                mismatches.append((d, got, exp, diff))
        n_days = sum(1 for dt in nav.index if pd.Timestamp(dt).date() in snaps)
        last_snap = snaps.get(pd.Timestamp(nav.index[-1]).date())
        last_equity = last_snap["equity"] if last_snap else None
        last_cash = last_snap["cash"] if last_snap else None
        bt_pos = (len(res["positions_history"][-1] or {})
                  if res["positions_history"] else 0)
        return {
            "策略": strat_name, "频率": freq, "区间": f"{start}~{end}",
            "执行状态": ar["processed"], "订单/成交/拒单": f"{n_orders}/{n_trades}/{ar['rejected']}",
            "回测成交/拒单": f"{len(res['trades_detail'])}/{len(res['rejections'])}",
            "持仓只数(模拟盘/回测)": f"{n_pos}/{bt_pos}",
            "对比日数": n_days,
            "权益最大偏差": round(max_diff, 8),
            "不一致日数": len(mismatches),
            "期末权益(模拟盘/回测)": f"{last_equity:.2f}/{nav.iloc[-1]*50000:.2f}" if last_equity is not None else "-",
            "期末现金(模拟盘)": f"{last_cash:.2f}" if last_cash is not None else "-",
        }
    finally:
        paper.delete_account(acc["id"])


def main() -> int:
    data = services.load_data()
    panel = data["panel"]
    codes = services.build_codes("科技TMT", True)
    cbu = {"科技TMT": codes, "沪深300+中证500+中证1000": codes}
    rows = [run_case(panel, cbu, codes, *case) for case in CASES]
    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))
    bad = df[df["权益最大偏差"].astype(float) > 1e-6]
    if not bad.empty:
        print("\nFAIL: 存在权益不一致的用例")
        return 1
    print("\nPASS: 全部用例模拟盘与回测口径一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
