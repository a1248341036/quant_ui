# -*- coding: utf-8 -*-
"""分钟撮合 vs 开盘撮合 的成交对照(同一策略同一窗口):
- run A: 正常(分钟数据 -> 下单时点分钟价成交)
- run B: 关闭分钟预取(回落执行日开盘成交)
输出: 两边逐笔成交 + 价格对照, 并抽查分钟价换算是否正确。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

START, END, CAPITAL = "2025-01-02", "2026-08-28", 100_000.0


def load_code() -> str:
    src = (ROOT / "scripts" / "jq_repro" / "_test_guojiu.py").read_text(
        encoding="utf-8")
    return src.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]


def trades_df(res: dict) -> pd.DataFrame:
    t = pd.DataFrame(res["trades"])
    if t.empty:
        return t
    t["date"] = t["date"].astype(str)
    return t[["date", "bought", "sold", "num_hold", "turnover"]]


def main() -> int:
    import jq_data
    from core.event_engine.jq import runtime as jq_runtime
    from core.event_engine.jq.entry import run_jq_backtest

    code = load_code()

    res_a = run_jq_backtest(code, start=START, end=END, capital=CAPITAL)
    ta = trades_df(res_a)
    print(f"\n[run A 分钟撮合] 总收益 {res_a['metrics']['总收益']:.2%}, "
          f"事件笔数 {len(ta)}")

    # run B: 关闭分钟预取 -> 全部回落开盘成交
    orig = jq_runtime.JQRuntime.prefetch_minutes
    jq_runtime.JQRuntime.prefetch_minutes = lambda self, minutes: None
    try:
        res_b = run_jq_backtest(code, start=START, end=END, capital=CAPITAL)
    finally:
        jq_runtime.JQRuntime.prefetch_minutes = orig
    tb = trades_df(res_b)
    print(f"[run B 开盘撮合] 总收益 {res_b['metrics']['总收益']:.2%}, "
          f"事件笔数 {len(tb)}")

    merged = ta.merge(tb, on="date", how="outer", suffixes=("_min", "_open"))
    merged = merged.sort_values("date")
    first_div = None
    for _, r in merged.iterrows():
        same = (str(r.get("bought_min")) == str(r.get("bought_open"))
                and str(r.get("sold_min")) == str(r.get("sold_open")))
        if not same and first_div is None:
            first_div = r["date"]
    print(f"首次成交分化日期: {first_div}")

    print("\n-- 首次分化前后各 8 次成交对照 --")
    if first_div is not None:
        idx = merged.index[merged["date"] == first_div][0]
        lo, hi = max(0, idx - 8), min(len(merged), idx + 9)
        for _, r in merged.iloc[lo:hi].iterrows():
            print(f"  {r['date']}  A买[{r.get('bought_min')}]卖[{r.get('sold_min')}]"
                  f"  |  B买[{r.get('bought_open')}]卖[{r.get('sold_open')}]")

    # 抽查 run A 前 6 笔的成交基准价(明细里的 fill price)
    from core.event_engine.jq.entry import _LAST_RT
    rt = _LAST_RT
    print("\n-- run A 成交价抽查(前 6 笔) --")
    det = res_a["trades_detail"]
    for f in det[:6]:
        dt = pd.Timestamp(f["date"])
        raw = rt._minute_px_raw(f["code"], dt.replace(hour=10, minute=0))
        print(f"  {dt.date()} {f['code']} {f['side']} price={f['price']:.3f}"
              f"  (raw10:00={raw})")

    ta.to_csv(ROOT / "artifacts" / "guojiu_trades_minute.csv",
              index=False, encoding="utf-8-sig")
    tb.to_csv(ROOT / "artifacts" / "guojiu_trades_open.csv",
              index=False, encoding="utf-8-sig")
    print("\n成交清单已存 artifacts/guojiu_trades_minute.csv / _open.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
