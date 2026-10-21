# -*- coding: utf-8 -*-
"""盲测段回撤归因：最大回撤窗口定位 + 组合/基准日收益分解 + 持仓风格特征（2026-10-20）。

**要回答的问题**：purged 周频口径盲测段（2025-01~2026-09）回撤 −32.2%——
发生在什么时候？是市场（基准同跌）还是组合特质？回撤时的持仓押了什么风格？

**方法**：
1. 重放周频 lag=1 回测（与 test_signal_lag 完全同口径）得到 nav；
2. 定位最大回撤窗口（peak→trough）与 Top-3 回撤窗口；
3. 窗口内逐日分解：组合日收益 = 基准日收益 + 日超额 → 超额部分累加即"特质贡献"；
4. 窗口内持仓的风格画像：金额（am20 流动性分层）、20 日已实现波动、动量，
   与全市场中位数对比（只读，不回流）。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-cache", default=str(
        ROOT / "artifacts" / "alphaagent" / "harness_runs" / "2026-10-20_increment" / "full_period_scores_purged.parquet"))
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    sdf = pd.read_parquet(args.score_cache)
    dcol = "date" if "date" in sdf.columns else "datetime"
    sdf[dcol] = pd.to_datetime(sdf[dcol])
    score_wide = sdf.set_index([dcol, "instrument"])["score"].sort_index().unstack("instrument").shift(1)  # lag=1

    panel = load_panel_from_cne(
        start="2024-11-01", end=args.end, include_fundamentals=False,
        keep_columns=("adj_close", "is_trade", "not_st", "amount", "volume", "close"))
    px_wide = panel["adj_close"].astype("float64").unstack("instrument")
    ok_wide = (panel["is_trade"].fillna(False).astype(bool)
               & panel["not_st"].fillna(False).astype(bool)).unstack("instrument")
    amt_wide = panel["amount"].astype("float64").unstack("instrument")
    close_wide = panel["close"].astype("float64").unstack("instrument")
    vol20 = (close_wide.pct_change().rolling(20).std() * np.sqrt(244))
    mom20 = close_wide.pct_change(20)

    common_days = score_wide.index.unique().intersection(px_wide.index)
    day_list = np.array(sorted(common_days))
    reb_days = day_list[:: args.rebalance]
    cost = args.cost_bps / 1e4
    prev_w, nav = None, [1.0]
    nav_dates, hold_style = [reb_days[0]], []
    for k in range(len(reb_days) - 1):
        d0, d1 = reb_days[k], reb_days[k + 1]
        if d0 not in px_wide.index or d1 not in px_wide.index or d0 not in score_wide.index:
            continue
        s = score_wide.loc[d0].dropna()
        ok = ok_wide.loc[d0].reindex(s.index).fillna(False).astype(bool)
        p0 = px_wide.loc[d0].reindex(s.index)
        valid = ok & p0.notna() & np.isfinite(s)
        s, p0 = s[valid], p0[valid]
        if len(s) < args.top_n:
            continue
        sel = s.nlargest(args.top_n).index
        w = pd.Series(1.0 / len(sel), index=sel)
        ret = (px_wide.loc[d1].reindex(sel) / p0.reindex(sel) - 1).fillna(0.0)
        gross = float((w * ret).sum())
        turn = 0.5 * float((w.sub(prev_w.reindex(w.index).fillna(0.0), fill_value=0.0)).abs().sum()) \
            if prev_w is not None else 0.5
        net = gross - turn * 2 * cost
        prev_w = w
        b_ok = ok_wide.loc[d0].reindex(p0.index).fillna(False).astype(bool)
        bsel = p0.index[b_ok & p0.notna()]
        bret = float(((px_wide.loc[d1].reindex(bsel) / p0.reindex(bsel) - 1).fillna(0.0)).mean())
        nav.append(nav[-1] * (1 + net))
        nav_dates.append(d1)
        hold_style.append({
            "date": str(pd.Timestamp(d1).date()),
            "med_amt": float(amt_wide.loc[d0].reindex(sel).median()),
            "med_vol20": float(vol20.loc[d0].reindex(sel).median()),
            "med_mom20": float(mom20.loc[d0].reindex(sel).median()),
            "mkt_med_amt": float(amt_wide.loc[d0].reindex(bsel).median()),
            "mkt_med_vol20": float(vol20.loc[d0].reindex(bsel).median()),
            "mkt_med_mom20": float(mom20.loc[d0].reindex(bsel).median()),
        })

    nav = pd.Series(nav, index=pd.to_datetime(nav_dates))
    ret_p = nav.pct_change().fillna(0.0)
    # 基准日收益（等权，同窗）
    bench = pd.Series(index=nav.index, dtype="float64")
    for d in nav.index[1:]:
        bench[d] = (px_wide.loc[d] / px_wide.loc[_prev_reb(nav.index, d)] - 1).mean()
    bench = bench.fillna(0.0)

    dd = nav / nav.cummax() - 1.0
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    print(f"[dd] 盲测段回撤：peak {peak.date()} → trough {trough.date()}  "
          f"深度 {dd.min():+.2%}（{len(nav.loc[peak:trough])} 个调仓期）", flush=True)

    # 回撤窗口内组合 vs 基准分解（超额 = 组合 − 基准，逐期累加）
    w_p = (1 + ret_p.loc[peak:trough]).prod() - 1
    w_b = (1 + bench.loc[peak:trough]).prod() - 1
    print(f"[dd] 窗口内：组合 {w_p:+.2%} vs 基准 {w_b:+.2%} → 特质(超额) {w_p - w_b:+.2%}", flush=True)
    print(f"[dd]   → 市场贡献 {w_b:+.2%}，alpha 贡献 {w_p - w_b:+.2%}", flush=True)

    # Top-3 回撤窗口
    print("\n[dd] Top-3 回撤窗口：", flush=True)
    dd_copy = dd.copy()
    for i in range(3):
        if dd_copy.empty:
            break
        t = dd_copy.idxmin()
        p = nav.loc[:t].idxmax()
        depth = dd_copy.min()
        print(f"  #{i+1}: {p.date()} → {t.date()}  深度 {depth:+.2%}  "
              f"窗口基准 {((1+bench.loc[p:t]).prod()-1):+.2%}", flush=True)
        dd_copy = dd_copy[dd_copy.index < p]

    style = pd.DataFrame(hold_style).set_index("date")
    trough_s = str(pd.Timestamp(trough).date())
    near = [d for d in style.index if d <= trough_s][-5:]
    print(f"\n[dd] 回撤谷底附近持仓风格（vs 全市场中位）:", flush=True)
    for d in near:
        r = style.loc[d]
        print(f"  {d}: 持仓 amt {r['med_amt']/1e6:.1f}M (市场 {r['mkt_med_amt']/1e6:.1f}M) | "
              f"vol20 {r['med_vol20']:.1%} (市场 {r['mkt_med_vol20']:.1%}) | "
              f"mom20 {r['med_mom20']:+.1%} (市场 {r['mkt_med_mom20']:+.1%})", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "max_dd": {"peak": str(peak.date()), "trough": str(trough.date()),
                       "depth": float(dd.min()),
                       "window_ret_p": float(w_p), "window_ret_b": float(w_b)},
            "style_rows": hold_style,
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[dd] 明细写入 {args.out}")
    return 0


def _prev_reb(index, d):
    loc = index.get_loc(d)
    return index[loc - 1]


if __name__ == "__main__":
    raise SystemExit(main())
