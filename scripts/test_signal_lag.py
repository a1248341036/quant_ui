# -*- coding: utf-8 -*-
"""前视审计 · 信号滞后实验（2026-10-20）。

**要回答的问题**：`full_period_sharpe.py` 的调仓是「d0 收盘算分 → d0 收盘价买入」。
分数里含 d0 收盘信息，而以 d0 收盘价成交在实盘上只有尾盘竞价勉强可行——若把信号
**滞后一天**（用 T−1 收盘的分数在 T 建仓）业绩大幅衰减，则原结果存在执行不可行的
乐观偏差；若衰减不大，则前视影响可忽略。

**方法**：同一份 walk-forward 分数缓存（full_period_scores.parquet），对
score_wide 做 `shift(lag)` 后按同一组合规则回测，输出 lag=0/1/2 的对照。
**口径**：与 full_period_sharpe 完全一致（top50 等权 / 20 日调仓 / 10bp 单边 /
adj_close 收益 / 基准=同过滤全市场等权）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRADING_DAYS = 244.0


def _metrics(nav: pd.Series, ppy: float) -> dict:
    if len(nav) < 4:
        return {}
    ret = nav.pct_change().dropna()
    n_per = max(1, len(ret))
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    ann = float((1 + total) ** (ppy / n_per) - 1) if total > -1 else float("nan")
    vol = float(ret.std() * np.sqrt(ppy))
    sharpe = float(ann / vol) if vol > 0 else float("nan")
    cummax = nav.cummax()
    mdd = float(((nav - cummax) / cummax).min())
    return {"total_return": total, "annual_return": ann, "annual_vol": vol,
            "sharpe": sharpe, "max_drawdown": mdd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-cache", default=str(
        ROOT / "artifacts" / "alphaagent" / "harness_runs" / "2026-10-20_increment" / "full_period_scores.parquet"))
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--rebalance", type=int, default=20)
    ap.add_argument("--lags", default="0,1,2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    sc_path = Path(args.score_cache)
    if not sc_path.is_file():
        print(f"[lag] 分数缓存不存在: {sc_path}")
        return 1
    sdf = pd.read_parquet(sc_path)
    dcol = "date" if "date" in sdf.columns else "datetime"
    sdf[dcol] = pd.to_datetime(sdf[dcol])
    score = sdf.set_index([dcol, "instrument"])["score"].sort_index()
    score_wide = score.unstack("instrument")
    print(f"[lag] 分数 {score_wide.shape}（{score_wide.index.min().date()}~{score_wide.index.max().date()}）",
          flush=True)

    t0 = time.perf_counter()
    panel = load_panel_from_cne(start="2019-12-01", end=args.end,
                                include_fundamentals=False,
                                keep_columns=("adj_close", "is_trade", "not_st"))
    print(f"[lag] panel {panel.shape} load={time.perf_counter()-t0:.0f}s", flush=True)
    px = panel["adj_close"].astype("float64")
    trade_ok = pd.Series(True, index=panel.index)
    for col in ("is_trade", "not_st"):
        if col in panel.columns:
            trade_ok &= panel[col].fillna(False).astype(bool)
    px_wide = px.unstack("instrument")
    ok_wide = trade_ok.unstack("instrument")

    day_list = np.array(sorted(score_wide.index.unique()))
    reb_days = day_list[:: args.rebalance]
    ppy = TRADING_DAYS / args.rebalance
    cost = args.cost_bps / 1e4

    results = {}
    for lag in [int(x) for x in args.lags.split(",")]:
        sw = score_wide.shift(lag)  # 按行 shift：loc[d0] 给的是 d0 前第 lag 个交易日的分数
        prev_w = None
        nav_m, nav_b, nav_dates = [1.0], [1.0], [reb_days[0]]
        turns = []
        for k in range(len(reb_days) - 1):
            d0, d1 = reb_days[k], reb_days[k + 1]
            if d0 not in px_wide.index or d1 not in px_wide.index or d0 not in sw.index:
                continue
            s = sw.loc[d0].dropna()
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
            bret = (px_wide.loc[d1].reindex(bsel) / p0.reindex(bsel) - 1).fillna(0.0)
            nav_m.append(nav_m[-1] * (1 + net))
            nav_b.append(nav_b[-1] * (1 + float(bret.mean())))
            nav_dates.append(d1)
            turns.append(turn)
        nav_m = pd.Series(nav_m, index=pd.to_datetime(nav_dates))
        nav_b = pd.Series(nav_b, index=pd.to_datetime(nav_dates))
        mm, mb = _metrics(nav_m, ppy), _metrics(nav_b, ppy)
        results[f"lag{lag}"] = {**mm, "bench_annual": mb["annual_return"],
                                "excess_annual": mm["annual_return"] - mb["annual_return"],
                                "excess_sharpe": mm["sharpe"] - mb["sharpe"],
                                "avg_turnover": float(np.mean(turns)) if turns else float("nan")}
        print(f"[lag] lag={lag}: 年化 {mm['annual_return']:+.2%}  夏普 {mm['sharpe']:.2f}  "
              f"回撤 {mm['max_drawdown']:.2%}  累计 {mm['total_return']:+.2%}  "
              f"超额年化 {mm['annual_return']-mb['annual_return']:+.2%}", flush=True)
        # 逐年分段（组合 vs 基准）
        for lo, hi, nm in (("2021-01-01", "2021-12-31", "2021"),
                           ("2022-01-01", "2022-12-31", "2022"),
                           ("2023-01-01", "2023-12-31", "2023"),
                           ("2024-01-01", "2024-12-31", "2024"),
                           ("2025-01-01", "2026-12-31", "2025-26盲测")):
            sub = nav_m[(nav_m.index >= lo) & (nav_m.index <= hi)]
            subb = nav_b[(nav_b.index >= lo) & (nav_b.index <= hi)]
            if len(sub) < 3:
                continue
            ms, mbs = _metrics(sub, ppy), _metrics(subb, ppy)
            if not ms or not mbs:
                continue
            print(f"[lag]   {nm:12s} 组合年化 {ms['annual_return']:+8.2%} 夏普 {ms['sharpe']:5.2f} "
                  f"回撤 {ms['max_drawdown']:+7.2%} | 基准 {mbs['annual_return']:+7.2%} | "
                  f"超额 {ms['annual_return']-mbs['annual_return']:+.2%}", flush=True)

    print("\n[lag] ===== 前视敏感度结论 =====")
    a0 = results.get("lag0", {}).get("excess_annual")
    a1 = results.get("lag1", {}).get("excess_annual")
    if a0 is not None and a1 is not None:
        print(f"[lag] 超额年化: lag0={a0:+.2%} → lag1={a1:+.2%}（衰减 {a0-a1:+.2%}）")
        verdict = ("⚠️ 原结果严重依赖当日收盘成交（乐观偏差大）" if (a0 - a1) > 0.015
                   else "✅ 对信号滞后不敏感，执行时点前视影响小")
        print(f"[lag] 判定: {verdict}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
        print(f"[lag] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
