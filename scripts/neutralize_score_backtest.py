# -*- coding: utf-8 -*-
"""score 风格中性化实验：残差分数 vs 原始分数的组合回测对比（2026-10-20）。

**背景**（归因结论，见 docs/review/2026-10-20_blind_exposure_purged.md）：
原始 score 的 IC 有 48.4% 来自风格暴露（反动量 t=-33.5 最显著），风格反转期
（2026 上半年）回撤 −31.7%。中性化后特质 IC 0.0185 / ICIR 0.452 全程为正。

**方法**：**每日截面中性化**——每个交易日把 score 对当期风格暴露（mom20/vol20/
am20/turn20 的截面 rank）+ 申万行业哑变量做截面回归，残差即中性化分数。
只用 T 日已知信息、系数逐日截面估计，无时间泄漏；不涉及任何未来数据。

**对比**：原始 vs 中性化 × 简化周频（lag=1, top50, 10bp）+ 引擎月频（真实执行）。
输出逐年分段，判断「去掉风格押注后引擎口径能否同时站上正超额与回撤<30%」。
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
    return {"total_return": total, "annual_return": ann, "annual_vol": vol,
            "sharpe": float(ann / vol) if vol > 0 else float("nan"),
            "max_drawdown": float(((nav - nav.cummax()) / nav.cummax()).min())}


def _run_backtest(score_wide: pd.DataFrame, px_wide: pd.DataFrame, ok_wide: pd.DataFrame,
                  rebalance: int, top_n: int, cost_bps: float) -> pd.Series:
    day_list = np.array(sorted(score_wide.index.unique()))
    reb_days = day_list[::rebalance]
    cost = cost_bps / 1e4
    prev_w, nav, dates = None, [1.0], [reb_days[0]]
    for k in range(len(reb_days) - 1):
        d0, d1 = reb_days[k], reb_days[k + 1]
        if d0 not in px_wide.index or d1 not in px_wide.index:
            continue
        s = score_wide.loc[d0].dropna()
        ok = ok_wide.loc[d0].reindex(s.index).fillna(False).astype(bool)
        p0 = px_wide.loc[d0].reindex(s.index)
        valid = ok & p0.notna() & np.isfinite(s)
        s, p0 = s[valid], p0[valid]
        if len(s) < top_n:
            continue
        sel = s.nlargest(top_n).index
        w = pd.Series(1.0 / len(sel), index=sel)
        ret = (px_wide.loc[d1].reindex(sel) / p0.reindex(sel) - 1).fillna(0.0)
        gross = float((w * ret).sum())
        turn = 0.5 * float((w.sub(prev_w.reindex(w.index).fillna(0.0), fill_value=0.0)).abs().sum()) \
            if prev_w is not None else 0.5
        prev_w = w
        nav.append(nav[-1] * (1 + gross - turn * 2 * cost))
        dates.append(d1)
    return pd.Series(nav, index=pd.to_datetime(dates))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-cache", default=str(
        ROOT / "artifacts" / "alphaagent" / "harness_runs" / "2026-10-20_increment" / "full_period_scores_purged.parquet"))
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from core.engine import run_backtest_config
    from core.engine.config import BacktestConfig

    sdf = pd.read_parquet(args.score_cache)
    dcol = "date" if "date" in sdf.columns else "datetime"
    sdf[dcol] = pd.to_datetime(sdf[dcol])
    score_long = sdf.set_index([dcol, "instrument"])["score"].sort_index()
    print(f"[neut] 分数 {len(score_long)} 行", flush=True)

    need = ("adj_close", "close", "amount", "turnover_rate", "industry_sw_l1",
            "is_trade", "not_st")
    t0 = time.perf_counter()
    panel = load_panel_from_cne(start="2020-06-01", end=args.end,
                                include_fundamentals=False, keep_columns=need)
    print(f"[neut] panel {panel.shape} load={time.perf_counter()-t0:.0f}s", flush=True)

    close_wide = panel["close"].astype("float64").unstack("instrument")
    mom20 = close_wide.pct_change(20)
    vol20 = close_wide.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS)
    am20 = panel["amount"].astype("float64").unstack("instrument").rolling(20).mean()
    turn20 = panel["turnover_rate"].astype("float64").unstack("instrument").rolling(20).mean()
    # 行业静态 map（每股取最后非空）
    ind_map = (panel["industry_sw_l1"].dropna()
               .groupby(level=1).last())
    print(f"[neut] 行业 {ind_map.nunique()} 个 / 覆盖 {ind_map.notna().sum()} 股", flush=True)

    # ── 每日截面中性化 ───────────────────────────────────────────────
    styles = {"mom20": mom20, "vol20": vol20, "am20": am20, "turn20": turn20}
    style_rank = {k: v.rank(axis=1, pct=True) - 0.5 for k, v in styles.items()}
    dates_all = score_long.index.get_level_values(0).unique()
    ind_names = sorted(ind_map.dropna().unique())
    ind_of = ind_map.to_dict()

    neut_vals = np.full(len(score_long), np.nan, dtype="float32")
    idx_arr = np.arange(len(score_long))
    # 预取 long 表数组加速
    lvl0 = score_long.index.get_level_values(0)
    lvl1 = score_long.index.get_level_values(1)
    s_arr = score_long.to_numpy(dtype="float64")

    print("[neut] 逐日截面回归...", flush=True)
    t1 = time.perf_counter()
    for gi, d in enumerate(dates_all):
        m = (lvl0 == d)
        if not m.any():
            continue
        codes_d = lvl1[m]
        y = s_arr[m]
        cols = []
        for k in styles:
            try:
                cols.append(style_rank[k].loc[d].reindex(codes_d).to_numpy())
            except KeyError:
                cols.append(np.full(len(codes_d), np.nan))
        X = np.column_stack([np.ones(len(codes_d))] + cols)
        # 行业哑变量
        inds_d = pd.Series([ind_of.get(c) for c in codes_d])
        for ind in ind_names:
            X = np.column_stack([X, (inds_d == ind).to_numpy(dtype="float64")])
        good = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if good.sum() < 500:
            continue
        Xg, yg = X[good], y[good]
        beta, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
        resid = y - X @ beta
        # 残差截面 zscore（回 top_n 排序的量纲稳定）
        r = resid[good]
        r = (r - r.mean()) / (r.std() + 1e-12)
        neut_vals[idx_arr[m][good]] = r.astype("float32")
        if (gi + 1) % 200 == 0:
            print(f"  [{gi+1}/{len(dates_all)}] {time.perf_counter()-t1:.0f}s", flush=True)
    score_neut = pd.Series(neut_vals, index=score_long.index, name="score_neut")
    print(f"[neut] 中性化完成 {time.perf_counter()-t1:.0f}s，"
          f"非空 {(~np.isnan(neut_vals)).sum()} 行", flush=True)

    if args.out:
        out_s = Path(args.out)
        out_s.parent.mkdir(parents=True, exist_ok=True)
        score_neut.rename("score").reset_index().rename(
            columns={"level_0": "datetime", "level_1": "instrument"})\
            .to_parquet(str(out_s).replace(".json", "_neutral_scores.parquet"), index=False)
        print(f"[neut] 中性化分数写入 {out_s}".replace(".json", "_neutral_scores.parquet"), flush=True)

    # ── 对比回测 ─────────────────────────────────────────────────────
    px_wide = panel["adj_close"].astype("float64").unstack("instrument")
    ok_wide = (panel["is_trade"].fillna(False).astype(bool)
               & panel["not_st"].fillna(False).astype(bool)).unstack("instrument")
    sw_orig = score_long.unstack("instrument")
    sw_neut = score_neut.unstack("instrument")
    ppy = TRADING_DAYS / 5

    results = {}
    print("\n[neut] ===== 简化口径 周频 lag=1（top50, 10bp）=====", flush=True)
    for tag, sw in (("original", sw_orig), ("neutralized", sw_neut)):
        nav = _run_backtest(sw.shift(1), px_wide, ok_wide, 5, args.top_n, args.cost_bps)
        mm = _metrics(nav, ppy)
        results[f"weekly_{tag}"] = mm
        print(f"[neut] {tag:12s}: 年化 {mm['annual_return']:+.2%}  夏普 {mm['sharpe']:.2f}  "
              f"回撤 {mm['max_drawdown']:.2%}", flush=True)

    print("\n[neut] ===== 引擎口径 月频（真实执行）=====", flush=True)
    need2 = ("adj_close", "close", "open", "high", "low", "amount", "volume",
             "is_trade", "not_st", "is_st", "up_limit", "down_limit")
    panel2 = load_panel_from_cne(start="2020-06-01", end=args.end,
                                 include_fundamentals=False, keep_columns=need2)
    panel2 = panel2.reset_index().rename(columns={"datetime": "date", "instrument": "code"})
    panel2["turnover"] = panel2["turnover_rate"]
    panel2["am20"] = panel2.groupby("code")["amount"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    panel2["turn20"] = panel2.groupby("code")["turnover_rate"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    codes2 = sorted(panel2["code"].unique())
    wide_orig = sdf.pivot_table(index=dcol, columns="instrument", values="score")
    # 复用已算好的中性化分数（逐日回归 47s，避免重算）
    neut_path = Path(str(args.out).replace(".json", "_neutral_scores.parquet")) if args.out else None
    if neut_path and neut_path.is_file():
        ndf = pd.read_parquet(neut_path)
        ndc = ndf.columns[0] if "datetime" not in ndf.columns else "datetime"
        ndf[ndc] = pd.to_datetime(ndf[ndc])
        wide_neut = ndf.pivot_table(index=ndc, columns="instrument", values="score")
        print(f"[neut] 复用中性化分数 {len(ndf)} 行", flush=True)
    else:
        sdf_n = score_neut.rename("score").reset_index()
        dc2 = sdf_n.columns[0]
        wide_neut = sdf_n.pivot_table(index=dc2, columns="instrument", values="score")
    for tag, wide in (("original", wide_orig), ("neutralized", wide_neut)):
        try:
            cfg = BacktestConfig(panel=panel2, codes=codes2, factor="pred", ascending=False,
                                 start=args.start, end=args.end, capital=1_000_000.0,
                                 top_n=args.top_n, freq="monthly", external_scores=wide)
            res = run_backtest_config(cfg)
            m = res.get("metrics") or {}
            print(f"[neut] {tag:12s}: 年化 {m.get('年化收益', float('nan')):+.2%}  "
                  f"夏普 {m.get('夏普', float('nan')):.2f}  回撤 {m.get('最大回撤', float('nan')):.2%}",
                  flush=True)
            results[f"engine_{tag}"] = {k: v for k, v in m.items()}
        except Exception as exc:  # noqa: BLE001
            print(f"[neut] engine {tag} 失败: {type(exc).__name__}: {exc}", flush=True)

    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                       encoding="utf-8")
        print(f"\n[neut] 明细写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
