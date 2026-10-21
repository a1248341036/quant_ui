# -*- coding: utf-8 -*-
"""全区间（2021-01 ~ 2026-09）ML 组合年化夏普 —— walk-forward 逐段预测 + 统一回测（2026-10-20）。

**为什么要重跑**：仓库里没有全区间 ML 分数。现有的 score 只有
`retrain_20261011b/scores.parquet`（2024-07~2024-12，119 天）和
`data/stock/pred_demo.parquet`（2025-01 起，408 天），2021-2024 上半段没有预测值。

**口径（唯一、固定、可复现）**
- 因子：候选池（candidate_main 里可求值的）+ ML 增量库（ml_increment_main），"总的"
- Walk-forward（expanding，每年滚一次，**训练永不看未来**）：
      2020 全年训练      → 预测 2021
      2020-2021 训练     → 预测 2022
      2020-2022 训练     → 预测 2023
      2020-2023 训练     → 预测 2024
      2020-2024 训练     → 预测 2025-01 ~ 2026-09（盲测段只预测不训练）
  ⚠️ 2020 年**无法给样本外业绩**（panel 从 2020 起，2020 只能是训练期）。
  ⚠️ 2025-01 起是盲测段：本脚本**不训练**，只做预测+回测，等于消费一次盲测评估。
- 组合：每 20 个交易日调仓；每期按 score 降序取前 `--top-n` 只（默认 50）；
  过滤 is_trade/not_st/有收盘价；**等权**满仓；按 adj_close 计收益；
  换手成本单边 `--cost-bps`（默认 10bp，双边 20bp）。
- 基准：同调仓日、同过滤条件下的**全市场等权**收益。
- 指标：年化收益、年化波动、年化夏普（rf=0）、最大回撤、相对基准的超额年化/超额夏普。

用法：
    python scripts/full_period_sharpe.py --out artifacts/.../full_period.json
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
LABEL = "label_20d_close_to_close"
INC_LIB = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "ml_increment_main" / "mining_ml_increment_registry.json"
TRADING_DAYS = 244.0


def _fit(X: pd.DataFrame, y: pd.Series, mask: pd.Series, args):
    import lightgbm as lgb

    m = lgb.LGBMRegressor(
        n_estimators=args.max_rounds, learning_rate=0.05, num_leaves=31,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
        n_jobs=args.n_jobs, verbose=-1)
    m.fit(X[mask], y[mask])
    return m


def _metrics(nav: pd.Series) -> dict:
    """nav: 日频净值（已按调仓日对齐的日收益复利）。"""
    if len(nav) < 20:
        return {}
    ret = nav.pct_change().dropna()
    ann = float((1 + ret.mean()) ** TRADING_DAYS - 1) if ret.mean() > -1 else float("nan")
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(ann / vol) if vol > 0 else float("nan")
    cummax = nav.cummax()
    mdd = float(((nav - cummax) / cummax).min())
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    return {"n_days": int(len(ret)), "total_return": total, "annual_return": ann,
            "annual_vol": vol, "sharpe": sharpe, "max_drawdown": mdd,
            "start": str(nav.index[0].date()), "end": str(nav.index[-1].date())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--top-n", type=int, default=50, help="每期持仓只数（等权）")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="单边换手成本（bp）")
    ap.add_argument("--rebalance", type=int, default=20, help="调仓间隔（交易日）")
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names
    from alphaagent.factor.mining.eval.combo_increment import cs_zscore, load_candidate_exprs

    # ── 因子集：候选池 + ML 增量库（"总的"）──────────────────────────
    exprs = list(load_candidate_exprs())
    if INC_LIB.is_file():
        reg = json.loads(INC_LIB.read_text(encoding="utf-8"))
        seen = {e["name"] for e in exprs}
        for name, rec in reg.items():
            if name not in seen and rec.get("expr"):
                exprs.append({"name": name, "expr": rec["expr"]})
    print(f"[full] 因子集：{len(exprs)} 个（候选池 + ML 增量库）", flush=True)

    need: set[str] = {LABEL, "adj_close", "is_trade", "not_st"}
    for c in exprs:
        try:
            need |= set(referenced_column_names(c["expr"]))
        except Exception:  # noqa: BLE001
            pass
    need = {c for c in need if not str(c).startswith(("funda_", "fsi_"))}

    t0 = time.perf_counter()
    # 多取 30 天，保证首个调仓日有前值
    panel = load_panel_from_cne(start="2019-12-01", end=args.end,
                                include_fundamentals=False, keep_columns=tuple(sorted(need)))
    print(f"[full] panel {panel.shape} load={time.perf_counter()-t0:.0f}s", flush=True)
    if LABEL not in panel.columns:
        print(f"[full] 缺 {LABEL}")
        return 1

    y = panel[LABEL].astype("float32")
    dates = pd.Series(panel.index.get_level_values(0), index=panel.index)

    print("[full] 求值因子...", flush=True)
    zs = {}
    for i, c in enumerate(exprs, 1):
        try:
            v = eval_multi_line_factor(c["expr"], panel)
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0] if v.shape[1] == 1 else v.mean(axis=1)
            s = pd.Series(v).astype("float32").reindex(panel.index)
            if s.notna().mean() >= 0.5:
                zs[c["name"]] = cs_zscore(s)
        except Exception:  # noqa: BLE001
            continue
        if i % 20 == 0:
            print(f"  [{i}/{len(exprs)}] 可用 {len(zs)}", flush=True)
    X = pd.DataFrame(zs)
    print(f"[full] 可用因子 {X.shape[1]} 个", flush=True)

    # ── Walk-forward 逐段预测 ────────────────────────────────────────
    segments = [
        ("2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2020-01-01", "2024-12-31", "2025-01-01", args.end),
    ]
    preds = []
    for tr_s, tr_e, oo_s, oo_e in segments:
        tr = (dates >= pd.Timestamp(tr_s)) & (dates <= pd.Timestamp(tr_e))
        oo = (dates >= pd.Timestamp(oo_s)) & (dates <= pd.Timestamp(oo_e))
        if tr.sum() < 10000 or oo.sum() < 1000:
            print(f"[full] 跳过段 {oo_s}~{oo_e}（样本不足 tr={int(tr.sum())} oo={int(oo.sum())}）")
            continue
        t1 = time.perf_counter()
        m = _fit(X, y, tr, args)
        p = m.predict(X[oo])
        sub = pd.DataFrame({"score": p}, index=X[oo].index)
        preds.append(sub)
        tag = "盲测段（只预测不训练）" if oo_s >= "2025-01-01" else ""
        print(f"[full] train {tr_s}~{tr_e} → OOS {oo_s}~{oo_e}: "
              f"{int(oo.sum())} 行 训练{time.perf_counter()-t1:.0f}s {tag}", flush=True)
        del m
    if not preds:
        print("[full] 没有可用预测段")
        return 1
    score = pd.concat(preds).sort_index()
    print(f"[full] 预测覆盖 {len(score)} 行 / {score.index.get_level_values(0).nunique()} 天", flush=True)

    # ── 组合回测（20 日调仓、top-N 等权、含成本）─────────────────────
    px = panel["adj_close"].astype("float64") if "adj_close" in panel.columns else None
    if px is None:
        print("[full] 缺 adj_close，无法回测")
        return 1
    trade_ok = pd.Series(True, index=panel.index)
    for col in ("is_trade", "not_st"):
        if col in panel.columns:
            trade_ok &= panel[col].fillna(False).astype(bool)

    day_list = np.array(sorted(score.index.get_level_values(0).unique()))
    px_wide = px.unstack("instrument")
    ok_wide = trade_ok.unstack("instrument")
    score_wide = score["score"].unstack("instrument")

    reb_days = day_list[:: args.rebalance]
    nav_m, nav_b = [1.0], [1.0]
    nav_dates = [reb_days[0]]
    turnovers = []
    holdings = []
    cost = args.cost_bps / 1e4
    prev_w = None
    for k in range(len(reb_days) - 1):
        d0, d1 = reb_days[k], reb_days[k + 1]
        if d0 not in px_wide.index or d1 not in px_wide.index:
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
        # 基准：同日全市场等权
        bsel = p0.index
        bret = (px_wide.loc[d1].reindex(bsel) / p0.reindex(bsel) - 1).fillna(0.0)
        nav_m.append(nav_m[-1] * (1 + net))
        nav_b.append(nav_b[-1] * (1 + float(bret.mean())))
        nav_dates.append(d1)
        turnovers.append(turn)
        holdings.append(len(sel))

    nav_m = pd.Series(nav_m, index=pd.to_datetime(nav_dates))
    nav_b = pd.Series(nav_b, index=pd.to_datetime(nav_dates))
    mm, mb = _metrics(nav_m), _metrics(nav_b)
    if not mm or not mb:
        print("[full] 回测期数不足")
        return 1

    exc_ann = mm["annual_return"] - mb["annual_return"]
    exc_sharpe = mm["sharpe"] - mb["sharpe"]

    # 分段业绩
    seg_rows = []
    for lo, hi, name in (("2021-01-01", "2021-12-31", "2021"),
                         ("2022-01-01", "2022-12-31", "2022"),
                         ("2023-01-01", "2023-12-31", "2023"),
                         ("2024-01-01", "2024-12-31", "2024"),
                         ("2025-01-01", "2026-12-31", "2025-2026 盲测段")):
        sub = nav_m[(nav_m.index >= lo) & (nav_m.index <= hi)]
        if len(sub) < 3:
            continue
        rel = sub / sub.iloc[0]
        s = _metrics(rel)
        if s:
            seg_rows.append({"seg": name, **{k: s[k] for k in
                            ("annual_return", "sharpe", "max_drawdown", "total_return")}})

    print(f"\n[full] ===== 全区间业绩 {mm['start']} ~ {mm['end']} =====")
    print(f"[full] 因子 {X.shape[1]} 个 | 每期 top{args.top_n} 等权 | 调仓 {args.rebalance} 日 | "
          f"成本 {args.cost_bps}bp 单边 | 平均持仓 {np.mean(holdings):.1f} 只 | "
          f"平均单期换手 {np.mean(turnovers):.2f}")
    print(f"[full] 组合   : 年化 {mm['annual_return']:+.2%}  波动 {mm['annual_vol']:.2%}  "
          f"夏普 {mm['sharpe']:.2f}  回撤 {mm['max_drawdown']:.2%}  累计 {mm['total_return']:+.2%}")
    print(f"[full] 基准(全市场等权): 年化 {mb['annual_return']:+.2%}  夏普 {mb['sharpe']:.2f}  "
          f"回撤 {mb['max_drawdown']:.2%}")
    print(f"[full] 超额   : 年化 {exc_ann:+.2%}  夏普差 {exc_sharpe:+.2f}")
    print(f"\n[full] 分段：")
    for r in seg_rows:
        print(f"[full]   {r['seg']:16s} 年化 {r['annual_return']:+7.2%}  "
              f"夏普 {r['sharpe']:5.2f}  回撤 {r['max_drawdown']:+7.2%}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "config": {"start": mm["start"], "end": mm["end"], "n_factors": int(X.shape[1]),
                       "top_n": args.top_n, "rebalance": args.rebalance,
                       "cost_bps": args.cost_bps,
                       "avg_holdings": float(np.mean(holdings)),
                       "avg_turnover": float(np.mean(turnovers))},
            "portfolio": mm, "benchmark": mb,
            "excess": {"annual": exc_ann, "sharpe": exc_sharpe},
            "segments": seg_rows,
            "note": "walk-forward expanding，训练永不看未来；2025-01 起为盲测段（只预测不训练），本报告即消费一次盲测评估",
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[full] 报告写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
