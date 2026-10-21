# -*- coding: utf-8 -*-
"""全区间业绩（带项目标准仓位控制）—— 把缓存分数喂给 core.engine 回测（2026-10-20）。

**为什么**：`full_period_sharpe.py` 的简化等权口径没带仓位控制（夏普 0.40、回撤
−37%）。项目已盲测终审通过的 vol targeting（0.25 档）+ sell-down（增量预算）在
`core.engine.BacktestConfig` 里原生支持，本脚本用**同一份 walk-forward 分数**
（`full_period_scores.parquet`，2021-2026，训练永不看未来）跑三组对照：

    A. 仓位控制 off（与简化口径可比）
    B. vol_target_annual=0.25 + sell_down_to_target=True（推荐配置，盲测终审通过）
    C. vol_target_annual=0.15（更保守档）

分数生成纪律不变：训练段永不看未来，2025+ 盲测段只预测不训练。
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

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCORE_CACHE = ROOT / "artifacts" / "alphaagent" / "harness_runs" / "2026-10-20_increment" / "full_period_scores.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-cache", default=str(SCORE_CACHE))
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--capital", type=float, default=1_000_000.0,
                    help="引擎回测资金（默认 100 万；此前 10 万口径会出现现金不足拒单）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from core.engine import run_backtest_config
    from core.engine.config import BacktestConfig

    sc_path = Path(args.score_cache)
    if not sc_path.is_file():
        print(f"[eng] 分数缓存不存在: {sc_path}（先跑 full_period_sharpe.py --score-cache）")
        return 1
    t0 = time.perf_counter()
    sdf = pd.read_parquet(sc_path)
    dcol = "date" if "date" in sdf.columns else ("datetime" if "datetime" in sdf.columns else sdf.columns[0])
    ccol = "instrument" if "instrument" in sdf.columns else sdf.columns[1]
    sdf[dcol] = pd.to_datetime(sdf[dcol])
    print(f"[eng] 分数 {len(sdf)} 行  {sdf[dcol].min().date()} ~ {sdf[dcol].max().date()}  "
          f"load={time.perf_counter()-t0:.0f}s", flush=True)

    # 引擎需要的面板列：价格/交易状态/流动性约束
    need_cols = ("close", "adj_close", "open", "high", "low", "amount", "volume",
                 "is_trade", "not_st", "is_st", "up_limit", "down_limit")
    panel = load_panel_from_cne(start="2020-06-01", end=args.end,
                                include_fundamentals=False,
                                keep_columns=tuple(need_cols))
    print(f"[eng] panel {panel.shape}", flush=True)

    score = sdf.set_index([dcol, ccol])["score"]
    panel = panel.join(score.rename("ml_score"), how="left")
    n_scored = int(panel["ml_score"].notna().sum())
    print(f"[eng] ml_score 并入面板：覆盖 {n_scored}/{len(panel)} 行", flush=True)

    # core.engine 吃**平表**（date/code 列，JQ panel 口径），把 MultiIndex 展平
    panel = panel.reset_index().rename(columns={"datetime": "date", "instrument": "code"})
    # JQ 口径派生列：am20/turn20 是引擎的输入列（20 日均值），turnover 是换手原名
    panel["turnover"] = panel["turnover_rate"]
    panel["am20"] = panel.groupby("code")["amount"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    panel["turn20"] = panel.groupby("code")["turnover_rate"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    codes = sorted(panel["code"].unique())
    print(f"[eng] 展平后 {panel.shape}，codes {len(codes)}", flush=True)

    # 引擎约定：外部分数因子名固定为 "pred"，external_scores 是 date×code 宽表
    score_wide = sdf.pivot_table(index=dcol, columns=ccol, values="score")
    print(f"[eng] 分数宽表 {score_wide.shape}", flush=True)

    base = dict(
        panel=panel, codes=codes, factor="pred", ascending=False,
        start=args.start, end=args.end, capital=args.capital, top_n=args.top_n,
        freq="monthly", external_scores=score_wide,
    )

    variants = [
        ("A_off",       {}),
        ("B_vol25_sd",  {"vol_target_annual": 0.25, "vol_target_lo": 0.3,
                         "vol_target_hi": 1.5, "vol_target_window": 60,
                         "sell_down_to_target": True}),
        ("C_vol15_sd",  {"vol_target_annual": 0.15, "vol_target_lo": 0.3,
                         "vol_target_hi": 1.5, "vol_target_window": 60,
                         "sell_down_to_target": True}),
    ]

    results = {}
    for name, extra in variants:
        print(f"\n[eng] ===== {name} {extra or ''} =====", flush=True)
        try:
            cfg = BacktestConfig(**base, **extra)
            res = run_backtest_config(cfg)
            results[name] = res
            keep = ("annual_return", "sharpe", "max_drawdown", "excess_annual",
                    "excess_sharpe", "avg_num_hold", "total_return")
            brief = {k: res.get(k) for k in keep if k in res}
            if not brief and isinstance(res, dict):
                gm = res.get("gate", {}).get("group_metrics") or res.get("group_metrics") or {}
                brief = {k: gm.get(k) for k in ("annual_return", "sharpe", "max_drawdown",
                                                "excess_annual", "excess_sharpe") if k in gm}
            print(f"[eng] {name}: {json.dumps(brief, ensure_ascii=False, default=str)[:400]}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[eng] {name} 失败: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
        print(f"\n[eng] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
