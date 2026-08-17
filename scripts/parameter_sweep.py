#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数稳健性扫描：walk-forward + 参数热力图。

用法：
  # 双均线金叉事件策略的参数网格（默认）
  python scripts/parameter_sweep.py --mode event \
      --short-list 3,5,8,10,13 --long-list 10,20,30,60 --folds 4

  # 已有因子策略的 walk-forward（不做参数扫描）
  python scripts/parameter_sweep.py --mode factor --strategy "双均线多头 5/20"

输出到 results/parameter_sweep/：
  sweep_summary.csv    参数组合 x 跨窗口汇总（按均值夏普降序）
  sweep_windows.csv    逐参数逐窗口明细
  heatmap_sharpe.csv   参数热力图数据（short x long -> 均值夏普）
  heatmap_sharpe.png   参数热力图
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data import load_panel, load_tech  # noqa: E402
from core.backtest_archive import save_run  # noqa: E402
from core.walkforward import (  # noqa: E402
    golden_cross_sweep,
    walk_forward_factor,
)
from strategies.registry import STRATEGIES  # noqa: E402


def build_codes():
    panel = load_panel()
    tech = load_tech()
    codes = sorted(set(tech["code"]) & set(panel["code"].unique()))
    codes = [c for c in codes if not c.startswith(("300", "301", "688", "689"))]
    return panel, codes


def save_heatmap(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(df.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([f"long={c}" for c in df.columns])
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([f"short={r}" for r in df.index])
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = df.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Mean Sharpe by MA period (walk-forward)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["event", "factor"], default="event")
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--capital", type=float, default=50000.0)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--max-weight", type=float, default=0.5)
    ap.add_argument("--short-list", default="3,5,8,10,13")
    ap.add_argument("--long-list", default="10,20,30,60")
    ap.add_argument("--strategy", default="双均线多头 5/20")
    ap.add_argument("--out", default="results/parameter_sweep")
    args = ap.parse_args()

    panel, codes = build_codes()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "event":
        shorts = [int(x) for x in args.short_list.split(",") if x.strip()]
        longs = [int(x) for x in args.long_list.split(",") if x.strip()]
        print(f"双均线金叉参数扫描: short={shorts} long={longs} "
              f"folds={args.folds} 窗口={args.start}~{args.end}")
        summary, heatmap, windows = golden_cross_sweep(
            panel, codes, args.start, args.end, args.capital,
            short_list=shorts, long_list=longs, n_folds=args.folds,
            top_n=args.top_n, max_weight=args.max_weight,
            warmup_days=args.warmup,
        )
        summary.to_csv(out / "sweep_summary.csv", index=False)
        windows.to_csv(out / "sweep_windows.csv", index=False)
        heatmap.to_csv(out / "heatmap_sharpe.csv")
        save_heatmap(heatmap, out / "heatmap_sharpe.png")
        save_run(
            kind="sweep_cli",
            params={
                "mode": "event", "start": args.start, "end": args.end,
                "capital": args.capital, "folds": args.folds,
                "short_list": shorts, "long_list": longs,
                "top_n": args.top_n, "max_weight": args.max_weight,
                "warmup": args.warmup,
            },
            summary={
                "mode": "event", "n_combos": len(summary),
                "best": summary.head(5).to_dict(orient="records"),
                "output": str(out / "sweep_summary.csv"),
            },
            nav=summary.to_dict(orient="records"),
            bench=heatmap.reset_index().rename(columns={"index": "short"})
            .to_dict(orient="records"),
            trades=windows.to_dict(orient="records"),
            data_version=str(panel["date"].max().date()),
        )
        cols = ["short", "long", "mean_sharpe", "median_sharpe", "std_sharpe",
                "mean_total", "worst_total", "win_rate", "mean_mdd"]
        pd.set_option("display.width", 160)
        print("\n===== 参数组合稳健性 Top10（按均值夏普） =====")
        print(summary[cols].head(10).to_string(index=False,
              float_format=lambda x: f"{x:.3f}"))
        print(f"\n输出目录: {out}/")
    else:
        if args.strategy not in STRATEGIES:
            print(f"未知策略: {args.strategy}", file=sys.stderr)
            return 1
        s = STRATEGIES[args.strategy]
        print(f"因子策略 walk-forward: {args.strategy} "
              f"(factor={s['factor']}, folds={args.folds})")
        windows = walk_forward_factor(
            panel, codes, s["factor"], s["ascending"],
            args.start, args.end, args.capital,
            top_n=args.top_n, n_folds=args.folds, warmup_days=args.warmup,
        )
        windows.to_csv(out / "factor_windows.csv", index=False)
        save_run(
            kind="sweep_cli",
            params={
                "mode": "factor", "strategy": args.strategy,
                "start": args.start, "end": args.end,
                "capital": args.capital, "folds": args.folds,
                "top_n": args.top_n, "warmup": args.warmup,
            },
            summary={
                "mode": "factor", "strategy": args.strategy,
                "n_windows": len(windows),
                "output": str(out / "factor_windows.csv"),
            },
            nav=windows.to_dict(orient="records"),
            data_version=str(panel["date"].max().date()),
        )
        pd.set_option("display.width", 160)
        print("\n===== 逐窗口指标 =====")
        print(windows.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\n汇总:", {
            "mean_total": float(windows["total"].mean()),
            "mean_sharpe": float(windows["sharpe"].mean()),
            "worst_total": float(windows["total"].min()),
            "win_rate": float((windows["total"] > 0).mean()),
        })
        print(f"\n输出目录: {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
