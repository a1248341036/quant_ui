#!/usr/bin/env python3
"""统一盲测裁决 harness：对双方（原版 AlphaAgent / quant_ui）交付的因子表达式，
在同一盲测段（默认 2025-01-01 起，挖掘链路从未见过）用 quant_ui 引擎做一次性离线裁决。

指标（论文口径 + quant_ui 口径对齐）:
  - IC       : 逐日截面 RankIC 均值（Spearman，ddof=1）
  - ICIR     : IC 均值 / IC 标准差（不年化，与 cs_ic_summary 同口径）
  - coverage : 盲测段平均截面覆盖率
  - LS_AR / LS_Sharpe / LS_MDD : 十分组多空（D10-D1）日收益年化 / 夏普 / 最大回撤
  - L_AR     : 十分组多头（D10）年化超额（减全样本日均收益）
  - yearly_ic: 分年度 IC（alpha 衰减观察，论文核心主张）

多重检验警示：盲测段每重测一次就烧掉一分，克制重跑。

用法:
  .venv/Scripts/python.exe scripts/alphaagent_blind_judge.py \
      --expr-dir D:/Quant/AlphaAgent/artifacts/factorzoo/stock_1d/expressions \
      --expr-dir artifacts/alphaagent/factorzoo/production_main/expressions \
      --out artifacts/alphaagent/blind_judge/report_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

QUANT_UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUANT_UI_ROOT))

from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402
from alphaagent.dsl import eval_factor  # noqa: E402
from alphaagent.factor.stacking.dataset import daily_spearman_ic  # noqa: E402


def _day_bounds(panel: pd.DataFrame) -> dict[pd.Timestamp, tuple[int, int]]:
    dt = panel.index.get_level_values("datetime")
    codes, start = pd.factorize(dt.to_numpy(), sort=True)
    order = np.argsort(codes, kind="stable")
    bounds: dict[pd.Timestamp, tuple[int, int]] = {}
    for gi in range(len(start)):
        pos = np.flatnonzero(codes[order] == gi)
        bounds[pd.Timestamp(start[gi])] = (int(pos[0]), int(pos[-1]) + 1)
    return bounds


def long_short_decile(
    scores: pd.Series, label: pd.Series, day_bounds: dict, n_decile: int = 10
) -> pd.Series:
    """逐日十分位 D10-D1 等权日收益（label 为前瞻收益，信号日对齐）。"""
    out = []
    for d, (a, b) in day_bounds.items():
        s = scores.iloc[a:b]
        y = label.iloc[a:b]
        m = s.notna() & y.notna()
        if m.sum() < n_decile * 5:
            continue
        sv = s[m].to_numpy()
        yv = y[m].to_numpy()
        q = pd.qcut(sv, n_decile, labels=False, duplicates="drop")
        if q is None or len(np.unique(q)) < n_decile:
            continue
        d10 = yv[q == n_decile - 1].mean()
        d1 = yv[q == 0].mean()
        out.append((d, d10, d10 - d1, float(np.mean(yv))))
    if not out:
        return pd.DataFrame(columns=["d10", "ls", "mkt"]).set_index(pd.DatetimeIndex([]))
    return pd.DataFrame(
        [(row[1], row[2], row[3]) for row in out],
        columns=["d10", "ls", "mkt"],
        index=pd.DatetimeIndex([row[0] for row in out]),
    )


def perf_stats(ret: pd.Series, freq: int = 252) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {}
    cum = (1 + r).cumprod()
    yrs = len(r) / freq
    ar = cum.iloc[-1] ** (1 / yrs) - 1 if cum.iloc[-1] > 0 else -1.0
    vol = r.std(ddof=1) * np.sqrt(freq)
    sharpe = (r.mean() * freq / vol) if vol > 0 else 0.0
    dd = (cum / cum.cummax() - 1).min()
    return {"ann_ret": float(ar), "sharpe": float(sharpe), "max_dd": float(dd)}


def judge_expression(expr: str, panel: pd.DataFrame, label: pd.Series, bounds: dict) -> dict:
    t0 = time.perf_counter()
    s = eval_factor(expr, panel)
    s = s.reindex(panel.index) if not s.index.equals(panel.index) else s
    # ST 剔除：与 submit 盲测终审同口径（评估横截面不含风险警示板）
    from alphaagent.factor.metrics.st_mask import mask_values as _st_mask_values

    s = _st_mask_values(s, panel)
    dts = panel.index.get_level_values("datetime")
    ic = daily_spearman_ic(
        s.to_numpy(dtype=np.float64), label.to_numpy(dtype=np.float64), dts
    )
    ic = ic.dropna()
    res: dict = {
        "ic": float(ic.mean()) if len(ic) else None,
        "icir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 2 and ic.std(ddof=1) > 0 else None,
        "ic_days": int(len(ic)),
        "coverage": float(s.notna().mean()),
        "eval_ms": round((time.perf_counter() - t0) * 1000),
    }
    yearly = ic.groupby(ic.index.year).mean()
    res["yearly_ic"] = {str(k): round(float(v), 5) for k, v in yearly.items()}
    ls = long_short_decile(s, label, bounds)
    if len(ls) > 20:
        res["long_short"] = perf_stats(ls["ls"])
        res["long_only_excess"] = perf_stats(ls["d10"] - ls["mkt"])
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expr-dir", action="append", required=True, help="表达式目录（可多次，标签=目录名）")
    p.add_argument("--start", default="2025-01-01", help="盲测段起始（锁定段，勿频繁重测）")
    p.add_argument("--end", default=None)
    p.add_argument("--label", default="label_1d_close_to_close")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    end = args.end or ""
    panel = load_panel_from_cne(start=args.start, end=end or None)
    # adapter 会向前回看缓冲（滚动算子需要历史），盲测样本必须精确裁到请求区间
    dt_lvl = panel.index.get_level_values("datetime")
    panel = panel[(dt_lvl >= pd.Timestamp(args.start)) & (dt_lvl <= pd.Timestamp(end or dt_lvl.max()))]
    if args.label not in panel.columns:
        raise SystemExit(f"panel 缺少标签列 {args.label}")
    label = panel[args.label]
    bounds = _day_bounds(panel)
    dt = panel.index.get_level_values("datetime")
    print(f"blind panel: {panel.shape}  {dt.min().date()} -> {dt.max().date()}")

    report: dict = {"panel_range": [str(dt.min().date()), str(dt.max().date())], "label": args.label, "factors": {}}
    for d in args.expr_dir:
        dd = Path(d)
        exprs = sorted(dd.glob("*.dsl"))
        print(f"\n[{dd}] {len(exprs)} expressions")
        for f in exprs:
            expr = f.read_text(encoding="utf-8").strip()
            try:
                res = judge_expression(expr, panel, label, bounds)
                ic = res.get("ic")
                print(
                    f"  {f.stem:52s} IC={ic:+.4f} ICIR={res['icir']:+.3f} cov={res['coverage']:.2f}"
                    f" LS_AR={res.get('long_short', {}).get('ann_ret', float('nan')):+.2%}"
                    f" LS_Sharpe={res.get('long_short', {}).get('sharpe', float('nan')):.2f}"
                )
            except Exception as exc:  # noqa: BLE001
                res = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                print(f"  {f.stem:52s} ERROR {res['error'][:80]}")
            report["factors"][f"{dd.name}/{f.stem}"] = res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
