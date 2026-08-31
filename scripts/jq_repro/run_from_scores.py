# -*- coding: utf-8 -*-
"""因子分数矩阵 -> 事件引擎 桥接验证。

用 -log(总市值) 作为分数矩阵(越大=市值越小=越优先, 与小市值策略同序),
风险筛选(三正/ST/停牌/次新/退市名/价格上限)走 filters 接口,
排序与筛选解耦 —— 这正是 AlphaAgent 挖掘因子接入时的形态。

预期: 与 run_smallcap.py(cand_fn 直连版) 指标逐位一致。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import jq_data  # noqa: E402
from template import ReproConfig, ma_num_fn, run, score_cand_fn  # noqa: E402

MIN_MV, MAX_MV = 3e4, 1e7        # 万元: 3亿 ~ 1000亿
HIGHEST = 60.0                   # 元
TOP_KEEP = 50


def main() -> None:
    t0 = time.time()
    engine_panel, meta, close_raw_df = jq_data.load_panel("2016-01-04", "2026-08-28")
    tables = jq_data.build_tables(engine_panel, meta, close_raw_df)

    # ---- 1) 分数矩阵: -log(市值), 越大越优先(越小市值) ----
    with np.errstate(invalid="ignore"):
        mv = np.where(tables.mv > 0, tables.mv, np.nan)
        score = -np.log(mv)
    score_mat = pd.DataFrame(score, index=tables.dates, columns=tables.codes)
    n_daily = score_mat.notna().sum(axis=1)
    print(f"[{time.time()-t0:.0f}s] 分数矩阵: {score_mat.shape}, "
          f"日均覆盖 {n_daily.mean():.0f} 只", flush=True)

    # ---- 2) 风险筛选(点时): 三正 + ST + 停牌 + 次新 + 退市名 + 价格上限 ----
    inc = jq_data.load_income()
    fin_ok = jq_data.fin_ok_matrix(inc, tables.dates, tables.codes,
                                   jq_data.triple_positive_pred())
    eligible = (fin_ok & ~tables.is_st & ~tables.paused
                & tables.listed_ok & ~tables.delist_name[None, :]
                & np.isfinite(tables.close_raw) & (tables.close_raw <= HIGHEST)
                & (tables.mv >= MIN_MV) & (tables.mv <= MAX_MV))

    def filters(i: int) -> np.ndarray:
        return eligible[i]

    # ---- 3) 桥接: 分数矩阵 -> cand_fn; MA 仓位沿用 ----
    cand_fn = score_cand_fn(score_mat, tables, top_keep=TOP_KEEP,
                            filters=filters)
    level, ma = jq_data.ew_index(tables, clip=0.2, ma_window=10)
    num_fn = ma_num_fn(level, ma, base=7, span=3, scale_rel=0.025, lo=4)

    cfg = ReproConfig(
        start="2016-01-04", end="2026-08-28", capital=1_000_000.0,
        stock_num=7, max_single_weight=0.12, max_exposure=0.70,
        stoploss=0.07, take_profit_multiple=2.0, market_crash=0.05,
        limit_open_sell=True, rebalance_weekday=1, pass_months=(4,),
        buy_cost=0.0001, sell_cost=0.0011, slippage_bps=0.0,
        warmup_days=45, out_dir=HERE / "out_scores",
    )
    out = run(cfg, cand_fn, num_fn, tables=tables, panel=engine_panel,
              bench_level=level)
    m = out["metrics"]
    print(f"[{time.time()-t0:.0f}s] 指标:", flush=True)
    for k, v in m.items():
        if k != "bench":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  bench:", {k: round(v, 4) for k, v in m["bench"].items()})

    # ---- 4) 等价性对照: 与 cand_fn 直连版(run_smallcap) 的 metrics.json ----
    ref = HERE / "out_smallcap" / "metrics.json"
    if ref.exists():
        import json
        ref_m = json.loads(ref.read_text(encoding="utf-8"))
        keys = ["总收益", "年化收益", "年化波动", "夏普", "最大回撤",
                "卡玛", "胜率", "超额年化", "超额夏普"]
        diffs = {k: abs(float(m[k]) - float(ref_m[k])) for k in keys
                 if k in ref_m and k in m}
        worst = max(diffs.values())
        print(f"[等价性] 与 cand_fn 直连版最大指标偏差: {worst:.2e} "
              f"-> {'PASS' if worst < 1e-6 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
