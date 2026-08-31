# -*- coding: utf-8 -*-
"""小市值三正策略 —— 用 pipeline 模板表达(等价性验证)。

与根目录 repro_jq_smallcap.py 必须产出一致指标:
- 候选: 主板域, 市值3~1000亿 + 三正财务 + 非ST/停牌/次新/退市名,
        按市值升序, 价格<=60元(未复权), 收盘涨停不入池
- 仓位: MA10 sigmoid 连续映射(域等权指数, 相对阈值 2.5%), 4~7 只
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import jq_data  # noqa: E402
from template import ReproConfig, ma_num_fn, run  # noqa: E402

MIN_MV, MAX_MV = 3e4, 1e7        # 万元: 3亿 ~ 1000亿
HIGHEST = 60.0                   # 元
TOP_KEEP = 50                    # 候选保留数(原版 stock_num*3=21, 留余量)


def build_cand_fn(tables: jq_data.MarketTables, fin_ok: np.ndarray):
    """预计算每个信号日的候选列表: [(code, passed, is_hl), ...] 市值升序."""
    eligible = ((tables.mv >= MIN_MV) & (tables.mv <= MAX_MV)
                & fin_ok & ~tables.is_st & ~tables.paused
                & tables.listed_ok & ~tables.delist_name[None, :])
    mv_e = np.where(eligible, tables.mv, np.inf)
    close_raw, hl = tables.close_raw, tables.hl
    T, K = tables.mv.shape
    cand: dict = {}
    for i in range(T):
        row = mv_e[i]
        if not np.isfinite(row).any():
            cand[tables.dates[i]] = []
            continue
        k = min(TOP_KEEP, int(np.isfinite(row).sum()))
        idx = np.argpartition(row, k - 1)[:k]
        idx = idx[np.isfinite(row[idx])]
        idx = idx[np.argsort(row[idx], kind="stable")]
        cand[tables.dates[i]] = [
            (tables.codes[j], bool(close_raw[i, j] <= HIGHEST), bool(hl[i, j]))
            for j in idx]
    return lambda sig: cand.get(sig, [])


def main() -> None:
    import time
    t0 = time.time()
    engine_panel, meta, close_raw_df = jq_data.load_panel("2016-01-04", "2026-08-28")
    tables = jq_data.build_tables(engine_panel, meta, close_raw_df)
    inc = jq_data.load_income()
    fin_ok = jq_data.fin_ok_matrix(inc, tables.dates, tables.codes,
                                   jq_data.triple_positive_pred())
    cand_fn = build_cand_fn(tables, fin_ok)
    level, ma = jq_data.ew_index(tables, clip=0.2, ma_window=10)
    num_fn = ma_num_fn(level, ma, base=7, span=3, scale_rel=0.025, lo=4)
    print(f"[{time.time()-t0:.0f}s] 数据+候选就绪", flush=True)

    cfg = ReproConfig(
        start="2016-01-04", end="2026-08-28", capital=1_000_000.0,
        stock_num=7, max_single_weight=0.12, max_exposure=0.70,
        stoploss=0.07, take_profit_multiple=2.0, market_crash=0.05,
        limit_open_sell=True, rebalance_weekday=1, pass_months=(4,),
        buy_cost=0.0001, sell_cost=0.0011, slippage_bps=0.0,
        warmup_days=45, out_dir=HERE / "out_smallcap",
    )
    out = run(cfg, cand_fn, num_fn, tables=tables, panel=engine_panel)
    m = out["metrics"]
    print(f"[{time.time()-t0:.0f}s] 指标:", flush=True)
    for k, v in m.items():
        if k != "bench":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  bench:", {k: round(v, 4) for k, v in m["bench"].items()})


if __name__ == "__main__":
    main()
