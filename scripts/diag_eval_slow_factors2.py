"""一次性诊断（实验2）：12 路并发复算慢因子 vs 单线程，量化并发放大系数。

与实验1同一 panel、同一 eval_factor；用 ThreadPoolExecutor(12) 模拟 run2 的
线程池并发。若并发下这些因子同样 100-600s，则问题在并发环境（CPU 争抢/
GIL/排队）；若并发下依旧 <3s，则 run2 当时有外部干扰（僵尸进程占 CPU）。

用法: .venv\Scripts\python.exe scripts\diag_eval_slow_factors2.py
"""
from __future__ import annotations

import concurrent.futures
import time

from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.data.panel import slice_panel
from alphaagent.dsl import eval_factor

START, END = "2020-01-01", "2022-12-31"

EXPRS = [
    ("div_yield_ttm(run2:12.7s)", "dy = CS_ZSCORE(RANK($dv_ttm))\ndy"),
    ("vw_res_weekly_dev(run2:131.7s)", "week_res = SUBTRACT($adj_close@1w, $adj_vwap@1w)\nCS_ZSCORE(RANK(week_res))"),
    ("weekly_mom4(run2:133.2s)", "wm = TS_PCTCHANGE($adj_close@1w, 4)\nCS_ZSCORE(RANK(wm))"),
    ("vol_spike_cooldown(run2:417.7s)", "sc = TS_SINCE(GT($volume_ratio, 1.5))\nCS_ZSCORE(RANK(sc))"),
    ("top_fractal_dist(run2:601s超时)", "tf = TS_LAST_ARGTOPFRACTAL($adj_high, $adj_low)\nCS_ZSCORE(RANK(tf))"),
    ("pb_valuation_rev(run2:604.8s超时)", "pb = CS_ZSCORE(RANK($pb))\npb"),
]

def _run_one(item):
    name, expr = item
    t = time.perf_counter()
    try:
        out = eval_factor(expr, PANEL)
        msg = f"ok nonnull={int(out.notna().sum())}" if hasattr(out, "notna") else f"ok {type(out).__name__}"
    except Exception as exc:  # noqa: BLE001
        msg = f"ERR {type(exc).__name__}: {str(exc)[:80]}"
    return name, round((time.perf_counter() - t) * 1000), msg

PANEL = None

def main() -> None:
    global PANEL
    t0 = time.perf_counter()
    PANEL = slice_panel(
        load_panel_from_cne(start=START, end=END, universe_mask=False,
                            include_fundamentals=True, asset_type="stock"),
        start=START, end=END,
    )
    print(f"panel={PANEL.shape} {(time.perf_counter()-t0):.0f}s", flush=True)

    print("=== 单线程基线 (repeat) ===", flush=True)
    single = {}
    for item in EXPRS:
        name, ms, msg = _run_one(item)
        single[name] = ms
        print(f"  {name}: {ms:.0f} ms  {msg}", flush=True)

    print("=== 12 路并发 (同一批 6 个, 模拟 run2 11:43:25 批) ===", flush=True)
    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(_run_one, item) for item in EXPRS]
        for f in concurrent.futures.as_completed(futs):
            name, ms, msg = f.result()
            ratio = ms / single.get(name, 1)
            print(f"  {name}: {ms:.0f} ms  (x{ratio:.1f} vs single)  {msg}", flush=True)
    print(f"  批总耗时 {(time.perf_counter()-t):.0f} s", flush=True)

if __name__ == "__main__":
    main()