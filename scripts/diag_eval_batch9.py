"""一次性诊断（实验3）：完整复算 run2 batch9 全部 8 个表达式（含实验2 漏掉的
bottom_fractal_dist / max_amount_dist20 两个 TS_LAST_ARGBOTTOMFRACTAL / TS_ARGMAX kernel）。

与 run2 相同 panel（3151819×176 三段）、相同 12 路 ThreadPoolExecutor 并发、
相同表达式。若全部 <15s => 因子/内核/并发均非慢因，假超时源于 run2 进程当时环境；
若 bottom/max 数百秒 => 这两个 kernel 本身在并发下爆炸，是真慢因。

用法: .venv\\Scripts\\python.exe $env:TEMP\\diag_eval_batch9.py
"""
from __future__ import annotations

import concurrent.futures
import time

from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.data.panel import slice_panel
from alphaagent.dsl import eval_factor

START, END = "2020-01-01", "2022-12-31"

EXPRS = [
    ("div_yield_ttm(MAB)", "dy = CS_ZSCORE(RANK($dv_ttm))\ndy"),
    ("listed_days_old(MAB)", "ld = CS_ZSCORE(RANK($listed_days))\nld"),
    ("float_lockup_ratio(MAB)", "lr = DIVIDE($turnover_rate_f, $turnover_rate)\nCS_ZSCORE(RANK(lr))"),
    ("vol_spike_cooldown(417.7s)", "sc = TS_SINCE(GT($volume_ratio, 1.5))\nCS_ZSCORE(RANK(sc))"),
    ("top_fractal_dist(601s超时)", "tf = TS_LAST_ARGTOPFRACTAL($adj_high, $adj_low)\nCS_ZSCORE(RANK(tf))"),
    ("bottom_fractal_dist(无结果!)", "bf = TS_LAST_ARGBOTTOMFRACTAL($adj_high, $adj_low)\nCS_ZSCORE(RANK(bf))"),
    ("pb_valuation_rev(604.8s超时)", "pb = CS_ZSCORE(RANK($pb))\npb"),
    ("max_amount_dist20(无结果!)", "md = TS_ARGMAX($amount, 20)\nCS_ZSCORE(RANK(md))"),
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

    print("=== 12 路并发 (同一批 8 个, 模拟 run2 batch9) ===", flush=True)
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