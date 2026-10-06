"""一次性诊断：复算 run2 的 5 个慢/超时因子 vs 快的 div_yield_ttm。

在 run2 同一 panel（train 2020-01-01~2022-12-31, cne:// 源）上单线程逐表达式
调用 eval_factor 计时，判断 dsl_ms 是真实计算成本还是并发排队/资源争抢所致。

用法: .venv\Scripts\python.exe scripts\diag_eval_slow_factors.py
"""
from __future__ import annotations

import time

from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.data.panel import slice_panel
from alphaagent.dsl import eval_factor

START, END = "2020-01-01", "2022-12-31"

EXPRS = [
    ("div_yield_ttm(快, run2:12.7s)", "dy = CS_ZSCORE(RANK($dv_ttm))\ndy"),
    ("vw_res_weekly_dev(run2:131.7s)", "week_res = SUBTRACT($adj_close@1w, $adj_vwap@1w)\nCS_ZSCORE(RANK(week_res))"),
    ("weekly_mom4(run2:133.2s)", "wm = TS_PCTCHANGE($adj_close@1w, 4)\nCS_ZSCORE(RANK(wm))"),
    ("vol_spike_cooldown(run2:417.7s)", "sc = TS_SINCE(GT($volume_ratio, 1.5))\nCS_ZSCORE(RANK(sc))"),
    ("top_fractal_dist(run2:601s超时)", "tf = TS_LAST_ARGTOPFRACTAL($adj_high, $adj_low)\nCS_ZSCORE(RANK(tf))"),
    ("pb_valuation_rev(run2:604.8s超时)", "pb = CS_ZSCORE(RANK($pb))\npb"),
]

def main() -> None:
    t0 = time.perf_counter()
    panel = load_panel_from_cne(
        start=START, end=END, universe_mask=False,
        include_fundamentals=True, asset_type="stock",
    )
    panel = slice_panel(panel, start=START, end=END)
    print(f"panel loaded: shape={panel.shape}, {len(panel.columns)} cols, "
          f"{panel.index.get_level_values('instrument').nunique()} inst, {(time.perf_counter()-t0):.1f}s", flush=True)
    wanted = [c for c in panel.columns if str(c) in ("pb", "dv_ttm", "volume_ratio")]
    print("wanted cols present:", wanted, flush=True)
    for name, expr in EXPRS:
        t = time.perf_counter()
        try:
            out = eval_factor(expr, panel)
            if hasattr(out, "shape"):
                msg = f"ok shape={out.shape} nonnull={int(out.notna().sum())}"
            else:
                msg = f"ok type={type(out).__name__}"
        except Exception as exc:  # noqa: BLE001
            msg = f"ERR {type(exc).__name__}: {str(exc)[:100]}"
        print(f"{name}: {(time.perf_counter()-t)*1000:.0f} ms  {msg}", flush=True)

if __name__ == "__main__":
    main()