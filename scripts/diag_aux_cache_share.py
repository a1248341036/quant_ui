"""一次性诊断（实验3）：session 级 aux_cache 共享命中验证。

模拟 run2 的 engine 路径：同一个 split panel + 同一个 OrderedDict aux_cache，
连续评估两个 @1w 因子。若第二个命中缓存应 <1s；若 ~127s 则 aux_cache 未生效。
同时打印 key 与 id(panel)。
"""
from __future__ import annotations

import time
from collections import OrderedDict

from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.data.panel import slice_panel
from alphaagent.dsl.stock.aux_cache import get_or_build_aux_panel, _MAX_CACHE_ENTRIES

START, END = "2020-01-01", "2022-12-31"
EXPR1 = "week_res = SUBTRACT($adj_close@1w, $adj_vwap@1w)\nCS_ZSCORE(RANK(week_res))"
EXPR2 = "wm = TS_PCTCHANGE($adj_close@1w, 4)\nCS_ZSCORE(RANK(wm))"

def main() -> None:
    from alphaagent.dsl import eval_factor
    t0 = time.perf_counter()
    panel = slice_panel(
        load_panel_from_cne(start=START, end=END, universe_mask=False,
                            include_fundamentals=True, asset_type="stock"),
        start=START, end=END,
    )
    print(f"panel={panel.shape} load={(time.perf_counter()-t0):.0f}s id(panel)={id(panel)}", flush=True)
    aux_cache: OrderedDict = OrderedDict()

    # 先直接探测 build_timeframe_panel 的裸成本（不经 eval_factor 的 guard/prune 干扰）
    t = time.perf_counter()
    w = get_or_build_aux_panel(panel, "1w", base_interval="1d", cache=aux_cache)
    print(f"build 1w #1: {(time.perf_counter()-t)*1000:.0f} ms  keys={list(aux_cache.keys())} hit?= id match={getattr(w,'attrs',{}).get('_aux_panel_id')==id(panel)}", flush=True)
    t = time.perf_counter()
    w2 = get_or_build_aux_panel(panel, "1w", base_interval="1d", cache=aux_cache)
    print(f"build 1w #2 (should hit): {(time.perf_counter()-t)*1000:.0f} ms", flush=True)

    # eval_factor 完整路径，共享 cache
    for name, expr in [("EXPR1", EXPR1), ("EXPR2", EXPR2)]:
        t = time.perf_counter()
        try:
            out = eval_factor(expr, panel, aux_cache=aux_cache)
            msg = f"ok nonnull={int(out.notna().sum())}"
        except Exception as exc:  # noqa: BLE001
            msg = f"ERR {type(exc).__name__}: {str(exc)[:80]}"
        print(f"{name}: {(time.perf_counter()-t)*1000:.0f} ms  {msg}  aux_keys={list(aux_cache.keys())}", flush=True)

if __name__ == "__main__":
    main()