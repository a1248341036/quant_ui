# -*- coding: utf-8 -*-
"""冒烟计时：27 个非价量因子逐个求值（2024H2 小窗口），抓慢表达式。"""
import sys, time, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names

rows = [json.loads(l) for l in Path("artifacts/alphaagent/harness_runs/2026-10-20_increment/nonprice_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
need = {"label_20d_close_to_close"}
for r in rows:
    try: need |= set(referenced_column_names(r["expr"]))
    except Exception: pass
need = {c for c in need if not str(c).startswith(("funda_","fsi_"))}
t0=time.perf_counter()
panel = load_panel_from_cne(start="2024-06-01", end="2024-12-31", include_fundamentals=False, keep_columns=tuple(sorted(need)))
print(f"panel {panel.shape} load={time.perf_counter()-t0:.0f}s")
bad=[]
for r in rows:
    t=time.perf_counter()
    try:
        v = eval_multi_line_factor(r["expr"], panel)
        if isinstance(v, pd.DataFrame): v = v.iloc[:,0] if v.shape[1]==1 else v.mean(axis=1)
        s = pd.Series(v).astype("float32").reindex(panel.index)
        cov = float(s.notna().mean())
        dt = time.perf_counter()-t
        flag = "SLOW!" if dt>20 else ("thin" if cov<0.5 else "ok")
        if flag!="ok": bad.append(r["name"])
        print(f"{flag:5s} {dt:6.1f}s cov={cov:.2f} {r['name']}")
    except Exception as e:
        bad.append(r["name"])
        print(f"FAIL  {time.perf_counter()-t:6.1f}s {r['name']}: {type(e).__name__} {str(e)[:80]}")
print(f"\n需剔除/关注 {len(bad)}: {bad}")
