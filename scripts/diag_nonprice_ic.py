# -*- coding: utf-8 -*-
"""诊断：非价量因子逐个独立日度 IC（分年），判断负增量的来源。"""
import sys, time, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names
from alphaagent.factor.mining.eval.combo_increment import cs_zscore

rows = [json.loads(l) for l in Path("artifacts/alphaagent/harness_runs/2026-10-20_increment/nonprice_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
need = {"label_20d_close_to_close"}
for r in rows:
    try: need |= set(referenced_column_names(r["expr"]))
    except Exception: pass
need = {c for c in need if not str(c).startswith(("funda_","fsi_"))}
panel = load_panel_from_cne(start="2020-01-01", end="2024-12-31", include_fundamentals=False, keep_columns=tuple(sorted(need)))
lab = panel["label_20d_close_to_close"].astype("float32")
dates = pd.Series(panel.index.get_level_values(0), index=panel.index)

def daily_rank_ic(d, p, y):
    df = pd.DataFrame({"d": d, "p": p, "y": y}).dropna()
    if len(df) < 1000: return pd.Series(dtype="float64")
    df["pr"] = df.groupby("d")["p"].rank(); df["yr"] = df.groupby("d")["y"].rank()
    g = df.groupby("d"); n = g["pr"].size().astype(float)
    sp, sy = g["pr"].sum(), g["yr"].sum()
    spy = (df["pr"]*df["yr"]).groupby(df["d"]).sum()
    sp2 = (df["pr"]**2).groupby(df["d"]).sum(); sy2 = (df["yr"]**2).groupby(df["d"]).sum()
    corr = (spy - sp*sy/n) / np.sqrt((sp2-sp**2/n)*(sy2-sy**2/n))
    return corr.replace([np.inf,-np.inf], np.nan).dropna()

out=[]
for r in rows:
    try:
        v = eval_multi_line_factor(r["expr"], panel)
        if isinstance(v, pd.DataFrame): v = v.iloc[:,0] if v.shape[1]==1 else v.mean(axis=1)
        s = cs_zscore(pd.Series(v).astype("float32").reindex(panel.index))
        if s.notna().mean() < 0.5: continue
        ic = daily_rank_ic(dates.to_numpy(), s.to_numpy(), lab.to_numpy())
        yr = pd.Series(ic.values, index=pd.to_datetime(ic.index)).groupby(lambda x: x.year).mean()
        out.append({"name": r["name"], "ic24": float(yr.get(2024, np.nan)),
                    "ic2022": float(yr.get(2022, np.nan)), "ic2021": float(yr.get(2021, np.nan)),
                    "ic2020": float(yr.get(2020, np.nan)), "ic23": float(yr.get(2023, np.nan)),
                    "cov": float(s.notna().mean())})
    except Exception as e:
        print("FAIL", r["name"], e)
df = pd.DataFrame(out).sort_values("ic24", ascending=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
