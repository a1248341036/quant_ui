# -*- coding: utf-8 -*-
"""残差 IC（修正版）：逐日把门控因子对低波做秩回归，残差的日度 rank IC。"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.dsl.eval import eval_multi_line_factor
from alphaagent.factor.mining.eval.combo_increment import cs_zscore

GATES = {
  "np_inst_lv_gate": "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\ng = CS_BUCKET(CS_ZSCORE($inst_fund_ratio), 5)\nCS_GROUP_RANK(v, g)",
  "np_pred_lv_gate": "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\ng = CS_BUCKET(CS_ZSCORE(TS_MEAN(CS_WINSORIZE($pred_surprise,0.01,0.99), 20)), 5)\nCS_GROUP_RANK(v, g)",
  "np_blend_pred":   "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\np = CS_ZSCORE(TS_MEAN(CS_WINSORIZE($pred_surprise,0.01,0.99), 20))\nCS_ZSCORE(EMA(ADD(v, p), 8))",
}
panel = load_panel_from_cne(start="2024-01-01", end="2024-12-31", include_fundamentals=False,
    keep_columns=("label_20d_close_to_close","ret","inst_fund_ratio","pred_surprise"))
lab = panel["label_20d_close_to_close"].astype("float32")
lv = cs_zscore(eval_multi_line_factor("CS_ZSCORE(NEG(TS_STD($ret, 60)))", panel))
lv = pd.Series(lv).astype("float32") if not isinstance(lv, pd.DataFrame) else lv.iloc[:,0]
base = pd.DataFrame({"lv": lv, "y": lab})

print(f"{'factor':18s} {'residIC2024':>11s}")
for name, expr in GATES.items():
    v = eval_multi_line_factor(expr, panel)
    v = pd.Series(v).astype("float32") if not isinstance(v, pd.DataFrame) else v.iloc[:,0]
    df = base.copy(); df["v"] = cs_zscore(v)
    df = df.dropna()
    df["d"] = df.index.get_level_values(0)
    def resid_r(g):
        a = g["lv"].rank().to_numpy(); b = g["v"].rank().to_numpy()
        A = np.c_[a, np.ones(len(a))]
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        return pd.Series(b - A @ beta, index=g.index)
    df["res"] = df.groupby("d", group_keys=False).apply(resid_r)
    def ic_of(col):
        dd = df["d"]; 
        pr = df.groupby(dd)[col].rank(); yr = df.groupby(dd)["y"].rank()
        j = pd.DataFrame({"d": dd, "pr": pr, "yr": yr})
        g2 = j.groupby("d"); n = g2["pr"].size().astype(float)
        sp, sy = g2["pr"].sum(), g2["yr"].sum()
        spy = (j["pr"]*j["yr"]).groupby(j["d"]).sum()
        sp2 = (j["pr"]**2).groupby(j["d"]).sum(); sy2 = (j["yr"]**2).groupby(j["d"]).sum()
        corr = (spy - sp*sy/n) / np.sqrt((sp2-sp**2/n)*(sy2-sy**2/n))
        return float(corr.mean())
    print(f"{name:18s} {ic_of('res'):+11.4f}")
