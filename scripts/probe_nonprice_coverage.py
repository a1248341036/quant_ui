# -*- coding: utf-8 -*-
"""快速探查：非价量域列的可用性与覆盖率（2020-2024）。"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

CANDS = """pred_surprise pred_change_mid pred_net_profit_mid pred_type pred_direction
ann_flag_forecast ann_flag_buyback ann_flag_lawsuit ann_cnt_20d ann_days_since
rc_rating rc_target_price rc_eps_y rc_cnt_90d
unlock_ratio_next unlock_ratio_past180 unlock_days_next
inst_social_security_ratio inst_qfii_ratio inst_fund_ratio inst_insurance_ratio inst_broker_ratio
th_top10_pct th_inst_pct holder_count_chg_pct
ff_super_net ff_large_net ff_medium_net ff_small_net
mgn_short_balance bt_premium_rate bt_amount
div_ratio ds_days""".split()

from alphaagent.data.adapters.cnequity import load_panel_from_cne
t0 = time.perf_counter()
panel = load_panel_from_cne(start="2020-01-01", end="2024-12-31",
                            include_fundamentals=False, keep_columns=tuple(CANDS))
print(f"panel {panel.shape} load={time.perf_counter()-t0:.0f}s")
have = [c for c in CANDS if c in panel.columns]
miss = [c for c in CANDS if c not in panel.columns]
print(f"有 {len(have)} / 缺 {len(miss)}")
if miss: print("缺:", miss)
rows=[]
for c in have:
    s = panel[c]
    rows.append({"col": c, "notna": float(s.notna().mean()),
                 "nonzero": float((s.fillna(0)!=0).mean()),
                 "nuniq": int(s.nunique()), "std": float(s.std() or 0)})
df = pd.DataFrame(rows).sort_values("notna", ascending=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
