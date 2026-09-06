import sys, time
sys.path.insert(0, "D:/Quant/quant_ui_cmp")
import warnings
warnings.filterwarnings("ignore")

from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.dsl import eval_factor
import pandas as pd
import numpy as np

panel = load_panel_from_cne(start="2020-01-01", end="2022-12-31")
label_col = "label_1d_close_to_close"
s = eval_factor("vol20 = TS_STD($ret, 20)\nNEG(vol20)", panel)

from alphaagent.factor.align import align_series_to_panel
from alphaagent.factor.evaluation.context import EvaluationContext
from alphaagent.factor.evaluation.profile import default_evaluation_profiles
from alphaagent.factor.evaluation import plugins as P
from alphaagent.factor.metrics import cs_ic_summary, factor_skew_kurtosis, cross_sectional_lag1_pearson_autocorr, decile_mean_label

prof = default_evaluation_profiles()["train_screen"]
values = align_series_to_panel(s, panel)
factor = pd.Series(values, index=panel.index, name="probe", dtype=np.float32)
label = panel[label_col]

ctx = EvaluationContext(panel=panel, factor=factor, label=label, profile=prof, factor_name="probe", label_holding_days=1)
for item in prof.transforms:
    P.get_transform(str(item["plugin"]))(ctx, dict(item.get("params") or {}))

def t(name, fn):
    t0 = time.perf_counter()
    r = fn()
    print(f"  {name:34s} {time.perf_counter()-t0:.2f}s")
    return r

daily_ic = t("daily_ic (pearson numba)", ctx.daily_ic)
daily_rik = t("daily_rank_ic (spearman numba)", ctx.daily_rank_ic)
t("cs_ic_summary", lambda: cs_ic_summary(daily_ic, daily_rik, holding_days=1))
fv = factor.to_numpy(dtype=np.float64, copy=False)
t("skew_kurt", lambda: factor_skew_kurtosis(fv))
t("lag1_pearson_autocorr", lambda: cross_sectional_lag1_pearson_autocorr(factor, min_pairs=30))
t("decile_mean_label", lambda: decile_mean_label(fv, label.to_numpy(dtype=np.float64, copy=False), n_deciles=10))
t("day_slices x1", lambda: panel.index.get_level_values("datetime"))
from alphaagent.factor.metrics import _day_slices
t("_day_slices(index)", lambda: _day_slices(panel.index))
