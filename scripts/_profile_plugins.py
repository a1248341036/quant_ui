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
expr = "vol20 = TS_STD($ret, 20)\nNEG(vol20)"
s = eval_factor(expr, panel)

from alphaagent.factor.align import align_series_to_panel
from alphaagent.factor.evaluation.context import EvaluationContext
from alphaagent.factor.evaluation.profile import default_evaluation_profiles
from alphaagent.factor.evaluation import plugins as P

prof = default_evaluation_profiles()["train_screen"]
values = align_series_to_panel(s, panel)
factor = pd.Series(values, index=panel.index, name="probe", dtype=np.float32)


def fresh_ctx() -> EvaluationContext:
    ctx = EvaluationContext(
        panel=panel, factor=factor.copy(), label=panel[label_col], profile=prof,
        factor_name="probe", label_holding_days=1,
    )
    for item in prof.transforms:
        P.get_transform(str(item["plugin"]))(ctx, dict(item.get("params") or {}))
    return ctx


t0 = time.perf_counter()
ctx = fresh_ctx()
print(f"transforms(total)={time.perf_counter()-t0:.2f}s")

total = 0.0
for m in prof.metrics:
    name = m["plugin"]; params = m.get("params", {})
    fn = P.get_metric(name)
    ctx = fresh_ctx()
    t0 = time.perf_counter()
    fn(ctx, dict(params))
    dt = time.perf_counter() - t0
    total += dt
    print(f"  {name:26s} {dt:.2f}s")
print(f"metrics sum={total:.2f}s")
