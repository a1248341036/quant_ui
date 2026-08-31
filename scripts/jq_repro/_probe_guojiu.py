# -*- coding: utf-8 -*-
"""国九策略选股链路逐环节探针: 定位 get_fundamentals/filter_stocks 哪一环过滤光了."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies" / "event"))
sys.path.insert(0, str(ROOT / "scripts" / "jq_repro"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from _runtime import JQContext  # noqa: E402
from core.event_engine.jq.runtime import JQRuntime  # noqa: E402
from core.event_engine.jq.query import query, valuation, income  # noqa: E402

ctx = JQContext(end="2024-12-31", lookback_days=865)
rt = JQRuntime("def initialize(context):\n    pass\n", ctx, 1_000_000.0)

rt.context.current_dt = pd.Timestamp("2024-06-05")
rt.context.previous_date = pd.Timestamp("2024-06-04")

pool = list(ctx.codes)
print("pool:", len(pool))

# ---- snapshot 检查 ----
snap = rt.get_snapshot(pd.Timestamp("2024-06-04"))
print("snapshot 行数:", len(snap))
for col in ("market_cap", "np_parent_company_owners", "net_profit",
            "operating_revenue", "limit_price", "open_raw"):
    if col in snap.columns:
        print(f"  {col}: 非NaN {snap[col].notna().mean():.2%}")
print("  market_cap.between(10,100):",
      int(snap["market_cap"].between(10, 100).sum()))
print("  np_parent>0:", int((snap["np_parent_company_owners"] > 0).sum()))
print("  net_profit>0:", int((snap["net_profit"] > 0).sum()))
print("  revenue>1e8:", int((snap["operating_revenue"] > 1e8).sum()))

# ---- filter_stocks 等价环节 ----
cur = rt.get_current_data()
lp = rt.history(1, unit="1m", field="close", security_list=pool[:500])
import datetime
cnt = {"paused": 0, "st": 0, "退": 0, "prefix": 0, "涨停": 0, "跌停": 0,
       "次新": 0, "ok": 0, "lp_empty": 0, "limit_nan": 0}
ok_list = []
for c in pool[:500]:
    cd = cur[c]
    if cd.paused:
        cnt["paused"] += 1
        continue
    if cd.is_st:
        cnt["st"] += 1
        continue
    if "退" in cd.name:
        cnt["退"] += 1
        continue
    if c.startswith(("30", "68", "8", "4")):
        cnt["prefix"] += 1
        continue
    seq = lp.get(c, [])
    if not seq:
        cnt["lp_empty"] += 1
        continue
    hl = cd.high_limit
    ll = cd.low_limit
    import numpy as np
    if not (pd.notna(hl) and seq[-1] < hl):
        cnt["涨停" if pd.notna(hl) else "limit_nan"] += 1
        continue
    if not (pd.notna(ll) and seq[-1] > ll):
        cnt["跌停"] += 1
        continue
    sd = ctx.list_date_map.get(c)
    if sd is not None and (rt.context.previous_date - sd) < datetime.timedelta(days=375):
        cnt["次新"] += 1
        continue
    cnt["ok"] += 1
    ok_list.append(c)
print("filter_stocks(前500):", cnt)

# ---- fundamentals 查询(不 in_ initial_list, 直接全池) ----
q = query(valuation.code, valuation.market_cap,
          income.np_parent_company_owners, income.net_profit,
          income.operating_revenue).filter(
    valuation.market_cap.between(10, 100),
    income.np_parent_company_owners > 0,
    income.net_profit > 0,
    income.operating_revenue > 1e8
).order_by(valuation.market_cap.asc()).limit(12)
df = rt.get_fundamentals(q)
print("get_fundamentals 全池:", len(df))
if len(df):
    print(df.head(5).to_string())

# ---- in_ initial_list 版 ----
q2 = query(valuation.code).filter(valuation.code.in_(ok_list))
df2 = rt.get_fundamentals(q2)
print("code.in_(ok_list):", len(df2))
