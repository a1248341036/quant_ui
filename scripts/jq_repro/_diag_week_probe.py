# -*- coding: utf-8 -*-
"""周触发日探针: 2025 年第一周 run_weekly(2) 在哪天触发? 1 月订单去向?"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    from strategies.event._runtime import JQContext
    from _runtime import JQContext as _JR  # noqa: F401
    ctx = JQContext(end="2025-02-15", lookback_days=800)
    dates = ctx.tables.dates
    iso = dates.isocalendar()
    wk = (iso["year"].to_numpy().astype("int64") * 100
          + iso["week"].to_numpy().astype("int64"))
    s = pd.Series(1, index=dates)
    wn = s.groupby(wk).cumsum()
    lo = dates.searchsorted(pd.Timestamp("2025-01-20"))
    hi = dates.searchsorted(pd.Timestamp("2025-02-13"))
    for d in dates[lo:hi]:
        print(d.date(), "week_n =", int(wn.loc[d]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
