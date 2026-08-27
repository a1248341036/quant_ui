"""低换手/低波动 在全A池的 walk-forward 样本外验证。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.data import load_universe
from core.store import PANEL_FILE
from core.walkforward import walk_forward_factor

codes = set(load_universe()["code"].astype(str).str.zfill(6))
pf = pd.read_parquet(PANEL_FILE, columns=["code"])
codes = sorted(codes & set(pf["code"].astype(str).str.zfill(6)))

df = pd.read_parquet(PANEL_FILE,
                     columns=["date", "open", "high", "low", "close",
                              "turnover", "amount", "code", "turn20",
                              "am20", "volume"],
                     filters=[("code", "in", codes)])
df["code"] = df["code"].astype(str).str.zfill(6)
df["date"] = pd.to_datetime(df["date"])
df = df[(df["date"] >= "2018-01-01")].copy()

out = ROOT / "pytmp"
for factor, asc in [("turn20", True), ("vol20", True)]:
    rows = walk_forward_factor(
        df, codes, factor=factor, ascending=asc,
        start="2019-01-01", end="2025-12-31",
        capital=100000, top_n=5, freq="monthly",
        n_folds=4, amount_q=0.2,
    )
    rows.to_csv(out / f"wf_{factor}.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== {factor} ascending={asc} 全A池 Top5 月频 10万 ===")
    print(rows.to_string(index=False))
    print(f"胜率(窗口收益>0): {(rows['total'] > 0).mean():.0%}  "
          f"最差窗口: {rows['total'].min():+.1%}")
