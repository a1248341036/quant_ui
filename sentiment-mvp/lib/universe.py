# -*- coding: utf-8 -*-
"""构建标的池：按近半年日均成交额取 top_n，落盘 data/universe.csv。"""

from pathlib import Path

import pandas as pd


def _norm(df):
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def build_universe(cfg: dict) -> pd.DataFrame:
    out = Path(cfg["universe"]["output"])
    if out.exists():
        df = pd.read_csv(out, dtype={"code": str})
        return _norm(df)

    panel = pd.read_parquet(cfg["universe"]["panel"])
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    recent = panel[panel["date"] >= (panel["date"].max() - pd.Timedelta(days=180))].copy()
    recent["amount"] = pd.to_numeric(recent["amount"], errors="coerce")
    rank = recent.groupby("code")["amount"].mean().sort_values(ascending=False)

    name_map = {}
    uni = pd.read_csv(cfg["universe"]["universe_csv"], dtype={"code": str})
    uni = _norm(uni)
    name_map.update(dict(zip(uni["code"], uni["name"])))
    tech = pd.read_csv(cfg["universe"]["tech_csv"], dtype={"code": str})
    tech = _norm(tech)
    name_map.update(dict(zip(tech["code"], tech["name"])))

    top_n = int(cfg["universe"]["top_n"])
    chosen = [c for c in rank.index if c in name_map][:top_n]
    df = pd.DataFrame({"code": chosen, "name": [name_map[c] for c in chosen]})
    df["amount_rank"] = df["code"].map(lambda c: int(rank.index.get_loc(c)) + 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
