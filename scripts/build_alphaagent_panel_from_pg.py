"""Build an AlphaAgent-compatible panel from CNEquant_dataset daily bars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from alphaagent.data.panel import build_panel_from_hq, save_panel


DEFAULT_INPUT = PROJECT_ROOT / "data" / "quant_dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "alphaagent" / "panel_1d.parquet"


def _load_cne_daily(root: Path) -> pd.DataFrame:
    """读取 quant_dataset/YYYY/YYYY/day/stock_daily.parquet 并纵向合并。"""
    import duckdb

    glob = str(root / "*" / "*" / "day" / "stock_daily.parquet")
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT * FROM read_parquet('{glob}') ORDER BY trade_date, ts_code"
        ).df()
    finally:
        con.close()


def build(input_path: Path, output_path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    if input_path.is_dir():
        raw = _load_cne_daily(input_path)
    else:
        raw = pd.read_parquet(input_path)
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"stock_daily.parquet missing columns: {missing}")

    raw["datetime"] = pd.to_datetime(raw["trade_date"])
    raw["instrument"] = raw["ts_code"].astype(str)
    hq = pd.DataFrame(index=pd.MultiIndex.from_arrays(
        [raw["datetime"], raw["instrument"]], names=["datetime", "instrument"]
    ))
    rename = {
        "vol": "volume", "adj_factor": "adjfactor",
        "circ_mv": "float_cap", "total_mv": "tot_cap",
    }
    for source, target in rename.items():
        if source in raw:
            hq[target] = raw[source].to_numpy()
    for col in ("open", "high", "low", "close", "amount"):
        hq[col] = raw[col].to_numpy()
    if "float_cap" not in hq:
        hq["float_cap"] = raw.get("circ_mv", 0.0).to_numpy() * 10000.0
    if "tot_cap" not in hq:
        hq["tot_cap"] = raw.get("total_mv", 0.0).to_numpy() * 10000.0
    for col in (
        "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb",
        "ps", "ps_ttm", "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share",
    ):
        hq[col] = raw[col].to_numpy() if col in raw else float("nan")
    hq["is_trade"] = (hq["volume"].fillna(0) > 0).astype("int8")
    hq["is_st"] = 0
    hq["not_st"] = 1
    hq = hq.sort_index()
    panel = build_panel_from_hq(
        hq, start=start, end=end, universe_mask=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_panel(panel, output_path)
    print(f"saved {output_path} rows={len(panel)} columns={len(panel.columns)}")
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AlphaAgent panel from quant_ui PG parquet")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    build(args.input, args.output, args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
