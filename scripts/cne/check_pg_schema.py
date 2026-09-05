# -*- coding: utf-8 -*-
"""Compare pg_parquet schemas vs curated target schemas for L3 datasets."""
import sys

sys.path.insert(0, r"D:\Quant\quant_ui\CNEquity\src")
import polars as pl  # noqa: E402

from cnequity.domain.schemas import DATASET_SCHEMAS  # noqa: E402

PAIRS = [
    ("balancesheet", r"D:\Quant\quant_ui\data\pg_parquet\balancesheet.parquet"),
    ("income", r"D:\Quant\quant_ui\data\pg_parquet\income.parquet"),
    ("cashflow", r"D:\Quant\quant_ui\data\pg_parquet\cashflow.parquet"),
    ("fina_indicator", r"D:\Quant\quant_ui\data\pg_parquet\fina_indicator.parquet"),
    ("forecast", r"D:\Quant\quant_ui\data\pg_parquet\forecast.parquet"),
    ("express", r"D:\Quant\quant_ui\data\pg_parquet\express.parquet"),
    ("top_holders", None),  # not in pg_parquet
    ("share_structure", None),
    ("financial_statement_items", None),
    ("earnings_disclosure_schedule", None),
]

for ds, path in PAIRS:
    target = DATASET_SCHEMAS.get(ds)
    tcols = {k: str(v) for k, v in target.items()} if target else {}
    print(f"--- {ds} ---")
    if path is None:
        print("  no pg_parquet source")
        print("  target cols:", sorted(tcols))
        continue
    sch = pl.scan_parquet(path).collect_schema()
    scols = {k: str(v) for k, v in sch.items()}
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    print(f"  source rows={n:,} cols={len(scols)}")
    missing_in_src = [c for c in tcols if c not in scols]
    extra_in_src = [c for c in scols if c not in tcols]
    type_mismatch = {c: (scols[c], tcols[c]) for c in tcols if c in scols and scols[c] != tcols[c]}
    if missing_in_src:
        print("  MISSING in source:", missing_in_src)
    if extra_in_src:
        print("  extra in source:", extra_in_src)
    if type_mismatch:
        print("  type mismatch:", type_mismatch)
    if not (missing_in_src or extra_in_src or type_mismatch):
        print("  schemas IDENTICAL")
