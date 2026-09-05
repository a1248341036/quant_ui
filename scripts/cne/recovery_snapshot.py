# -*- coding: utf-8 -*-
"""Dataset-level recovery snapshot: registry tier/layer x freshness."""
import subprocess
import sys

sys.path.insert(0, r"D:\Quant\quant_ui\CNEquity\src")
import polars as pl  # noqa: E402

from cnequity.config import load_config  # noqa: E402
from cnequity.query.reader import list_datasets  # noqa: E402
from cnequity.domain.datasets import is_dataset_enabled, is_stale, DATASETS  # noqa: E402
from cnequity.cli.main import _last_trading_day, shanghai_today  # noqa: E402

cfg = load_config(r"D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml")
anchor = _last_trading_day(cfg, shanghai_today())
df = list_datasets(config=cfg)

tiers = {}
for name in DATASETS:
    spec = DATASETS[name]
    tiers[name] = (spec.tier, spec.layer, spec.fetch_semantics)

rows = []
for r in df.iter_rows(named=True):
    name = r["dataset"]
    if not r["has_data"]:
        fresh = "empty"
    elif not is_dataset_enabled(name, cfg):
        fresh = "n/a"
    elif not r["watermarked"]:
        fresh = "n/a"
    else:
        mark = r["watermark"] or r["coverage_end"]
        fresh = "STALE" if is_stale(name, mark, anchor) else "fresh"
    tier, layer, sem = tiers.get(name, ("?", "?", "?"))
    rows.append({
        "tier": tier, "layer": layer, "dataset": name, "freshness": fresh,
        "coverage": f"{r['coverage_start']}..{r['coverage_end']}" if r["has_data"] else "-",
        "sem": sem,
    })

out = pl.DataFrame(rows).sort(["tier", "layer", "dataset"])
with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=40, tbl_width_chars=200):
    print(f"anchor(last trading day) = {anchor}")
    print(out)

print()
for t in sorted({r["tier"] for r in rows}):
    sub = [r for r in rows if r["tier"] == t]
    ok = sum(1 for r in sub if r["freshness"] in ("fresh", "n/a"))
    print(f"tier {t}: {ok}/{len(sub)} fresh-or-n/a  (empty/STALE: {len(sub)-ok})")
