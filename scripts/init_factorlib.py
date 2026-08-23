#!/usr/bin/env python3
"""初始化股票日频因子库：panel → canonical index + manifest."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.core.paths import FACTORZOO_DIR  # noqa: E402
from alphaagent.data.adapters.cnequity import CNE_SOURCE, is_cne_source, load_panel_from_cne  # noqa: E402
from alphaagent.data.panel import load_panel  # noqa: E402
from alphaagent.factor.zoo import init_library  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化 factorzoo 因子库")
    parser.add_argument("--output", type=Path, default=FACTORZOO_DIR, help="因子库根目录")
    parser.add_argument("--panel", default=CNE_SOURCE, help="panel 路径或 cne://（CNE 数据源）")
    parser.add_argument("--n-sample-rows", type=int, default=200_000)
    parser.add_argument("--max-factors", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    if is_cne_source(args.panel):
        panel = load_panel_from_cne(universe_mask=False)
        resolved_panel_path = Path(CNE_SOURCE)
    else:
        panel_path = Path(args.panel)
        panel = load_panel(panel_path)
        resolved_panel_path = panel_path.resolve()
    paths, manifest, index = init_library(
        args.output,
        panel=panel,
        panel_path=resolved_panel_path,
        n_sample_rows=min(args.n_sample_rows, len(panel)),
        max_factors=args.max_factors,
        sample_seed=args.sample_seed,
    )
    elapsed = time.perf_counter() - t0

    print(f"因子库: {paths.root}")
    print(f"panel: {args.panel} n_rows={manifest.n_rows}")
    print(f"n_sample_rows={manifest.n_sample_rows} max_factors={manifest.max_factors}")
    print(f"index_hash={manifest.index_hash}")
    print(f"shards={len(index.shards)}")
    print(f"耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
