# -*- coding: utf-8 -*-
"""Re-fetch rolling-window datasets whose daily window was lost to the wipe.

These datasets are fetch_semantics=snapshot, so `cne backfill` refuses them --
but their underlying endpoints are rolling windows that CAN serve recent
history (probed: news_headlines reaches ~15 days back). This script walks the
lost window date-by-date and writes through the regular staging/compact path.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import date, timedelta

import polars as pl

sys.path.insert(0, r"D:\Quant\quant_ui\CNEquity\src")

from cnequity.config import load_config  # noqa: E402
from cnequity.orchestrator.engine import JobEngine  # noqa: E402
from cnequity.orchestrator.manifest import Manifest  # noqa: E402
from cnequity.steps.http_common import write_fetched  # noqa: E402
from cnequity.adapters.eastmoney.rotation import (  # noqa: E402
    fetch_news_headlines,
    fetch_sector_fund_flow,
    fetch_hot_rank,
)

CONFIG_PATH = r"D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml"
# The wipe hit after 2026-08-28's data; re-fetch from 2026-08-29 onward.
WINDOW_START = date(2026, 8, 29)
WINDOW_END = date(2026, 9, 3)

FETCHERS = {
    # Only genuinely historical rolling windows. hot_rank / sector_fund_flow
    # stamp TODAY's live page onto the requested trade_date -- backfilling them
    # writes present-day values mislabelled as history (worse than empty), so
    # they are deliberately excluded and left to forward daily accumulation.
    "news_headlines": fetch_news_headlines,
}


def main() -> int:
    config = load_config(CONFIG_PATH)
    engine = JobEngine(config)
    manifest = Manifest(config.manifest_path)

    for dataset, fetcher in FETCHERS.items():
        print(f"=== {dataset}: {WINDOW_START} .. {WINDOW_END} ===", flush=True)
        # start_run mints the canonical run_id; use it everywhere (staging dir
        # name, batches, finish) so the manifest and the on-disk staging agree.
        run_id = manifest.start_run("backfill", {"dataset": dataset, "window": True})
        rows_total = 0
        for d in [WINDOW_START + timedelta(days=i) for i in range((WINDOW_END - WINDOW_START).days + 1)]:
            try:
                try:
                    df = fetcher(d, config=config)
                except TypeError:
                    df = fetcher(d)
            except Exception as exc:
                print(f"  {d}: FETCH ERROR {str(exc)[:100]}", flush=True)
                continue
            if df.is_empty():
                print(f"  {d}: 0 rows (skipped)", flush=True)
                continue
            try:
                write_fetched(config, run_id, dataset, df, source="eastmoney")
                rows_total += df.height
                print(f"  {d}: {df.height} rows staged", flush=True)
            except Exception as exc:
                print(f"  {d}: WRITE ERROR {str(exc)[:100]}", flush=True)
        # No per-run compact here: the lake mutation lock is contended by the
        # parallel rebuild queues. Staged rows are compacted later by the
        # final `cne compact` sweep once all queues drain.
        try:
            manifest.finish_run(run_id, "success" if rows_total else "warning")
        except Exception:
            pass
        print(f"=== {dataset}: total {rows_total} rows (run {run_id}) ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
