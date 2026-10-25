# Freshness gate: three kinds of "behind"

`cne status --datasets` judges every dataset against the last trading day. Three
distinct failure modes need three distinct answers — collapsing them into one
tolerance is how a whole session disappears silently.

## 1. Publication lag (`fresh` / `STALE`)

The vendor simply has not published yet: margin trading is T+1, quarterly
disclosures wait for the report season. `is_stale` covers this with
`DatasetSpec.max_staleness_days` (default 1 day; margin trading 2; quarterly
datasets 100).

## 2. A session nobody fetched (`STALE(missed)`)

If the 21:30 job does not fire (machine asleep, task not triggered) the next
trigger may be a weekend, which `cne run daily` skips as a non-trading day —
including `--stale-only`, because it anchors on *today*. The lake then sits
exactly one trading day behind, inside the tolerance above, and every check
calls it fresh.

A dataset is **missed** when its watermark is behind the anchor **and** the
manifest recorded no successful batch (`success` / `warning`) for that session.
Matched by `ingestion_runs.metadata_json.trade_date`, not by batch timestamps —
those are UTC, so a 21:30 Asia/Shanghai run lands on the previous UTC day.

```
cne status --datasets            # exits non-zero, rows read STALE(missed)
cne run catchup --trade-date YYYY-MM-DD   # gate datasets + market_breadth + compact
cne run daily --stale-only --trade-date YYYY-MM-DD
```

## 3. Watermark ahead of the lake (`STALE(missed)`)

A run advances the watermark when the step finishes; compact lands the staging
partition afterwards. A crash in between leaves a watermark that says the
session landed while curated has no partition for it (2026-10-25:
`daily_bars` watermark 10-23, curated still 10-22). The partition check
(`*.parquet` under `{partition_col}={value}`) catches this; external adapters
(`daily_bars`, `fund_nav`, `index_bars_external`, …) report coverage through
their registry instead, so the manifest signal is used for them.

## What the scheduled scripts do

`scripts/cne/run_cne_daily.ps1` (21:30) probes `status --datasets`; when the
anchor is older than today it runs `catchup --trade-date <anchor>` first and
pins `--stale-only` to the same date. Stage 8.5 then refreshes
`audit --full`, `stats refresh` and `sources` so the health snapshot, the lake
stats and the source-health report cannot go stale unnoticed.

`scripts/cne/run_cne_stale_retry.ps1` (23:00) does the same probe, then
catchup + stale-only. Both tasks run with `WakeToRun` enabled.

## Snapshot datasets cannot be replayed

`valuation_metrics`, `fund_flow`, `sector_*`, `hot_rank`, `sentiment_*`,
`analyst_consensus`, `index_constituents`, `industry_members`, `trading_status`
are live snapshots stamped with the run day. A missed day is only recoverable
through `cne backfill <dataset> --start … --end …` when the spec declares a
`backfill_source` (valuation_metrics → baostock, fund_flow → tushare,
sector_bars → ths). The rest are permanent holes: decide explicitly whether to
accept them or refill from another pipeline.
