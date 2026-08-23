# ADR 0005: Source routing vs source switching

- Status: Accepted
- Date: 2026-07-27
- Relates to: [ADR-0003](0003-canonical-curated-with-source-snapshots.md)

## Context

ADR-0003 says curated holds one row per primary key and that **source
switching is never automatic**. In practice the lake already routes disjoint
key sets to different adapters:

- Beijing Exchange symbols cannot be served by TDX/mootdx, so `daily_bars`
  routes them through Sina into curated (`source=sina`).
- When TDX tip batches fail, EastMoney push2 **clist** can fill the missing
  tip keys in minutes; per-symbol kline cannot (hours for the full market).

Without an explicit distinction, "never automatic" is easy to misread as
"only the configured primary may ever write curated," which is already false
and would block tip availability.

## Decision

Distinguish two operations:

1. **Routing** — assign **disjoint** primary-key sets to different adapters
   in the same ingestion. Each PK still has exactly one curated candidate.
   Examples: BJ → sina; tip-day keys missing after TDX → EastMoney clist
   gap-fill (`source=eastmoney`). Routing may be automatic.
2. **Switching** — change which source owns a PK that **already has** a
   canonical row (or prefer backup over a successful primary write).
   Switching is never automatic; a human (or an explicit repair command)
   decides. Compact must not let a later backup overwrite a same-run primary
   row for the same PK — gap-fill writes only keys absent from staged primary
   output.

Backup **snapshots** and `source_diff` remain for cross-audit even when
routing also stages backup rows for missing tip keys.

## Consequences

- Tip availability no longer depends on TDX succeeding for every batch.
- Provenance stays honest: routed rows carry the adapter that produced them.
- Callers must filter gap-fill to missing keys because compact uses
  `keep="last"` by `fetched_at`.
- Multi-day / history gaps still use per-symbol EastMoney kline (or other
  history adapters); clist is tip-snapshot only and stamps `trade_date`
  from the run.

## Alternatives considered

- Promote EastMoney or ths to daily primary for all SH/SZ: unnecessary when
  TDX is healthy; ths is too slow for a full-market tip.
- Keep failover as snapshot-only: preserves purity but leaves curated tip
  holes whenever TDX flakes (the current production failure mode).
- Silently reinterpret ADR-0003 without a new record: hides the BJ/sina
  precedent and invites future regressions.
