# ADR 0003: Canonical curated rows with separate multi-source snapshots

- Status: Accepted
- Date: 2026-06-28

## Context

The same field (e.g. a close price or dividend) can be served by multiple
sources (tdx_protocol, eastmoney, sina, ...) with subtle discrepancies. If
backup sources silently overwrite primary data, lineage and reproducibility
break.

## Decision

`curated/` holds exactly **one canonical row per primary key**, with `source`,
`data_version`, `fetched_at` as **columns** (not partition keys). Backup-source
data lands in `meta/source_snapshots/{dataset}/source=.../data_version=.../`.
The `audit` step compares primary vs snapshots and emits `source_diffs`.
**Source switching is never automatic** — a human decides.

**Routing vs switching:** “Never automatic” means **switching** the preferred
source for a primary key that already has a canonical row. It does **not**
forbid **routing** disjoint key sets to different adapters in one run (e.g.
BJ → sina, tip-day TDX gaps → EastMoney clist). See
[ADR-0005](0005-source-routing-vs-switching.md). Backup snapshots remain for
cross-audit even when routing stages missing tip keys into curated.

**Per-dataset canonical source:** The canonical `source` column reflects whichever
adapter owns that dataset for the given ingestion mode (e.g. `corporate_actions`
uses `eastmoney` for daily ex-date scans and `tdx_protocol` for symbol backfill).
Backup snapshots use the alternate source for cross-audit only.

## Consequences

- Reproducible, auditable canonical dataset; provenance always present.
- Cross-source disagreements are observable instead of hidden.
- Requires snapshot storage + diff tooling.

## Alternatives considered

- Last-writer-wins across sources: hides disagreements, non-reproducible.
- Storing every source in curated with a `source` partition: explodes
  partitions and pushes conflict resolution onto every consumer.
