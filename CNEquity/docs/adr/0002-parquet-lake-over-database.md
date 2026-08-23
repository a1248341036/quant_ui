# ADR 0002: Parquet data lake as the delivery format (not a database)

- Status: Accepted
- Date: 2026-06-28

## Context

Research consumers want reproducible, portable A-share data without operating a
database server. Datasets are append-mostly, time-partitioned, and queried
analytically (scans, joins, aggregations).

## Decision

Deliver a partitioned **Parquet** data lake (zstd) as the primary artifact,
organized in medallion-style layers: `staging` (per-run raw landing),
`curated` (one canonical row per primary key), `derived` (computed datasets),
and `meta` (manifest, quality, snapshots). Expose an **optional** DuckDB view
layer for SQL; DuckDB is a query convenience, not the source of truth.

## Consequences

- Zero-ops delivery: files are the product; DuckDB/Polars query them directly.
- Easy backup/transfer; columnar + partition pruning gives good scan perf.
- We must implement compaction and PK de-duplication ourselves (in `storage/`).

## Alternatives considered

- DuckDB-only storage (as in some reference projects): single-file DB couples
  storage to the engine and complicates parallel writes and portability.
- Postgres/ClickHouse: operational burden contrary to the product goal.
