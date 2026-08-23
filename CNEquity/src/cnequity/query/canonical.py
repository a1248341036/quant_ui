"""Canonical row helpers shared by non-API lake consumers."""

from __future__ import annotations

import polars as pl

from cnequity.domain.schemas import PRIMARY_KEYS


def dedupe_by_primary_key(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Keep one row per registered PK, preferring the freshest provenance.

    Validated lake rows always carry ``fetched_at``. The no-provenance fallback
    still collapses duplicate keys so a malformed legacy fragment cannot
    multiply a quality join; schema checks remain responsible for reporting
    that the fragment is incomplete.
    """
    primary_key = PRIMARY_KEYS.get(dataset, [])
    if df.is_empty() or not primary_key or any(k not in df.columns for k in primary_key):
        return df
    if "fetched_at" in df.columns:
        df = df.sort("fetched_at")
    return df.unique(subset=primary_key, keep="last", maintain_order=True)


def dedupe_lazy_by_primary_key(lf: pl.LazyFrame, dataset: str) -> pl.LazyFrame:
    """Lazy equivalent of :func:`dedupe_by_primary_key`."""
    primary_key = PRIMARY_KEYS.get(dataset, [])
    columns = set(lf.collect_schema().names())
    if not primary_key or any(k not in columns for k in primary_key):
        return lf
    if "fetched_at" in columns:
        lf = lf.sort("fetched_at")
    return lf.unique(subset=primary_key, keep="last", maintain_order=True)
