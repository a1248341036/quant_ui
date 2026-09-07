"""Compact eligibility: skip datasets whose staged data is not settled yet.

Only batches still being written (``running``) or presumed dead but
unreconciled (``stale``) block compaction. Terminal ``warning``/``failed``
batches hold complete staging files and compact idempotently (PK dedupe);
blocking on them stranded valid subsets in staging forever — the watermark
safety net against partial coverage lives in the watermark computation
(last-contiguous-dense walk-back), not here.
"""

from __future__ import annotations

from cnequity.orchestrator.manifest import Manifest


def refresh_batch_liveness(
    manifest: Manifest,
    run_id: str,
    *,
    stale_after_seconds: float,
) -> int:
    """Promote heartbeat-expired running batches to stale before compact gating."""
    return manifest.promote_running_to_stale(run_id, stale_after_seconds=stale_after_seconds)


def datasets_with_blocking_batches(manifest: Manifest, run_id: str) -> frozenset[str]:
    """Return dataset names that still have running/stale batches for *run_id*."""
    counts = manifest.blocking_batch_counts_by_dataset(run_id)
    return frozenset(ds for ds, n in counts.items() if n > 0)


def compact_allowed(
    manifest: Manifest,
    run_id: str,
    dataset: str,
    *,
    stale_after_seconds: float | None = None,
) -> tuple[bool, int]:
    """Return (allowed, blocking_batch_count) for compacting *dataset* in *run_id*."""
    if stale_after_seconds is not None:
        refresh_batch_liveness(manifest, run_id, stale_after_seconds=stale_after_seconds)
    blocking = manifest.blocking_batch_counts_by_dataset(run_id).get(dataset, 0)
    return blocking == 0, blocking
