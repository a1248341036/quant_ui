from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import polars as pl

from cnequity.domain.datasets import granularity_for_dataset
from cnequity.domain.partitions import Granularity
from cnequity.domain.schemas import PRIMARY_KEYS, validate_dataframe
from cnequity.storage.atomic import write_parquet_atomic

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class StagingWriter:
    def __init__(self, staging_root: Path | str):
        # Process-pool workers may pass a str path across the boundary.
        self.staging_root = Path(staging_root)

    def write_batch(
        self,
        dataset: str,
        run_id: str,
        batch_id: str,
        df: pl.DataFrame,
    ) -> Path:
        # Compactable external datasets (e.g. daily_bars via tushare_wide)
        # store rows in the adapter's native format (33-column wide table),
        # not CNE's narrow schema.  validate_dataframe would reject them.
        from cnequity.domain.datasets import DATASETS

        spec = DATASETS.get(dataset)
        if spec is not None and spec.compactable and spec.layer == "external":
            pass  # skip schema validation — adapter owns the format
        else:
            df = validate_dataframe(df, dataset)
        out_dir = self.staging_root / dataset / f"run_id={run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"part-{batch_id}.parquet"
        write_parquet_atomic(path, df, compression="zstd")
        return path

    def list_run_files(self, dataset: str, run_id: str) -> list[Path]:
        run_dir = self.staging_root / dataset / f"run_id={run_id}"
        if not run_dir.exists():
            return []
        return sorted(run_dir.rglob("*.parquet"))


class CuratedWriter:
    def __init__(self, curated_root: Path):
        self.curated_root = curated_root

    def partition_path(self, dataset: str, partition_col: str, partition_value: str) -> Path:
        return self.curated_root / dataset / f"{partition_col}={partition_value}"

    def write_partition(
        self,
        dataset: str,
        partition_col: str,
        partition_value: str,
        df: pl.DataFrame,
        part_name: str = "part-0.parquet",
    ) -> Path:
        out_dir = self.partition_path(dataset, partition_col, partition_value)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / part_name
        write_parquet_atomic(path, df, compression="zstd")
        # A previous run may have left fragment files (for example after a
        # staging write used ``part-<batch>.parquet``), possibly under a
        # temporary subdirectory. Keeping them beside the canonical file makes
        # every later recursive scan read the same rows twice, even though the
        # merge above already deduplicated them. Remove only old parquet
        # descendants, and only after the atomic replacement has succeeded so
        # a failed write leaves the old partition readable.
        for stale in out_dir.rglob("*.parquet"):
            if stale != path:
                stale.unlink()
        return path


def _partition_values(df: pl.DataFrame, partition_col: str, granularity: Granularity) -> pl.Series:
    """Directory value per row for *partition_col*.

    Date columns map through the dataset's period (day/month/year); non-date
    keys like ``report_period`` ("2024Q1") are already period labels and are
    used verbatim.
    """
    col = df.get_column(partition_col)
    if col.dtype != pl.Date:
        return col.cast(pl.Utf8)
    if granularity == "year":
        return col.dt.strftime("%Y")
    if granularity == "month":
        return col.dt.strftime("%Y-%m")
    return col.dt.strftime("%Y-%m-%d")


def compact_dataset(
    staging_root: Path,
    curated_root: Path,
    dataset: str,
    run_id: str,
    partition_col: str | None = "trade_date",
    granularity: Granularity | None = None,
) -> int:
    """Merge staging batches into curated partitions, dedupe by PK.

    Granularity decides how many dates share one partition directory; it comes
    from the dataset's registry entry unless overridden. See
    ``domain/partitions.py`` for why it is not always a day.
    """
    if granularity is None:
        granularity = granularity_for_dataset(dataset)
    staging = StagingWriter(staging_root)
    curated = CuratedWriter(curated_root)
    files = staging.list_run_files(dataset, run_id)
    if not files:
        return 0

    # Re-validate on read: staging/curated written before a schema change
    # (e.g. fetched_at str → timestamp) must be normalized before concat.
    combined = pl.concat(
        [validate_dataframe(pl.read_parquet(f), dataset) for f in files],
        how="diagonal_relaxed",
    )
    pk = PRIMARY_KEYS.get(dataset, [])
    if pk:
        combined = combined.sort("fetched_at").unique(subset=pk, keep="last")

    if partition_col not in combined.columns:
        out_dir = curated.curated_root / dataset
        out_path = out_dir / "part-merged.parquet"
        existing_files = sorted(out_dir.rglob("*.parquet")) if out_dir.exists() else []
        if existing_files:
            existing = pl.concat(
                [validate_dataframe(pl.read_parquet(path), dataset) for path in existing_files],
                how="diagonal_relaxed",
            )
            combined = pl.concat([existing, combined], how="diagonal_relaxed")
            if pk:
                combined = combined.sort("fetched_at").unique(subset=pk, keep="last")
        out_dir.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(out_path, combined, compression="zstd")
        for stale in out_dir.rglob("*.parquet"):
            if stale != out_path:
                stale.unlink()
        return combined.height

    _PART = "__partition__"
    combined = combined.with_columns(
        _partition_values(combined, partition_col, granularity).alias(_PART)
    )

    total = 0
    for key, group in combined.partition_by(_PART, as_dict=True).items():
        val_str = str(key[0] if isinstance(key, tuple) else key)
        group = group.drop(_PART)
        existing_dir = curated.partition_path(dataset, partition_col, val_str)
        frames = [group]
        if existing_dir.exists():
            for existing in existing_dir.rglob("*.parquet"):
                frames.append(validate_dataframe(pl.read_parquet(existing), dataset))
        merged = pl.concat(frames, how="diagonal_relaxed")
        if pk:
            merged = merged.sort("fetched_at").unique(subset=pk, keep="last")
        curated.write_partition(dataset, partition_col, val_str, merged, "part-merged.parquet")
        total += merged.height
    return total


def _partition_col_for_dataset(dataset: str) -> str | None:
    """Return the partition column for a dataset from the registry."""
    from cnequity.domain.datasets import DATASETS

    spec = DATASETS.get(dataset)
    return spec.partition_col if spec else None


# ---------------------------------------------------------------------------
# External compaction: merge staging into an adapter's own files.
# ---------------------------------------------------------------------------


def compact_dataset_external(
    staging_root: Path,
    dataset: str,
    run_id: str,
    adapter,
    config,
) -> int:
    """Merge staging batches into an external adapter's own files.

    Unlike ``compact_dataset`` which writes to ``curated/``, this function
    writes back to the adapter's original file layout (e.g. yearly Tushare-wide
    archive).  The adapter provides:

    * ``compact_layout()`` → ``"yearly_file"`` or ``"hive"``
    * ``compact_target(config, dataset, trade_date)`` → Path
    * ``compact_pk()`` → list[str]  (PK for dedup; may differ from CNE)

    Staging data is **not** validated against CNE schemas — external files use
    vendor-controlled schemas (e.g. 33-column Tushare wide table with
    ``ts_code`` instead of ``symbol``) that differ from CNE's normalized
    contract.  Deduplication uses the adapter's PK, not ``PRIMARY_KEYS``,
    because the column names differ (``ts_code`` vs ``symbol``, ``vol`` vs
    ``volume``).
    """
    staging = StagingWriter(staging_root)
    files = staging.list_run_files(dataset, run_id)
    if not files:
        return 0

    combined = pl.concat(
        [pl.read_parquet(f) for f in files],
        how="diagonal_relaxed",
    )

    pk: list[str] = []
    if hasattr(adapter, "compact_pk"):
        pk = adapter.compact_pk(dataset) or []
    if not pk:
        pk = PRIMARY_KEYS.get(dataset, [])

    if pk:
        # Dedup staging batches.  Cannot sort by ``fetched_at`` — external
        # staging may not carry that column (Tushare wide format doesn't).
        combined = combined.unique(subset=pk, keep="last")

    layout = adapter.compact_layout()

    if layout == "yearly_file":
        return _compact_yearly_file(combined, dataset, pk, adapter, config)
    elif layout == "hive":
        raise NotImplementedError("hive layout not yet implemented for external compact")
    else:
        raise ValueError(f"unknown compact layout: {layout!r}")


def _compact_yearly_file(
    combined: pl.DataFrame,
    dataset: str,
    pk: list[str],
    adapter,
    config,
) -> int:
    """Merge *combined* staging rows into yearly archive files.

    Groups staging by year (derived from the dataset's partition column), then
    for each year calls the adapter's ``compact_target()`` to resolve the file
    path and performs an atomic anti-join merge.
    """
    from cnequity.domain.datasets import DATASETS

    spec = DATASETS.get(dataset)
    partition_col = spec.partition_col if spec else "trade_date"
    if partition_col not in combined.columns:
        raise ValueError(
            f"compact_dataset_external: dataset {dataset!r} staging data "
            f"missing partition column {partition_col!r}"
        )

    # Group by year (derived from the partition date column).
    combined = combined.with_columns(
        pl.col(partition_col).dt.year().cast(pl.Utf8).alias("__year__")
    )

    total = 0
    for key, group in combined.partition_by("__year__", as_dict=True).items():
        year_str = str(key[0] if isinstance(key, tuple) else key)
        group = group.drop("__year__")

        first_date = group.select(pl.col(partition_col).min()).item()
        target_path = adapter.compact_target(config, dataset, first_date)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        rows = _atomic_merge_yearly(target_path, group, pk)
        total += rows
        logger.info(
            "compact_external %s year=%s: %d rows merged -> %s (total %d)",
            dataset, year_str, group.height, target_path, rows,
        )

    return total


def _atomic_merge_yearly(
    target_path: Path,
    new_df: pl.DataFrame,
    pk: list[str],
) -> int:
    """Atomically merge *new_df* into *target_path*, deduplicating by *pk*.

    Uses anti-join to remove rows from the existing file whose PKs appear in
    *new_df*, then concatenates.  The swap is atomic via ``os.replace``.

    Does **not** validate against CNE schema — external files use vendor
    schemas.  Does **not** sort by ``fetched_at`` — external files may not
    carry that column.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(target_path.parent),
        prefix=".merge-",
        suffix=".parquet",
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    try:
        if target_path.exists():
            old = pl.read_parquet(target_path)

            # Align trade_date type: legacy files may store datetime, new rows
            # store Date.  The anti-join requires matching key types.
            new_aligned = new_df
            if (
                "trade_date" in old.columns
                and "trade_date" in new_aligned.columns
                and old.schema["trade_date"] != new_aligned.schema["trade_date"]
            ):
                new_aligned = new_aligned.with_columns(
                    pl.col("trade_date").cast(old.schema["trade_date"])
                )

            pk_cols = [c for c in pk if c in old.columns and c in new_aligned.columns]
            if pk_cols:
                old_without_overlap = old.join(
                    new_aligned.select(pk_cols).unique(),
                    on=pk_cols,
                    how="anti",
                )
                combined = pl.concat(
                    [old_without_overlap, new_aligned],
                    how="diagonal_relaxed",
                )
            else:
                combined = pl.concat([old, new_aligned], how="vertical_relaxed")
        else:
            combined = new_df

        combined.write_parquet(tmp_path, compression="zstd")
        os.replace(tmp_path, target_path)
        return combined.height
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
