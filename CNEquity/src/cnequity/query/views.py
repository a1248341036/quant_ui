"""DuckDB view layer — one view per dataset, generated from the registry."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from cnequity.config import Config
from cnequity.domain.datasets import DATASETS, DatasetSpec
from cnequity.domain.partitions import uses_hive
from cnequity.domain.schemas import DATASET_SCHEMAS, PRIMARY_KEYS
from cnequity.external import registry as ext_registry


def _duckdb_type(dtype: pl.DataType) -> str:
    if isinstance(dtype, pl.Datetime):
        return "TIMESTAMPTZ" if dtype.time_zone else "TIMESTAMP"
    if dtype == pl.Date:
        return "DATE"
    if dtype == pl.Boolean:
        return "BOOLEAN"
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        return "BIGINT"
    if dtype in (pl.Float32, pl.Float64):
        return "DOUBLE"
    return "VARCHAR"


def _empty_view_sql(name: str) -> str:
    schema = DATASET_SCHEMAS[name]
    cols = ",\n            ".join(
        f"CAST(NULL AS {_duckdb_type(dtype)}) AS {col}" for col, dtype in schema.items()
    )
    return f"""
        CREATE OR REPLACE VIEW {name} AS
        SELECT
            {cols}
        WHERE false
    """


def _view_glob(data_root: str, spec: DatasetSpec) -> tuple[str, bool]:
    # *data_root* is always POSIX-form (see ensure_duckdb_views): DuckDB's
    # read_parquet glob accepts `/` on every platform, and backslashes would
    # either escape the SQL string or fail to match files on Windows.
    layer_dir = "derived" if spec.layer == "derived" else "curated"
    if spec.partition_col is None:
        return f"{data_root}/{layer_dir}/{spec.name}/**/*.parquet", False
    # Hive parsing only for day granularity: a `trade_date=2024` directory
    # cannot be read as the DATE column it sits beside. The real column is in
    # the file either way, so the view is identical apart from pruning.
    return (
        f"{data_root}/{layer_dir}/{spec.name}/**/*.parquet",
        uses_hive(spec.partition_granularity),
    )


def _glob_has_files(pattern: str) -> bool:
    base = pattern.split("**")[0].split("*")[0].rstrip("/")
    p = Path(base)
    if not p.exists():
        return False
    return any(p.rglob("*.parquet"))


def _external_source_sql(paths: list[Path]) -> str:
    """Build a DuckDB source over adapter-owned files without copying them."""
    quoted = []
    for path in paths:
        value = path.resolve().as_posix().replace("'", "''")
        quoted.append(f"'{value}'")
    suffixes = {p.suffix.lower() for p in paths}
    if suffixes == {".parquet"}:
        return f"read_parquet([{', '.join(quoted)}], union_by_name=true)"
    if suffixes == {".csv"}:
        return (
            f"read_csv([{', '.join(quoted)}], "
            "header=true, union_by_name=true, ignore_errors=true, "
            "delim=',', quote='\"', escape='\"', strict_mode=false, "
            "null_padding=true)"
        )
    raise ValueError(f"mixed external file types are not supported: {sorted(suffixes)}")


def _external_daily_bars_view(external_paths: list[Path]) -> str:
    """Expose the Tushare-wide archive through CNE's canonical bar contract."""
    source = _external_source_sql(external_paths)
    return f"""
        SELECT
            CAST(ts_code AS VARCHAR) AS symbol,
            CAST(trade_date AS DATE) AS trade_date,
            CAST(open AS DOUBLE) AS open,
            CAST(high AS DOUBLE) AS high,
            CAST(low AS DOUBLE) AS low,
            CAST(close AS DOUBLE) AS close,
            CAST(round(CAST(vol AS DOUBLE) * 100.0) AS BIGINT) AS volume,
            CAST(amount AS DOUBLE) * 1000.0 AS amount,
            'quant_dataset_tushare' AS source,
            'tushare-wide-v1' AS data_version,
            CAST('2000-01-01 00:00:00+00' AS TIMESTAMPTZ) AS fetched_at
        FROM {source}
        WHERE COALESCE(vol, 0) > 0
          AND COALESCE(open, 0) > 0
          AND COALESCE(high, 0) > 0
          AND COALESCE(low, 0) > 0
          AND COALESCE(close, 0) > 0
    """


def _view_select_sql(
    name: str,
    glob_path: str,
    hive: bool,
    *,
    columns: set[str] | None = None,
) -> str:
    """Build a canonical dataset view over all parquet fragments.

    DuckDB is a separate read path from :func:`cnequity.query.reader.load`.
    Apply the same latest-by-``fetched_at`` PK rule here, otherwise an old
    fragment or overlapping retry can multiply rows in SQL joins while the
    Python API returns one canonical observation.
    """
    primary_key = PRIMARY_KEYS.get(name, [])
    source = (
        f"read_parquet('{glob_path}', hive_partitioning={str(hive).lower()}, union_by_name=true)"
    )
    # A few pre-schema-migration fragments in the wild (and lightweight
    # A few pre-schema-migration fragments in the wild (and lightweight
    # bootstrap fixtures) do not carry provenance. There is no honest
    # freshness tie-breaker in that case; keep the raw view readable and let
    # the schema/quality checks report the malformed fragment.
    if not primary_key or (
        columns is not None and not set([*primary_key, "fetched_at"]).issubset(columns)
    ):
        return f"SELECT * FROM {source}"
    partition_by = ", ".join(primary_key)
    return (
        "SELECT * FROM "
        f"{source} "
        "QUALIFY ROW_NUMBER() OVER ("
        f"PARTITION BY {partition_by} ORDER BY fetched_at DESC NULLS LAST"
        ") = 1"
    )


def ensure_duckdb_views(config: Config, *, require_data: bool = False) -> Path:
    db_path = config.duckdb_path or (config.data_root / "duckdb" / "cnequity.duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # as_posix() keeps Windows drive letters (`C:/…`) while turning `\` into
    # `/`, which is what the SQL literals below and DuckDB's glob both want.
    root = config.data_root.resolve().as_posix().replace("'", "''")

    con = duckdb.connect(str(db_path))
    con.execute(f"SET memory_limit='{config.duckdb_memory_limit}'")
    con.execute(f"SET threads={config.duckdb_threads}")

    for name, spec in sorted(DATASETS.items()):
        # Adapter-mounted datasets are first-class query names, but their files
        # stay at their source paths. daily_bars is the special case where an
        # external archive supplies a native dataset's full history.
        adapter = ext_registry.external_adapter(config, name)
        if adapter is not None and not getattr(adapter, "is_native", False):
            try:
                external_paths = adapter.files(config, name)
            except Exception:
                external_paths = []
            if external_paths:
                if name == "daily_bars":
                    select_sql = _external_daily_bars_view(external_paths)
                else:
                    select_sql = f"SELECT * FROM {_external_source_sql(external_paths)}"
                con.execute(
                    f"""
                    CREATE OR REPLACE VIEW {name} AS
                    {select_sql}
                    """
                )
                continue
            if name not in DATASET_SCHEMAS:
                continue
        glob_path, hive = _view_glob(root, spec)
        if _glob_has_files(glob_path) or require_data:
            source = (
                f"read_parquet('{glob_path}', "
                f"hive_partitioning={str(hive).lower()}, union_by_name=true)"
            )
            columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {name} AS
                {_view_select_sql(name, glob_path, hive, columns=columns)}
                """
            )
        else:
            # Skip datasets without a registered schema (e.g. external
            # read-only bridges) — they have no curated parquet to view.
            if name not in DATASET_SCHEMAS:
                continue
            con.execute(_empty_view_sql(name))

    # Adjusted bars per ADR-0004: only hfq factors are stored.
    #   hfq price = raw * factor
    #   qfq price = raw * factor / anchor   (anchor = symbol's latest hfq factor)
    # adj_* keeps its historical qfq meaning; adj_is_exact mirrors the Python API.
    con.execute(
        """
        CREATE OR REPLACE VIEW daily_bars_adj AS
        WITH hfq AS (
            SELECT symbol, trade_date, factor
            FROM adj_factors
            WHERE adjust_type = 'hfq'
        ),
        anchors AS (
            SELECT symbol, factor AS hfq_anchor
            FROM hfq
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) = 1
        )
        SELECT
            b.*,
            h.factor IS NOT NULL AS adj_is_exact,
            b.open  * COALESCE(h.factor, 1.0) AS hfq_open,
            b.high  * COALESCE(h.factor, 1.0) AS hfq_high,
            b.low   * COALESCE(h.factor, 1.0) AS hfq_low,
            b.close * COALESCE(h.factor, 1.0) AS hfq_close,
            b.open  * COALESCE(h.factor / a.hfq_anchor, 1.0) AS qfq_open,
            b.high  * COALESCE(h.factor / a.hfq_anchor, 1.0) AS qfq_high,
            b.low   * COALESCE(h.factor / a.hfq_anchor, 1.0) AS qfq_low,
            b.close * COALESCE(h.factor / a.hfq_anchor, 1.0) AS qfq_close,
            b.close * COALESCE(h.factor / a.hfq_anchor, 1.0) AS adj_close
        FROM daily_bars b
        LEFT JOIN hfq h
          ON b.symbol = h.symbol AND b.trade_date = h.trade_date
        LEFT JOIN anchors a
          ON b.symbol = a.symbol
        """
    )
    con.close()
    return db_path
