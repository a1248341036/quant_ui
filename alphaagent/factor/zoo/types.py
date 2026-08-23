"""因子库核心类型（股票日频 stock_1d）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "stock"
DEFAULT_BAR_INTERVAL = "1d"


class FactorStatus(str, Enum):
    partial = "partial"
    full = "full"
    materializing = "materializing"


@dataclass(frozen=True)
class RowSlice:
    start: int
    stop: int  # exclusive

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError(f"非法 RowSlice: {self.start}:{self.stop}")


@dataclass(frozen=True)
class TimeShard:
    shard_id: str
    start_row: int
    stop_row: int
    datetime_start: str
    datetime_end: str


@dataclass
class LibraryManifest:
    dataset: str
    bar_interval: str
    universe_path: str
    n_rows: int
    n_sample_rows: int
    max_factors: int
    dtype: str = "float32"
    index_hash: str = ""
    sample_seed: int = 42
    version: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "bar_interval": self.bar_interval,
            "universe_path": str(self.universe_path),
            "n_rows": self.n_rows,
            "n_sample_rows": self.n_sample_rows,
            "max_factors": self.max_factors,
            "dtype": self.dtype,
            "index_hash": self.index_hash,
            "sample_seed": self.sample_seed,
            "version": self.version,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LibraryManifest:
        known = {
            "dataset",
            "bar_interval",
            "universe_path",
            "panel_path",
            "n_rows",
            "n_sample_rows",
            "n_sketch",
            "max_factors",
            "dtype",
            "index_hash",
            "sample_seed",
            "sketch_seed",
            "version",
            "base_interval",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        panel_path = data.get("panel_path", data.get("universe_path", ""))
        return cls(
            dataset=str(data.get("dataset", DEFAULT_DATASET)),
            bar_interval=str(data.get("bar_interval", DEFAULT_BAR_INTERVAL)),
            universe_path=str(panel_path),
            n_rows=int(data["n_rows"]),
            n_sample_rows=int(
                data["n_sample_rows"] if "n_sample_rows" in data else data["n_sketch"]
            ),
            max_factors=int(data["max_factors"]),
            dtype=str(data.get("dtype", "float32")),
            index_hash=str(data.get("index_hash", "")),
            sample_seed=int(data.get("sample_seed", data.get("sketch_seed", 42))),
            version=int(data.get("version", 1)),
            extra=extra,
        )

    @property
    def panel_path(self) -> str:
        return str(self.extra.get("panel_path", self.universe_path))


@dataclass(frozen=True)
class FactorMeta:
    factor_id: str
    name: str
    expr: str
    col_idx: int
    status: FactorStatus
    finite_count: int = 0
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorLibraryPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def rows_parquet(self) -> Path:
        return self.index_dir / "rows.parquet"

    @property
    def shards_json(self) -> Path:
        return self.index_dir / "shards.json"

    @property
    def sample_row_ids(self) -> Path:
        return self.index_dir / "sample_row_ids.parquet"

    @property
    def sample_dir(self) -> Path:
        return self.root / "sample"

    @property
    def sample_summary_memmap(self) -> Path:
        return self.sample_dir / "factor_samples.f32.memmap"

    @property
    def sample_summary_meta(self) -> Path:
        return self.sample_dir / "factor_samples.meta.json"

    @property
    def values_dir(self) -> Path:
        return self.root / "values"

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @property
    def factors_parquet(self) -> Path:
        return self.meta_dir / "factors.parquet"

    @property
    def expressions_dir(self) -> Path:
        return self.root / "expressions"

    @property
    def similarity_dir(self) -> Path:
        return self.root / "similarity"

    def factor_values_path(self, factor_id: str) -> Path:
        return self.values_dir / f"f_{factor_id}.f32.memmap"

    def resolve_sample_row_ids(self) -> Path:
        return self.sample_row_ids

    def resolve_sample_summary_memmap(self) -> Path:
        return self.sample_summary_memmap

    def resolve_sample_summary_meta(self) -> Path:
        return self.sample_summary_meta
