"""因子库：全量 memmap + 抽样摘要。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from alphaagent.factor.zoo.catalog import FactorCatalog
from alphaagent.factor.zoo.index import RowIndex, verify_index_hash
from alphaagent.factor.zoo.types import (
    FactorLibraryPaths,
    FactorStatus,
    LibraryManifest,
    RowSlice,
)


def _load_manifest(path: Path) -> LibraryManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LibraryManifest.from_dict(data)


def _finite_count(values: np.ndarray) -> int:
    v = np.asarray(values, dtype=np.float32)
    return int(np.isfinite(v).sum())


def _create_memmap_1d(path: Path, n_rows: int, values: np.ndarray | None = None) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype=np.float32, mode="w+", shape=(n_rows,))
    arr[:] = np.nan
    if values is not None:
        if len(values) != n_rows:
            raise ValueError(f"values 长度 {len(values)} != n_rows {n_rows}")
        arr[:] = np.asarray(values, dtype=np.float32)
    arr.flush()
    return arr


def _open_memmap_1d(path: Path, n_rows: int, mode: str = "r") -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(n_rows,))


def _create_sample_summary_memmap(path: Path, max_factors: int, n_sample_rows: int) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype=np.float32, mode="w+", shape=(max_factors, n_sample_rows))
    arr[:] = np.nan
    arr.flush()
    return arr


def _open_sample_summary_memmap(
    path: Path, max_factors: int, n_sample_rows: int, mode: str = "r"
) -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(max_factors, n_sample_rows))


def _normalize_sample_meta(meta: dict, *, n_sample_rows: int, max_factors: int) -> dict:
    out = dict(meta)
    if "n_sample_rows" not in out and "n_sketch" in out:
        out["n_sample_rows"] = out["n_sketch"]
    out.setdefault("n_sample_rows", n_sample_rows)
    out.setdefault("max_factors", max_factors)
    out.setdefault("n_factors", 0)
    out.setdefault("factor_id_to_col", {})
    return out


class FactorZoo:
    def __init__(
        self,
        paths: FactorLibraryPaths,
        manifest: LibraryManifest,
        index: RowIndex,
        catalog: FactorCatalog,
    ) -> None:
        self.paths = paths
        self.manifest = manifest
        self.index = index
        self.catalog = catalog

    @classmethod
    def open(cls, root: Path, *, verify_hash: bool = True) -> FactorZoo:
        paths = FactorLibraryPaths(root=Path(root).expanduser().resolve())
        if not paths.manifest.is_file():
            raise FileNotFoundError(f"因子库未初始化: {paths.manifest}")
        manifest = _load_manifest(paths.manifest)
        index = RowIndex.load(paths)
        if verify_hash:
            verify_index_hash(manifest, index)
        catalog = FactorCatalog(paths.factors_parquet)
        return cls(paths, manifest, index, catalog)

    def _ensure_sample_summary_files(self) -> None:
        memmap_path = self.paths.resolve_sample_summary_memmap()
        if not memmap_path.is_file():
            _create_sample_summary_memmap(
                self.paths.sample_summary_memmap,
                self.manifest.max_factors,
                self.index.n_sample_rows,
            )
            meta = {
                "max_factors": self.manifest.max_factors,
                "n_sample_rows": self.index.n_sample_rows,
                "n_factors": 0,
                "factor_id_to_col": {},
            }
            self.paths.sample_summary_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_sample_summary_meta(self) -> dict:
        self._ensure_sample_summary_files()
        meta_path = self.paths.resolve_sample_summary_meta()
        if meta_path.is_file():
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            return _normalize_sample_meta(
                raw,
                n_sample_rows=self.index.n_sample_rows,
                max_factors=self.manifest.max_factors,
            )
        return {
            "max_factors": self.manifest.max_factors,
            "n_sample_rows": self.index.n_sample_rows,
            "n_factors": 0,
            "factor_id_to_col": {},
        }

    def _save_sample_summary_meta(self, meta: dict) -> None:
        self.paths.sample_summary_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def append_factor(
        self,
        *,
        factor_id: str,
        name: str,
        expr: str,
        values: np.ndarray,
        status: FactorStatus = FactorStatus.full,
        extra: dict[str, Any] | None = None,
    ) -> int:
        n_rows = self.manifest.n_rows
        values = np.asarray(values, dtype=np.float32)
        if len(values) != n_rows:
            raise ValueError(f"values 长度 {len(values)} != n_rows {n_rows}")

        meta = self._load_sample_summary_meta()
        col_map: dict[str, int] = {str(k): int(v) for k, v in meta.get("factor_id_to_col", {}).items()}
        if factor_id in col_map:
            raise ValueError(f"factor_id 已存在: {factor_id}")

        col_idx = int(meta.get("next_col_idx", meta.get("n_factors", 0)))
        if col_idx >= self.manifest.max_factors:
            raise RuntimeError(
                f"因子列数已达 max_factors={self.manifest.max_factors}，需扩容或 compact"
            )

        out_path = self.paths.factor_values_path(factor_id)
        if out_path.is_file():
            raise ValueError(f"全量存储文件已存在: {out_path}")
        _create_memmap_1d(out_path, n_rows, values)

        summary = _open_sample_summary_memmap(
            self.paths.resolve_sample_summary_memmap(),
            self.manifest.max_factors,
            self.index.n_sample_rows,
            mode="r+",
        )
        summary[col_idx, :] = values[self.index.sample_row_ids]
        summary.flush()

        col_map[factor_id] = col_idx
        meta["factor_id_to_col"] = col_map
        meta["next_col_idx"] = col_idx + 1
        meta["n_factors"] = len(col_map)
        self._save_sample_summary_meta(meta)

        self.catalog.append(
            factor_id=factor_id,
            name=name,
            expr=expr,
            col_idx=col_idx,
            status=status,
            finite_count=_finite_count(values),
            extra=extra,
        )
        return col_idx

    def overwrite_factor(
        self,
        *,
        factor_id: str,
        name: str,
        expr: str,
        values: np.ndarray,
        status: FactorStatus = FactorStatus.full,
        extra: dict[str, Any] | None = None,
    ) -> int:
        entry = self.catalog.get(factor_id)
        if entry is None:
            raise KeyError(f"因子不存在，无法覆盖: {factor_id}")

        n_rows = self.manifest.n_rows
        values = np.asarray(values, dtype=np.float32)
        if len(values) != n_rows:
            raise ValueError(f"values 长度 {len(values)} != n_rows {n_rows}")

        col_idx = entry.col_idx
        out_path = self.paths.factor_values_path(factor_id)
        if out_path.is_file():
            mmap = _open_memmap_1d(out_path, n_rows, mode="r+")
            mmap[:] = values
            mmap.flush()
        else:
            _create_memmap_1d(out_path, n_rows, values)

        summary = _open_sample_summary_memmap(
            self.paths.resolve_sample_summary_memmap(),
            self.manifest.max_factors,
            self.index.n_sample_rows,
            mode="r+",
        )
        summary[col_idx, :] = values[self.index.sample_row_ids]
        summary.flush()

        self.catalog.update(
            factor_id,
            name=name,
            expr=expr,
            status=status,
            finite_count=_finite_count(values),
            extra=extra,
        )
        return col_idx

    def extend_factor_values(
        self,
        factor_id: str,
        tail: np.ndarray,
        *,
        old_n: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """尾部追加因子值：保留 [0:old_n) 前缀，写入 tail 至新 manifest.n_rows。"""
        entry = self.catalog.get(factor_id)
        if entry is None:
            raise KeyError(f"因子不存在: {factor_id}")

        tail = np.asarray(tail, dtype=np.float32)
        new_n = self.manifest.n_rows
        if old_n + len(tail) != new_n:
            raise ValueError(
                f"extend 长度不匹配: old_n={old_n} + tail={len(tail)} != n_rows={new_n}"
            )

        path = self.paths.factor_values_path(factor_id)
        if path.is_file():
            file_n = path.stat().st_size // np.dtype(np.float32).itemsize
            if file_n != old_n:
                raise ValueError(f"memmap 文件长度 {file_n} != old_n {old_n}")
            prefix = np.array(
                np.memmap(path, dtype=np.float32, mode="r", shape=(old_n,)),
                dtype=np.float32,
                copy=True,
            )
        else:
            prefix = np.full(old_n, np.nan, dtype=np.float32)

        full = np.concatenate([prefix, tail.astype(np.float32, copy=False)])
        _create_memmap_1d(path, new_n, full)

        col_idx = entry.col_idx
        summary = _open_sample_summary_memmap(
            self.paths.resolve_sample_summary_memmap(),
            self.manifest.max_factors,
            self.index.n_sample_rows,
            mode="r+",
        )
        summary[col_idx, :] = full[self.index.sample_row_ids]
        summary.flush()

        update_extra = dict(entry.extra or {})
        if extra is not None:
            update_extra.update(extra)
        self.catalog.update(
            factor_id,
            name=entry.name,
            expr=entry.expr,
            status=entry.status,
            finite_count=_finite_count(full),
            extra=update_extra,
        )

    @property
    def n_factors(self) -> int:
        return self.catalog.n_factors

    def read_factor(
        self,
        factor_id: str,
        *,
        row_slice: RowSlice | None = None,
        stride: int = 1,
    ) -> np.ndarray:
        path = self.paths.factor_values_path(factor_id)
        if not path.is_file():
            raise KeyError(f"因子不存在: {factor_id}")
        mmap = _open_memmap_1d(path, self.manifest.n_rows, mode="r")
        if row_slice is None:
            data = mmap[::stride]
        else:
            if row_slice.stop <= row_slice.start:
                return np.array([], dtype=np.float32)
            data = mmap[row_slice.start : row_slice.stop : stride]
        return np.array(data, dtype=np.float32, copy=True)

    def read_factors(
        self,
        factor_ids: list[str],
        *,
        row_slice: RowSlice | None = None,
        stride: int = 1,
    ) -> np.ndarray:
        cols = [self.read_factor(fid, row_slice=row_slice, stride=stride) for fid in factor_ids]
        if not cols:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(cols, axis=0)

    def read_sample_summaries(
        self, factor_ids: list[str] | None = None
    ) -> tuple[np.ndarray, list[str]]:
        meta = self._load_sample_summary_meta()
        col_map: dict[str, int] = {str(k): int(v) for k, v in meta["factor_id_to_col"].items()}
        if factor_ids is None:
            factor_ids = sorted(col_map.keys(), key=lambda x: col_map[x])
        summary = _open_sample_summary_memmap(
            self.paths.resolve_sample_summary_memmap(),
            self.manifest.max_factors,
            self.index.n_sample_rows,
            mode="r",
        )
        rows = []
        order: list[str] = []
        for fid in factor_ids:
            if fid not in col_map:
                raise KeyError(f"因子不在抽样摘要中: {fid}")
            order.append(fid)
            rows.append(np.array(summary[col_map[fid], :], dtype=np.float32, copy=True))
        if not rows:
            return np.zeros((0, self.index.n_sample_rows), dtype=np.float32), []
        return np.stack(rows, axis=0), order

    def extract_sample_from_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if len(values) != self.manifest.n_rows:
            raise ValueError(f"values 长度 {len(values)} != n_rows {self.manifest.n_rows}")
        return values[self.index.sample_row_ids].copy()

    def delete_factor(self, factor_id: str) -> None:
        entry = self.catalog.get(factor_id)
        if entry is None:
            raise KeyError(f"因子不存在: {factor_id}")

        col_idx = entry.col_idx
        self.catalog.remove(factor_id)

        meta = self._load_sample_summary_meta()
        col_map: dict[str, int] = {str(k): int(v) for k, v in meta.get("factor_id_to_col", {}).items()}
        col_map.pop(factor_id, None)
        meta["factor_id_to_col"] = col_map
        meta["n_factors"] = len(col_map)
        if "next_col_idx" not in meta:
            meta["next_col_idx"] = max(col_map.values(), default=-1) + 1
        self._save_sample_summary_meta(meta)

        summary = _open_sample_summary_memmap(
            self.paths.resolve_sample_summary_memmap(),
            self.manifest.max_factors,
            self.index.n_sample_rows,
            mode="r+",
        )
        summary[col_idx, :] = np.nan
        summary.flush()

        full_path = self.paths.factor_values_path(factor_id)
        if full_path.is_file():
            full_path.unlink()

        from alphaagent.factor.zoo.similarity import SimilarityMatrix

        sim = SimilarityMatrix(self.paths, self.manifest.max_factors)
        sim.remove_factor(col_idx, n_active=self.catalog.n_factors)

    def delete_factors(self, factor_ids: list[str]) -> int:
        deleted = 0
        for factor_id in factor_ids:
            try:
                self.delete_factor(factor_id)
                deleted += 1
            except KeyError:
                continue
        return deleted
