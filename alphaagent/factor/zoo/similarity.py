"""因子两两截面 Pearson 相似度（逐日横截面相关均值）。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.factor.zoo.index import RowIndex
from alphaagent.factor.zoo.types import FactorLibraryPaths
from alphaagent.factor.zoo.zoo import FactorZoo

SIMILARITY_KIND = "cross_sectional_pearson_mean"


def _pearson_ic(a: np.ndarray, b: np.ndarray, *, min_pairs: int = 2) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < min_pairs:
        return float("nan")
    xs = x[mask] - x[mask].mean()
    ys = y[mask] - y[mask].mean()
    denom = float(np.sqrt((xs * xs).sum() * (ys * ys).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((xs * ys).sum() / denom)


def panel_index_from_rows(rows: pd.DataFrame) -> pd.MultiIndex:
    dt = pd.to_datetime(rows["datetime"], errors="coerce")
    inst = rows["instrument"].astype(str)
    return pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])


def cross_sectional_pearson_series(
    a: np.ndarray,
    b: np.ndarray,
    row_index: RowIndex,
    *,
    min_pairs: int = 10,
) -> pd.Series:
    """逐日横截面 Pearson 相关，返回逐日序列。"""
    if len(a) != len(b):
        raise ValueError(f"因子长度不一致: {len(a)} vs {len(b)}")
    index = panel_index_from_rows(row_index.rows)
    fa = pd.Series(np.asarray(a, dtype=np.float32), index=index)
    fb = pd.Series(np.asarray(b, dtype=np.float32), index=index)

    rows: list[float] = []
    idx: list[object] = []
    for ts, f_sub in fa.groupby(level="datetime", sort=False):
        b_sub = fb.xs(ts, level="datetime")
        ic = _pearson_ic(
            f_sub.to_numpy(dtype=np.float64, copy=False),
            b_sub.to_numpy(dtype=np.float64, copy=False),
            min_pairs=min_pairs,
        )
        rows.append(ic)
        idx.append(ts)
    return pd.Series(rows, index=pd.Index(idx, name="datetime"), dtype=float)


def cross_sectional_pearson_mean(
    a: np.ndarray,
    b: np.ndarray,
    row_index: RowIndex,
    *,
    min_pairs: int = 10,
) -> float:
    """逐日横截面 Pearson 相关的均值（截面相似度）。"""
    daily = cross_sectional_pearson_series(a, b, row_index, min_pairs=min_pairs)
    finite = daily[np.isfinite(daily.to_numpy(dtype=float, copy=False))]
    if len(finite) == 0:
        return float("nan")
    return float(finite.mean())


class SimilarityMatrix:
    def __init__(self, paths: FactorLibraryPaths, max_factors: int) -> None:
        self.paths = paths
        self.max_factors = max_factors
        self.matrix_path = paths.similarity_dir / "pearson.f32.memmap"
        self.meta_path = paths.similarity_dir / "pearson.meta.json"

    def _ensure(self, n_factors: int) -> np.memmap:
        self.matrix_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.matrix_path.is_file():
            arr = np.memmap(
                self.matrix_path,
                dtype=np.float32,
                mode="w+",
                shape=(self.max_factors, self.max_factors),
            )
            arr[:] = np.nan
            if n_factors > 0:
                np.fill_diagonal(arr[:n_factors, :n_factors], 1.0)
            arr.flush()
            self._write_meta(n_factors)
            return arr
        return np.memmap(
            self.matrix_path,
            dtype=np.float32,
            mode="r+",
            shape=(self.max_factors, self.max_factors),
        )

    def _write_meta(self, n_factors: int) -> None:
        meta = {
            "n_factors": n_factors,
            "max_factors": self.max_factors,
            "kind": SIMILARITY_KIND,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_meta(self) -> dict:
        if not self.meta_path.is_file():
            return {"n_factors": 0, "max_factors": self.max_factors, "kind": SIMILARITY_KIND}
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        meta.setdefault("kind", SIMILARITY_KIND)
        return meta

    def get_matrix(self, n_factors: int) -> np.ndarray:
        if not self.matrix_path.is_file():
            return np.zeros((0, 0), dtype=np.float32)
        mmap = np.memmap(
            self.matrix_path,
            dtype=np.float32,
            mode="r",
            shape=(self.max_factors, self.max_factors),
        )
        return np.array(mmap[:n_factors, :n_factors], copy=True)

    def max_cross_sectional_correlation(
        self,
        zoo: FactorZoo,
        candidate_values: np.ndarray,
        *,
        exclude_factor_id: str | None = None,
        min_pairs: int = 10,
    ) -> float:
        """候选因子与库内因子的最大截面 |corr|。"""
        report = self.cross_sectional_neighbor_report(
            zoo,
            candidate_values,
            exclude_factor_id=exclude_factor_id,
            min_pairs=min_pairs,
            top_k=0,
        )
        return float(report["max_abs_corr"])

    def cross_sectional_neighbor_report(
        self,
        zoo: FactorZoo,
        candidate_values: np.ndarray,
        *,
        exclude_factor_id: str | None = None,
        min_pairs: int = 10,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """候选因子与库内因子的截面相关报告：max |corr| 与 top_k 最相似因子（含 expr）。"""
        order = zoo.catalog.list_factor_ids()
        if exclude_factor_id is not None:
            order = [fid for fid in order if fid != exclude_factor_id]

        corrs: list[tuple[str, float]] = []
        for fid in order:
            other = zoo.read_factor(fid)
            c = cross_sectional_pearson_mean(
                candidate_values, other, zoo.index, min_pairs=min_pairs
            )
            if np.isfinite(c):
                corrs.append((fid, float(c)))

        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        max_abs = max((abs(c) for _, c in corrs), default=0.0)
        top_slice = corrs[:top_k] if top_k > 0 else []

        enriched: list[dict[str, Any]] = []
        for fid, c in top_slice:
            meta = zoo.catalog.get(fid)
            enriched.append(
                {
                    "factor_id": fid,
                    "name": meta.name if meta is not None else fid,
                    "cs_corr": c,
                    "expr": meta.expr if meta is not None else None,
                }
            )

        return {
            "kind": SIMILARITY_KIND,
            "max_abs_corr": max_abs,
            "top_neighbors": enriched,
        }

    @staticmethod
    def _max_abs_corr_from_neighbors(neighbors: list[dict[str, Any]]) -> float:
        vals = [abs(float(nb["cs_corr"])) for nb in neighbors if nb.get("cs_corr") is not None]
        return max(vals, default=0.0)

    @staticmethod
    def _enrich_neighbors(zoo: FactorZoo, pairs: list[tuple[str, float]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for fid, c in pairs:
            meta = zoo.catalog.get(fid)
            enriched.append(
                {
                    "factor_id": fid,
                    "name": meta.name if meta is not None else fid,
                    "cs_corr": c,
                    "expr": meta.expr if meta is not None else None,
                }
            )
        return enriched

    def append_factor_correlations(
        self,
        zoo: FactorZoo,
        *,
        factor_id: str,
        col_idx: int,
        values: np.ndarray,
        min_pairs: int = 10,
        top_k: int = 3,
    ) -> dict:
        meta = zoo._load_sample_summary_meta()
        col_map: dict[str, int] = {
            str(k): int(v) for k, v in meta.get("factor_id_to_col", {}).items()
        }
        mat = self._ensure(max(col_idx + 1, int(meta.get("next_col_idx", col_idx + 1))))

        existing_order = [fid for fid in zoo.catalog.list_factor_ids() if str(fid) != str(factor_id)]
        corrs: list[tuple[str, float]] = []
        for fid in existing_order:
            other = zoo.read_factor(fid)
            c = cross_sectional_pearson_mean(values, other, zoo.index, min_pairs=min_pairs)
            j = col_map.get(fid)
            if j is not None and np.isfinite(c):
                mat[col_idx, j] = c
                mat[j, col_idx] = c
            if np.isfinite(c):
                corrs.append((fid, float(c)))
        mat[col_idx, col_idx] = 1.0
        mat.flush()

        n_active = zoo.n_factors
        self._write_meta(n_active)

        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_neighbors = self._enrich_neighbors(zoo, corrs[:top_k])
        return {
            "col_idx": col_idx,
            "n_factors": n_active,
            "kind": SIMILARITY_KIND,
            "max_abs_corr": self._max_abs_corr_from_neighbors(top_neighbors),
            "top_neighbors": top_neighbors,
        }

    def remove_factor(self, col_idx: int, *, n_active: int) -> None:
        if not self.matrix_path.is_file():
            return
        mat = np.memmap(
            self.matrix_path,
            dtype=np.float32,
            mode="r+",
            shape=(self.max_factors, self.max_factors),
        )
        mat[col_idx, :] = np.nan
        mat[:, col_idx] = np.nan
        mat.flush()
        self._write_meta(max(n_active, 0))
