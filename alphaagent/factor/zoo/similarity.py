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
SIMILARITY_BASIS_SAMPLED = "sample_row_ids"


def _sampled_pearson_vs_rows(
    x: np.ndarray,
    Y: np.ndarray,
    *,
    min_pairs: int = 10,
) -> np.ndarray:
    """NaN 感知的向量化 Pearson：x(S,) 与 Y(F,S) 每行的相关系数，返回 (F,)。

    在 zoo.index.sample_row_ids 抽样行上计算；等价于对每个因子在相同行子集上
    做 Pearson，避免逐因子重建千万行 MultiIndex 的全量循环。
    """
    x64 = np.asarray(x, dtype=np.float64)
    Y64 = np.asarray(Y, dtype=np.float64)
    mask = np.isfinite(x64)[None, :] & np.isfinite(Y64)
    n = mask.sum(axis=1)
    out = np.full(Y64.shape[0], np.nan, dtype=np.float64)
    valid = n >= max(int(min_pairs), 2)
    if not valid.any():
        return out
    xm = np.where(mask, x64[None, :], 0.0)
    ym = np.where(mask, Y64, 0.0)
    sx = xm.sum(axis=1)
    sy = ym.sum(axis=1)
    sxx = np.einsum("ij,ij->i", xm, xm)
    syy = np.einsum("ij,ij->i", ym, ym)
    sxy = np.einsum("ij,ij->i", xm, ym)
    nv = n.astype(np.float64)
    cov = sxy - sx * sy / nv
    vx = sxx - sx * sx / nv
    vy = syy - sy * sy / nv
    denom = np.sqrt(vx * vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov / denom, np.nan)
    out[valid] = corr[valid]
    return out


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
        candidate_values: np.ndarray | None,
        *,
        exclude_factor_id: str | None = None,
        min_pairs: int = 10,
        top_k: int = 3,
        candidate_sample: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """候选因子与库内因子的相似度报告（抽样行快速口径）。

        在 ``zoo.index.sample_row_ids``（默认 20 万行、按日分层）上计算候选与
        每个库内因子的 Pearson 相关，一次读入全部因子抽样摘要后向量化求解。
        与旧的"逐因子全量行 × 逐日 MultiIndex groupby"实现相比语义近似但快
        数个量级；``basis`` 字段标注口径供审计。

        ``candidate_values`` 须为 canonical 行序全量值；或直接传对齐到抽样行的
        ``candidate_sample``（会话域复用时避免构造 canonical 全量数组）。
        """
        empty = {"kind": SIMILARITY_KIND, "basis": SIMILARITY_BASIS_SAMPLED, "max_abs_corr": 0.0, "top_neighbors": []}
        order = [fid for fid in zoo.catalog.list_factor_ids() if fid != exclude_factor_id]
        if not order:
            return empty

        summaries, all_order = zoo.read_sample_summaries()
        pos = {fid: i for i, fid in enumerate(all_order)}
        used_order = [fid for fid in order if fid in pos]
        if not used_order:
            return empty
        rows_idx = [pos[fid] for fid in used_order]

        if candidate_sample is not None:
            cand_sample = np.asarray(candidate_sample, dtype=np.float32)
        else:
            if candidate_values is None:
                raise ValueError("candidate_values 与 candidate_sample 至少提供一个")
            cand_sample = zoo.extract_sample_from_values(candidate_values)
        corr_arr = _sampled_pearson_vs_rows(
            cand_sample, summaries[rows_idx], min_pairs=min_pairs
        )

        corrs: list[tuple[str, float]] = [
            (fid, float(c)) for fid, c in zip(used_order, corr_arr) if np.isfinite(c)
        ]
        corrs.sort(key=lambda p: abs(p[1]), reverse=True)
        max_abs = max((abs(c) for _, c in corrs), default=0.0)
        top_slice = corrs[:top_k] if top_k > 0 else []

        return {
            "kind": SIMILARITY_KIND,
            "basis": SIMILARITY_BASIS_SAMPLED,
            "n_sample_rows": int(len(cand_sample)),
            "max_abs_corr": max_abs,
            "top_neighbors": self._enrich_neighbors(zoo, top_slice),
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
        summaries, all_order = zoo.read_sample_summaries()
        pos = {fid: i for i, fid in enumerate(all_order)}
        used_order = [fid for fid in existing_order if fid in pos]
        corrs: list[tuple[str, float]] = []
        if used_order:
            rows_idx = [pos[fid] for fid in used_order]
            cand_sample = zoo.extract_sample_from_values(values)
            corr_arr = _sampled_pearson_vs_rows(
                cand_sample, summaries[rows_idx], min_pairs=min_pairs
            )
            for fid, c in zip(used_order, corr_arr):
                j = col_map.get(fid)
                if j is not None and np.isfinite(c):
                    mat[col_idx, j] = float(c)
                    mat[j, col_idx] = float(c)
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
            "basis": SIMILARITY_BASIS_SAMPLED,
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
