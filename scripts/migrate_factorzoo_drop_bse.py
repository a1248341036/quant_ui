#!/usr/bin/env python3
"""从 factorzoo canonical index 中剔除北交所（92/83/87/43）行并重映射因子值。

背景：CNE panel 曾包含北交所股票（投资者准入要求 50 万 + 24 个月），
导致 factorzoo 的 canonical index 含 250,810 行 BSE 占位。本脚本：

- 重建 index：去掉 BSE 行，row_id 重新连续编号，shard/sample 重建；
- production_main：因子值 memmap 按「旧非 BSE 行 → 新行序」重映射，
  并重建 sample 摘要 memmap（BSE 行值全为 NaN，无有效数据丢失）；
- candidate_main：values/ 为空（因子值在 registry 元数据），只重建 index。

用法：
    .venv\\Scripts\\python.exe scripts/migrate_factorzoo_drop_bse.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.factor.zoo.index import (  # noqa: E402
    RowIndex,
    build_row_index,
    build_sample_row_ids,
    build_time_shards,
    index_content_hash,
)
from alphaagent.factor.zoo.types import FactorLibraryPaths  # noqa: E402

BSE_PREFIXES = ("92", "83", "87", "43")


def _is_bse(inst: pd.Series) -> pd.Series:
    return inst.astype(str).str.startswith(BSE_PREFIXES)


def _rebuild_index(rows: pd.DataFrame, n_sample_rows: int, sample_seed: int) -> RowIndex:
    """去掉 BSE 行后重建 RowIndex（row_id 连续、shard/sample 重建）。"""
    inst = rows["instrument"].astype(str)
    keep = ~_is_bse(inst).to_numpy()
    new_rows = build_row_index(rows.loc[keep, ["datetime", "instrument"]])
    shards = build_time_shards(new_rows)
    sample_ids = build_sample_row_ids(
        new_rows, n_sample_rows=min(n_sample_rows, len(new_rows)), seed=sample_seed
    )
    return RowIndex(rows=new_rows, shards=shards, sample_row_ids=sample_ids)


def _remap_values(old_values: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """旧因子值（按旧行序）→ 新行序（去掉 BSE 行）。"""
    return np.asarray(old_values[keep_mask], dtype=np.float32)


def migrate_library(root: Path, *, dry_run: bool) -> dict:
    paths = FactorLibraryPaths(root=root)
    if not paths.manifest.is_file():
        return {"root": str(root), "skipped": "no manifest"}

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    old_rows = pd.read_parquet(paths.rows_parquet)
    old_n = int(manifest["n_rows"])
    if len(old_rows) != old_n:
        raise RuntimeError(f"{root}: rows.parquet {len(old_rows)} != manifest n_rows {old_n}")

    inst = old_rows["instrument"].astype(str)
    bse_mask = _is_bse(inst)
    keep_mask = ~bse_mask.to_numpy()
    n_bse = int(bse_mask.sum())
    if n_bse == 0:
        return {"root": str(root), "bse_rows": 0, "skipped": "no BSE rows"}

    n_sample_rows = int(manifest.get("n_sample_rows", manifest.get("n_sketch", 200_000)))
    sample_seed = int(manifest.get("sample_seed", manifest.get("sketch_seed", 42)))
    new_index = _rebuild_index(old_rows, n_sample_rows, sample_seed)
    new_n = new_index.n_rows

    # 因子值重映射（production_main 有 memmap；candidate_main values/ 为空）
    value_files = sorted(paths.values_dir.glob("f_*.f32.memmap")) if paths.values_dir.is_dir() else []
    remapped: list[tuple[Path, np.ndarray]] = []
    for vf in value_files:
        file_n = vf.stat().st_size // np.dtype(np.float32).itemsize
        if file_n != old_n:
            raise RuntimeError(f"{vf.name}: memmap 长度 {file_n} != old_n {old_n}")
        old_vals = np.memmap(vf, dtype=np.float32, mode="r", shape=(old_n,))
        new_vals = _remap_values(old_vals, keep_mask)
        remapped.append((vf, new_vals))
        del old_vals

    if dry_run:
        return {
            "root": str(root),
            "old_n": old_n,
            "new_n": new_n,
            "bse_rows": n_bse,
            "value_files": len(remapped),
            "dry_run": True,
        }

    # 备份旧 index 与 manifest
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = root / f"index.bak-drop-bse-{ts}"
    bak.mkdir(parents=True, exist_ok=True)
    for f in ("rows.parquet", "sample_row_ids.parquet", "shards.json"):
        src = paths.index_dir / f
        if src.is_file():
            shutil.copy2(src, bak / f)
    shutil.copy2(paths.manifest, bak / "manifest.json")

    # 写新 index
    new_index.save(paths)

    # 重写因子值 memmap
    for vf, new_vals in remapped:
        arr = np.memmap(vf, dtype=np.float32, mode="w+", shape=(new_n,))
        arr[:] = new_vals
        arr.flush()
        del arr

    # 重建 sample 摘要 memmap：从新因子值按新 sample_row_ids 提取
    sample_meta_path = paths.resolve_sample_summary_meta()
    if sample_meta_path.is_file():
        meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
        col_map = {str(k): int(v) for k, v in meta.get("factor_id_to_col", {}).items()}
        max_factors = int(meta.get("max_factors", manifest.get("max_factors", 500)))
        summary_path = paths.resolve_sample_summary_memmap()
        if summary_path.is_file() and col_map:
            new_summary = np.full(
                (max_factors, new_index.n_sample_rows), np.nan, dtype=np.float32
            )
            # 每个因子：新 memmap 值 → 新 sample_row_ids 提取
            for fid, col_idx in col_map.items():
                vf = paths.factor_values_path(fid)
                if vf.is_file():
                    vals = np.memmap(vf, dtype=np.float32, mode="r", shape=(new_n,))
                    new_summary[col_idx, :] = vals[new_index.sample_row_ids]
                    del vals
            arr = np.memmap(summary_path, dtype=np.float32, mode="w+",
                            shape=(max_factors, new_index.n_sample_rows))
            arr[:] = new_summary
            arr.flush()
            del arr
            meta["n_sample_rows"] = new_index.n_sample_rows
            sample_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 更新 manifest
    manifest["n_rows"] = new_n
    manifest["n_sample_rows"] = new_index.n_sample_rows
    manifest["index_hash"] = index_content_hash(new_index.rows)
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "root": str(root),
        "old_n": old_n,
        "new_n": new_n,
        "bse_rows": n_bse,
        "value_files": len(remapped),
        "backup": str(bak),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="剔除 factorzoo 北交所行并重映射因子值")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    parser.add_argument(
        "--roots", nargs="*", default=None,
        help="因子库根目录（默认 production_main + candidate_main）",
    )
    args = parser.parse_args()

    if args.roots:
        roots = [Path(r) for r in args.roots]
    else:
        base = ROOT / "artifacts" / "alphaagent" / "factorzoo"
        roots = [base / "production_main", base / "candidate_main"]

    for root in roots:
        try:
            result = migrate_library(root, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {root}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()