"""Canonical row index：panel (datetime, instrument) → row_id。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphaagent.data.panel import load_panel
from alphaagent.factor.zoo.types import (
    DEFAULT_BAR_INTERVAL,
    DEFAULT_DATASET,
    FactorLibraryPaths,
    LibraryManifest,
    RowSlice,
    TimeShard,
)


def _panel_to_index_frame(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel 必须是 (datetime, instrument) MultiIndex")
    if panel.index.names[:2] != ["datetime", "instrument"]:
        raise ValueError("panel 索引层须为 datetime, instrument")
    panel = panel.sort_index()
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(panel.index.get_level_values("datetime")),
            "instrument": panel.index.get_level_values("instrument").astype(str),
        }
    )


def _assign_quarter_shard_id(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("Q").astype(str)


def build_row_index(df: pd.DataFrame) -> pd.DataFrame:
    """排序并分配 row_id、shard_id。"""
    d = df.sort_values(["datetime", "instrument"], kind="mergesort").reset_index(drop=True)
    d["row_id"] = np.arange(len(d), dtype=np.int64)
    d["shard_id"] = _assign_quarter_shard_id(d["datetime"])
    return d[["row_id", "datetime", "instrument", "shard_id"]]


def index_content_hash(rows: pd.DataFrame) -> str:
    payload = rows[["row_id", "datetime", "instrument"]].astype(
        {"datetime": str, "instrument": str, "row_id": int}
    )
    digest = hashlib.sha256(payload.to_csv(index=False).encode("utf-8")).hexdigest()
    return digest[:16]


def build_time_shards(rows: pd.DataFrame) -> list[TimeShard]:
    shards: list[TimeShard] = []
    for shard_id, grp in rows.groupby("shard_id", sort=True):
        start_row = int(grp["row_id"].min())
        stop_row = int(grp["row_id"].max()) + 1
        dt_min = grp["datetime"].min()
        dt_max = grp["datetime"].max()
        shards.append(
            TimeShard(
                shard_id=str(shard_id),
                start_row=start_row,
                stop_row=stop_row,
                datetime_start=str(dt_min),
                datetime_end=str(dt_max),
            )
        )
    return shards


def build_sample_row_ids(
    rows: pd.DataFrame,
    *,
    n_sample_rows: int,
    seed: int = 42,
) -> np.ndarray:
    """按 shard 比例分层抽样 row_id（固定种子，可复现）。"""
    n_rows = len(rows)
    if n_sample_rows <= 0:
        raise ValueError("n_sample_rows 须为正整数")
    if n_sample_rows >= n_rows:
        return rows["row_id"].to_numpy(dtype=np.int64, copy=True)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    shard_ids = rows["shard_id"].unique()
    counts = rows.groupby("shard_id", sort=True).size()
    total = int(counts.sum())
    quotas = (counts / total * n_sample_rows).round().astype(int)
    diff = int(n_sample_rows - quotas.sum())
    if diff != 0:
        idx = int(np.argmax(counts.to_numpy()))
        quotas.iloc[idx] = int(quotas.iloc[idx]) + diff

    for sid in shard_ids:
        q = int(quotas.loc[sid])
        if q <= 0:
            continue
        pool = rows.loc[rows["shard_id"] == sid, "row_id"].to_numpy(dtype=np.int64)
        if q >= len(pool):
            chosen.extend(pool.tolist())
        else:
            pick = rng.choice(pool, size=q, replace=False)
            chosen.extend(pick.tolist())

    out = np.array(sorted(set(chosen)), dtype=np.int64)
    if len(out) > n_sample_rows:
        out = np.sort(rng.choice(out, size=n_sample_rows, replace=False))
    elif len(out) < n_sample_rows:
        remaining = np.setdiff1d(rows["row_id"].to_numpy(dtype=np.int64), out, assume_unique=False)
        need = n_sample_rows - len(out)
        if len(remaining) >= need:
            extra = rng.choice(remaining, size=need, replace=False)
            out = np.sort(np.concatenate([out, extra]))
    return out


class RowIndex:
    def __init__(self, rows: pd.DataFrame, shards: list[TimeShard], sample_row_ids: np.ndarray) -> None:
        self.rows = rows
        self.shards = shards
        self.sample_row_ids = sample_row_ids.astype(np.int64, copy=False)
        self._by_shard = {s.shard_id: s for s in shards}

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_sample_rows(self) -> int:
        return len(self.sample_row_ids)

    def shard_for_id(self, shard_id: str) -> TimeShard | None:
        return self._by_shard.get(shard_id)

    def row_slice_for_dates(self, start: str | None, end: str | None) -> RowSlice:
        dt = pd.to_datetime(self.rows["datetime"], errors="coerce")
        mask = pd.Series(True, index=self.rows.index)
        if start is not None:
            mask &= dt >= pd.Timestamp(start)
        if end is not None:
            mask &= dt < pd.Timestamp(end) + pd.Timedelta(days=1)
        idx = self.rows.index[mask]
        if len(idx) == 0:
            return RowSlice(0, 0)
        rows_hit = self.rows.loc[idx, "row_id"]
        return RowSlice(int(rows_hit.min()), int(rows_hit.max()) + 1)

    def save(self, paths: FactorLibraryPaths) -> None:
        paths.index_dir.mkdir(parents=True, exist_ok=True)
        self.rows.to_parquet(paths.rows_parquet, index=False)
        self.sample_row_ids_df().to_parquet(paths.sample_row_ids, index=False)
        shard_payload = [
            {
                "shard_id": s.shard_id,
                "start_row": s.start_row,
                "stop_row": s.stop_row,
                "datetime_start": s.datetime_start,
                "datetime_end": s.datetime_end,
            }
            for s in self.shards
        ]
        paths.shards_json.write_text(json.dumps(shard_payload, indent=2), encoding="utf-8")

    def sample_row_ids_df(self) -> pd.DataFrame:
        return pd.DataFrame({"row_id": self.sample_row_ids})

    @classmethod
    def load(cls, paths: FactorLibraryPaths) -> RowIndex:
        rows = pd.read_parquet(paths.rows_parquet)
        sample_ids = pd.read_parquet(paths.resolve_sample_row_ids())["row_id"].to_numpy(
            dtype=np.int64
        )
        raw_shards = json.loads(paths.shards_json.read_text(encoding="utf-8"))
        shards = [
            TimeShard(
                shard_id=str(s["shard_id"]),
                start_row=int(s["start_row"]),
                stop_row=int(s["stop_row"]),
                datetime_start=str(s["datetime_start"]),
                datetime_end=str(s["datetime_end"]),
            )
            for s in raw_shards
        ]
        return cls(rows=rows, shards=shards, sample_row_ids=sample_ids)

    @classmethod
    def build_from_panel(
        cls,
        panel: pd.DataFrame,
        *,
        n_sample_rows: int = 200_000,
        sample_seed: int = 42,
    ) -> RowIndex:
        frame = _panel_to_index_frame(panel)
        rows = build_row_index(frame)
        shards = build_time_shards(rows)
        sample_ids = build_sample_row_ids(rows, n_sample_rows=n_sample_rows, seed=sample_seed)
        return cls(rows=rows, shards=shards, sample_row_ids=sample_ids)

    @classmethod
    def build_from_panel_path(
        cls,
        panel_path: Path,
        *,
        n_sample_rows: int = 200_000,
        sample_seed: int = 42,
    ) -> RowIndex:
        from alphaagent.data.adapters.cnequity import is_cne_source, load_panel_from_cne
        if is_cne_source(panel_path):
            panel = load_panel_from_cne(universe_mask=False)
        else:
            panel = load_panel(panel_path)
        return cls.build_from_panel(
            panel,
            n_sample_rows=n_sample_rows,
            sample_seed=sample_seed,
        )


def init_library(
    root: Path,
    *,
    panel: pd.DataFrame | None = None,
    panel_path: Path | None = None,
    dataset: str = DEFAULT_DATASET,
    bar_interval: str = DEFAULT_BAR_INTERVAL,
    n_sample_rows: int = 200_000,
    max_factors: int = 2048,
    sample_seed: int = 42,
) -> tuple[FactorLibraryPaths, LibraryManifest, RowIndex]:
    """初始化因子库目录、manifest、canonical index。"""
    if panel is None and panel_path is None:
        raise ValueError("必须提供 panel 或 panel_path")
    paths = FactorLibraryPaths(root=Path(root).expanduser().resolve())
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.sample_dir.mkdir(parents=True, exist_ok=True)
    paths.values_dir.mkdir(parents=True, exist_ok=True)
    paths.meta_dir.mkdir(parents=True, exist_ok=True)

    if panel is None:
        assert panel_path is not None
        from alphaagent.data.adapters.cnequity import CNE_SOURCE, is_cne_source
        if is_cne_source(panel_path):
            resolved_panel_path = Path(CNE_SOURCE)
        else:
            resolved_panel_path = Path(panel_path).expanduser().resolve()
        index = RowIndex.build_from_panel_path(
            resolved_panel_path,
            n_sample_rows=n_sample_rows,
            sample_seed=sample_seed,
        )
        panel_path_str = str(resolved_panel_path)
    else:
        index = RowIndex.build_from_panel(
            panel,
            n_sample_rows=n_sample_rows,
            sample_seed=sample_seed,
        )
        from alphaagent.data.adapters.cnequity import CNE_SOURCE, is_cne_source
        if panel_path and is_cne_source(panel_path):
            panel_path_str = CNE_SOURCE
        elif panel_path:
            panel_path_str = str(panel_path.resolve())
        else:
            panel_path_str = ""

    index.save(paths)

    manifest = LibraryManifest(
        dataset=dataset,
        bar_interval=bar_interval,
        universe_path=panel_path_str,
        n_rows=index.n_rows,
        n_sample_rows=index.n_sample_rows,
        max_factors=max_factors,
        index_hash=index_content_hash(index.rows),
        sample_seed=sample_seed,
        extra={
            "panel_path": panel_path_str,
            "base_interval": bar_interval,
        },
    )
    paths.manifest.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths, manifest, index


def verify_index_hash(manifest: LibraryManifest, index: RowIndex) -> None:
    """校验 manifest 与当前 index 一致（轻量 O(1) 校验）。

    早期实现会在每次 FactorZoo.open 时重算完整 content hash
    （index_content_hash → 1460 万行 to_csv + sha256，单库约 10 秒），
    导致因子库打开/入库/正交检查等路径每次冷加载都要卡十几秒。

    rows.parquet 与 manifest 是同一事务写入的（index.save + manifest 落盘成对出现），
    内容级失配只有人为篡改才会发生，且行数/样本数不变量足以抓出"文件缺失、
    被替换、被截断"这类实际损坏。完整 content hash 仍由 index_content_hash 提供，
    需要严格比对时（如 realign 维护流程）可显式调用。
    """
    if index.n_rows != manifest.n_rows:
        raise ValueError(
            f"index/manifest 行数不匹配: index={index.n_rows} manifest={manifest.n_rows}"
        )
    if index.n_sample_rows != manifest.n_sample_rows:
        raise ValueError(
            f"index/manifest 样本行数不匹配: index={index.n_sample_rows} "
            f"manifest={manifest.n_sample_rows}"
        )


def verify_index_prefix_stable(
    old_rows: pd.DataFrame,
    new_rows: pd.DataFrame,
    old_n: int,
) -> bool:
    """新 index 前 old_n 行是否与旧 index 完全一致（datetime, instrument 序）。"""
    if len(new_rows) < old_n or len(old_rows) < old_n:
        return False
    old_prefix = old_rows.iloc[:old_n][["datetime", "instrument"]].copy()
    new_prefix = new_rows.iloc[:old_n][["datetime", "instrument"]].copy()
    old_prefix["datetime"] = pd.to_datetime(old_prefix["datetime"], errors="coerce")
    new_prefix["datetime"] = pd.to_datetime(new_prefix["datetime"], errors="coerce")
    old_prefix["instrument"] = old_prefix["instrument"].astype(str)
    new_prefix["instrument"] = new_prefix["instrument"].astype(str)
    return old_prefix.reset_index(drop=True).equals(new_prefix.reset_index(drop=True))


def extend_library_index(
    lib_root: Path,
    *,
    panel: pd.DataFrame,
    panel_path: Path,
) -> RowIndex:
    """panel 尾部追加且前缀稳定时：扩展 index/manifest，保留 sample_row_ids。"""
    paths = FactorLibraryPaths(root=Path(lib_root).expanduser().resolve())
    if not paths.manifest.is_file():
        raise FileNotFoundError(f"因子库未初始化: {paths.manifest}")

    old_index = RowIndex.load(paths)
    old_n = old_index.n_rows
    panel = panel.sort_index()

    frame = _panel_to_index_frame(panel)
    new_rows = build_row_index(frame)
    if len(new_rows) <= old_n:
        raise ValueError(f"panel 行数 {len(new_rows)} 未大于库 n_rows {old_n}")

    if not verify_index_prefix_stable(old_index.rows, new_rows, old_n):
        raise ValueError("index 前缀不稳定，无法增量扩展")

    shards = build_time_shards(new_rows)
    new_index = RowIndex(
        rows=new_rows,
        shards=shards,
        sample_row_ids=old_index.sample_row_ids.copy(),
    )
    new_index.save(paths)

    manifest_data = json.loads(paths.manifest.read_text(encoding="utf-8"))
    panel_path_str = str(Path(panel_path).expanduser().resolve())
    manifest_data["n_rows"] = new_index.n_rows
    manifest_data["index_hash"] = index_content_hash(new_rows)
    manifest_data["panel_path"] = panel_path_str
    manifest_data["universe_path"] = panel_path_str
    paths.manifest.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return new_index
