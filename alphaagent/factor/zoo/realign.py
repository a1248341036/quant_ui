"""panel 行数变化时重建 factorlib index 并重算库内因子值。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.dsl.eval import collect_aux_intervals_from_expr
from alphaagent.factor.ingest import (
    compute_ingest_metrics,
    mask_values_before_start,
    materialize_slice_to_canonical,
    prepare_stored_values,
)
from alphaagent.factor.types import DEFAULT_INGEST_POLICY, IngestPolicy
from alphaagent.factor.zoo import FactorZoo, init_library
from alphaagent.factor.zoo.index import (
    RowIndex,
    build_row_index,
    build_time_shards,
    extend_library_index,
    verify_index_prefix_stable,
)
from alphaagent.factor.zoo.similarity import SimilarityMatrix
from alphaagent.factor.zoo.types import FactorMeta, LibraryManifest

AUX_WARMUP_CALENDAR_DAYS = 35
DEFAULT_WARMUP_DAYS = 240
DEFAULT_WARMUP_RETRY_DAYS = 480
DEFAULT_OVERLAP_VERIFY_DAYS = 20


def _resolve_panel_path(path: Path) -> Path:
    from alphaagent.data.adapters.cnequity import CNE_SOURCE
    # Path('cne://') 会被不同平台归一化成 'cne:'/'cne:/' 等变体；
    # 统一识别逻辑源并返回规范字符串路径，避免 manifest 比较误报不一致。
    s = str(path).replace("\\", "/").rstrip("/")
    if s.rstrip("/") == "cne:".rstrip("/") or f"{s}/" == CNE_SOURCE:
        return Path(CNE_SOURCE)
    return Path(path).expanduser().resolve()


def panel_paths_match(a: Path, b: Path) -> bool:
    return _resolve_panel_path(a) == _resolve_panel_path(b)


def _reset_similarity_matrix(zoo: FactorZoo) -> None:
    sim = SimilarityMatrix(zoo.paths, zoo.manifest.max_factors)
    if sim.matrix_path.is_file():
        sim.matrix_path.unlink()
    if sim.meta_path.is_file():
        sim.meta_path.unlink()


def _rebuild_similarity(zoo: FactorZoo) -> None:
    _reset_similarity_matrix(zoo)
    if zoo.n_factors == 0:
        return
    sim = SimilarityMatrix(zoo.paths, zoo.manifest.max_factors)
    for fid in zoo.catalog.list_factor_ids():
        meta = zoo.catalog.get(fid)
        if meta is None:
            continue
        values = zoo.read_factor(fid)
        sim.append_factor_correlations(
            zoo,
            factor_id=fid,
            col_idx=meta.col_idx,
            values=values,
        )


def _build_extended_index(zoo: FactorZoo, panel: pd.DataFrame) -> RowIndex:
    from alphaagent.factor.zoo.index import RowIndex, _panel_to_index_frame

    frame = _panel_to_index_frame(panel)
    new_rows = build_row_index(frame)
    if not verify_index_prefix_stable(zoo.index.rows, new_rows, zoo.manifest.n_rows):
        raise ValueError("index 前缀不稳定，无法增量扩展")
    shards = build_time_shards(new_rows)
    return RowIndex(
        rows=new_rows,
        shards=shards,
        sample_row_ids=zoo.index.sample_row_ids.copy(),
    )


def _zoo_for_eval(zoo: FactorZoo, extended_index: RowIndex) -> FactorZoo:
    manifest = LibraryManifest(
        dataset=zoo.manifest.dataset,
        bar_interval=zoo.manifest.bar_interval,
        universe_path=zoo.manifest.universe_path,
        n_rows=extended_index.n_rows,
        n_sample_rows=zoo.manifest.n_sample_rows,
        max_factors=zoo.manifest.max_factors,
        dtype=zoo.manifest.dtype,
        index_hash=zoo.manifest.index_hash,
        sample_seed=zoo.manifest.sample_seed,
        version=zoo.manifest.version,
        extra=dict(zoo.manifest.extra),
    )
    return FactorZoo(zoo.paths, manifest, extended_index, zoo.catalog)


def _candidate_rows_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.index.to_frame(index=False)
    frame.columns = ["datetime", "instrument"]
    return build_row_index(
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(frame["datetime"]),
                "instrument": frame["instrument"].astype(str),
            }
        )
    )


def _apply_stored_clip(values: np.ndarray, extra: dict[str, Any]) -> np.ndarray:
    p01 = extra.get("clip_p01")
    p99 = extra.get("clip_p99")
    if p01 is None or p99 is None:
        return np.asarray(values, dtype=np.float32)
    out = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    out[finite] = np.clip(out[finite], float(p01), float(p99)).astype(np.float32, copy=False)
    return out


def eval_window_dates(
    index_rows: pd.DataFrame,
    *,
    old_n: int,
    warmup_days: int,
    expr: str,
) -> tuple[str, str, pd.Timestamp, pd.Timestamp]:
    """返回 (slice_start_str, slice_end_str, warmup_start_ts, update_start_ts)。"""
    if old_n >= len(index_rows):
        raise ValueError(f"old_n={old_n} >= index 行数 {len(index_rows)}")

    dt_col = pd.to_datetime(index_rows["datetime"], errors="coerce")
    update_start = pd.Timestamp(dt_col.iloc[old_n])
    slice_end = pd.Timestamp(dt_col.max())

    trade_days = pd.Series(dt_col.unique()).sort_values().reset_index(drop=True)
    pos = int(trade_days.searchsorted(update_start))
    warmup_idx = max(0, pos - int(warmup_days))
    warmup_start = pd.Timestamp(trade_days.iloc[warmup_idx])

    if collect_aux_intervals_from_expr(expr):
        warmup_start = warmup_start - pd.Timedelta(days=AUX_WARMUP_CALENDAR_DAYS)

    return (
        warmup_start.strftime("%Y-%m-%d"),
        slice_end.strftime("%Y-%m-%d"),
        warmup_start,
        update_start,
    )


def overlap_row_ids(
    index_rows: pd.DataFrame,
    *,
    old_n: int,
    update_start: pd.Timestamp,
    overlap_verify_days: int = DEFAULT_OVERLAP_VERIFY_DAYS,
) -> tuple[np.ndarray, pd.Timestamp]:
    """update 前最后 overlap_verify_days 个交易日的旧库 row_id（不含 update_start 当日）。"""
    dt = pd.to_datetime(index_rows["datetime"], errors="coerce")
    trade_days = pd.Series(dt.unique()).sort_values().reset_index(drop=True)
    pos = int(trade_days.searchsorted(update_start))
    start_idx = max(0, pos - int(overlap_verify_days))
    verify_start = pd.Timestamp(trade_days.iloc[start_idx]) if pos > 0 else update_start

    mask = (
        (index_rows["row_id"].to_numpy(dtype=np.int64) < old_n)
        & (dt >= verify_start)
        & (dt < update_start)
    )
    return index_rows.loc[mask, "row_id"].to_numpy(dtype=np.int64), verify_start


def verify_overlap_exact(
    stored: np.ndarray,
    computed: np.ndarray,
    overlap_ids: np.ndarray,
    *,
    index_rows: pd.DataFrame,
    max_samples: int = 5,
) -> tuple[bool, dict[str, Any]]:
    if len(overlap_ids) == 0:
        return True, {"n_overlap": 0, "n_mismatch": 0, "samples": []}

    stored_v = np.asarray(stored, dtype=np.float32)
    computed_v = np.asarray(computed, dtype=np.float32)
    mismatches: list[dict[str, Any]] = []
    n_mismatch = 0

    for rid in overlap_ids:
        rid_int = int(rid)
        s = stored_v[rid_int]
        c = computed_v[rid_int]
        s_fin = np.isfinite(s)
        c_fin = np.isfinite(c)
        ok = (s_fin and c_fin and s == c) or (not s_fin and not c_fin)
        if not ok:
            n_mismatch += 1
            if len(mismatches) < max_samples:
                row = index_rows.loc[index_rows["row_id"] == rid_int].iloc[0]
                mismatches.append(
                    {
                        "row_id": rid_int,
                        "datetime": str(row["datetime"]),
                        "instrument": str(row["instrument"]),
                        "stored": float(s) if s_fin else None,
                        "computed": float(c) if c_fin else None,
                    }
                )

    return n_mismatch == 0, {
        "n_overlap": int(len(overlap_ids)),
        "n_mismatch": n_mismatch,
        "samples": mismatches,
    }


def _rematerialize_factor(
    zoo: FactorZoo,
    meta: FactorMeta,
    panel: pd.DataFrame,
    policy: IngestPolicy,
) -> None:
    values_path = zoo.paths.factor_values_path(meta.factor_id)
    if values_path.is_file():
        values_path.unlink()
    stored_values, expr, aux_tags, clip_extra = prepare_stored_values(meta.expr, panel, zoo, policy)
    metrics = compute_ingest_metrics(stored_values, panel, policy)
    extra = dict(meta.extra or {})
    extra.update({**clip_extra, "aux_tags": aux_tags, "metrics": metrics})
    zoo.overwrite_factor(
        factor_id=meta.factor_id,
        name=meta.name,
        expr=expr,
        values=stored_values,
        status=meta.status,
        extra=extra,
    )


def _prepare_computed_values(
    meta: FactorMeta,
    panel: pd.DataFrame,
    zoo: FactorZoo,
    policy: IngestPolicy,
    *,
    old_n: int,
    warmup_days: int,
    overlap_verify_days: int = DEFAULT_OVERLAP_VERIFY_DAYS,
) -> tuple[np.ndarray, dict[str, Any]]:
    slice_start, slice_end, warmup_start, update_start = eval_window_dates(
        zoo.index.rows,
        old_n=old_n,
        warmup_days=warmup_days,
        expr=meta.expr,
    )
    mat = materialize_slice_to_canonical(
        meta.expr,
        panel,
        zoo,
        start=slice_start,
        end=slice_end,
    )
    values = _apply_stored_clip(mat.values, meta.extra or {})
    values = mask_values_before_start(values, zoo, policy.mask_before_start)
    overlap_ids, verify_start = overlap_row_ids(
        zoo.index.rows,
        old_n=old_n,
        update_start=update_start,
        overlap_verify_days=overlap_verify_days,
    )
    return values, {
        "slice_start": slice_start,
        "slice_end": slice_end,
        "warmup_start": str(warmup_start),
        "update_start": str(update_start),
        "warmup_days": warmup_days,
        "overlap_verify_days": overlap_verify_days,
        "overlap_verify_start": str(verify_start),
        "n_overlap": int(len(overlap_ids)),
        "_overlap_ids": overlap_ids,
    }


def remask_factorlib(
    lib_root: Path,
    *,
    panel: pd.DataFrame,
    policy: IngestPolicy | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """对库内全部因子按 policy 重新物化、mask 并重算 extra.metrics。"""
    pol = policy or DEFAULT_INGEST_POLICY
    lib_root = Path(lib_root).expanduser().resolve()
    panel = panel.sort_index()
    zoo = FactorZoo.open(lib_root)
    remasked: list[str] = []
    for fid in zoo.catalog.list_factor_ids():
        meta = zoo.catalog.get(fid)
        if meta is None:
            continue
        remasked.append(fid)
        if dry_run:
            continue
        _rematerialize_factor(zoo, meta, panel, pol)
    if not dry_run and remasked:
        _rebuild_similarity(zoo)
    return {
        "remasked_factors": remasked,
        "n_factors": len(remasked),
        "dry_run": dry_run,
        "ingest_policy": pol.to_dict(),
    }


def realign_factorlib_to_panel(
    lib_root: Path,
    *,
    panel: pd.DataFrame,
    panel_path: Path,
    policy: IngestPolicy | None = None,
) -> dict[str, Any]:
    """同一 panel 路径下行数变化：重建 manifest/index，并重算全部已有因子 memmap。"""
    pol = policy or DEFAULT_INGEST_POLICY
    lib_root = Path(lib_root).expanduser().resolve()
    panel_path = _resolve_panel_path(panel_path)
    panel = panel.sort_index()

    zoo = FactorZoo.open(lib_root, verify_hash=False)
    old_n_rows = zoo.manifest.n_rows
    if len(panel) == old_n_rows:
        return {"realigned": False, "mode": "noop", "n_rows": old_n_rows}

    saved: list[FactorMeta] = []
    for fid in zoo.catalog.list_factor_ids():
        meta = zoo.catalog.get(fid)
        if meta is not None:
            saved.append(meta)

    init_library(
        lib_root,
        panel=panel,
        panel_path=panel_path,
        n_sample_rows=min(zoo.manifest.n_sample_rows, len(panel)),
        max_factors=zoo.manifest.max_factors,
        sample_seed=zoo.manifest.sample_seed,
    )

    zoo = FactorZoo.open(lib_root, verify_hash=True)
    _reset_similarity_matrix(zoo)

    rematerialized: list[str] = []
    for meta in sorted(saved, key=lambda m: m.col_idx):
        _rematerialize_factor(zoo, meta, panel, pol)
        rematerialized.append(meta.factor_id)

    _rebuild_similarity(zoo)

    return {
        "realigned": True,
        "mode": "full",
        "old_n_rows": old_n_rows,
        "new_n_rows": len(panel),
        "panel_path": str(panel_path),
        "rematerialized_factors": rematerialized,
        "n_factors": len(rematerialized),
        "ingest_policy": pol.to_dict(),
    }


def incremental_realign_factorlib_to_panel(
    lib_root: Path,
    *,
    panel: pd.DataFrame,
    panel_path: Path,
    policy: IngestPolicy | None = None,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    warmup_retry_days: int = DEFAULT_WARMUP_RETRY_DAYS,
    overlap_verify_days: int = DEFAULT_OVERLAP_VERIFY_DAYS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """panel 尾部追加：T+N 窗口重算 + overlap 校验；失败则扩窗或全量 fallback。"""
    pol = policy or DEFAULT_INGEST_POLICY
    lib_root = Path(lib_root).expanduser().resolve()
    panel_path = _resolve_panel_path(panel_path)
    panel = panel.sort_index()

    zoo = FactorZoo.open(lib_root, verify_hash=False)
    old_n = zoo.manifest.n_rows
    new_n = len(panel)

    if new_n == old_n:
        return {
            "realigned": False,
            "mode": "noop",
            "n_rows": old_n,
            "dry_run": dry_run,
        }

    if not panel_paths_match(panel_path, Path(zoo.manifest.panel_path)):
        return {
            **realign_factorlib_to_panel(
                lib_root, panel=panel, panel_path=panel_path, policy=pol
            ),
            "mode": "full",
            "fallback_reason": "panel_path_mismatch",
        }

    candidate_rows = _candidate_rows_from_panel(panel)
    if not verify_index_prefix_stable(zoo.index.rows, candidate_rows, old_n):
        return {
            **realign_factorlib_to_panel(
                lib_root, panel=panel, panel_path=panel_path, policy=pol
            ),
            "mode": "full",
            "fallback_reason": "index_prefix_unstable",
        }

    saved: list[FactorMeta] = []
    for fid in zoo.catalog.list_factor_ids():
        meta = zoo.catalog.get(fid)
        if meta is not None:
            saved.append(meta)

    old_values_map: dict[str, np.ndarray] = {}
    for meta in saved:
        old_values_map[meta.factor_id] = zoo.read_factor(meta.factor_id)

    extended_index = _build_extended_index(zoo, panel)
    eval_zoo = _zoo_for_eval(zoo, extended_index)

    if not dry_run:
        extend_library_index(lib_root, panel=panel, panel_path=panel_path)
        zoo = FactorZoo.open(lib_root, verify_hash=True)
        _reset_similarity_matrix(zoo)
        eval_zoo = zoo

    factor_reports: dict[str, Any] = {}
    incremental_factors: list[str] = []
    fallback_factors: list[str] = []

    for meta in sorted(saved, key=lambda m: m.col_idx):
        fid = meta.factor_id
        stored_old = old_values_map[fid]
        ok = False
        used_warmup = warmup_days
        last_report: dict[str, Any] = {}

        for attempt_warmup in (warmup_days, warmup_retry_days):
            computed, win_info = _prepare_computed_values(
                meta,
                panel,
                eval_zoo,
                pol,
                old_n=old_n,
                warmup_days=attempt_warmup,
                overlap_verify_days=overlap_verify_days,
            )
            overlap_ids = win_info.pop("_overlap_ids")
            passed, overlap_report = verify_overlap_exact(
                stored_old,
                computed,
                overlap_ids,
                index_rows=eval_zoo.index.rows,
            )
            last_report = {**win_info, **overlap_report, "passed": passed}
            if passed:
                ok = True
                used_warmup = attempt_warmup
                break

        if ok:
            factor_reports[fid] = {
                "strategy": "incremental",
                "warmup_days": used_warmup,
                **last_report,
            }
            incremental_factors.append(fid)
            if not dry_run:
                tail = computed[old_n:new_n]
                metrics = compute_ingest_metrics(
                    np.concatenate([stored_old, tail]),
                    panel,
                    pol,
                )
                extra = dict(meta.extra or {})
                extra["metrics"] = metrics
                extra["incremental_realign"] = {
                    "warmup_days": used_warmup,
                    "slice_start": last_report.get("slice_start"),
                    "slice_end": last_report.get("slice_end"),
                }
                zoo.extend_factor_values(
                    fid,
                    tail,
                    old_n=old_n,
                    extra=extra,
                )
            continue

        factor_reports[fid] = {
            "strategy": "full_fallback",
            "warmup_days_tried": [warmup_days, warmup_retry_days],
            **last_report,
        }
        fallback_factors.append(fid)
        if not dry_run:
            _rematerialize_factor(zoo, meta, panel, pol)

    if not dry_run and (incremental_factors or fallback_factors):
        _rebuild_similarity(zoo)

    return {
        "realigned": True,
        "mode": "incremental" if not fallback_factors else "incremental_with_fallback",
        "dry_run": dry_run,
        "old_n_rows": old_n,
        "new_n_rows": new_n,
        "panel_path": str(panel_path),
        "n_factors": len(saved),
        "incremental_factors": incremental_factors,
        "fallback_factors": fallback_factors,
        "factor_reports": factor_reports,
        "overlap_verify_days": overlap_verify_days,
        "ingest_policy": pol.to_dict(),
    }


def list_append_boundary_old_n(
    index_rows: pd.DataFrame,
    *,
    append_trade_days: list[int] | None = None,
) -> list[dict[str, Any]]:
    """按「尾部追加 K 个交易日」生成滚动测试点（old_n = 该日首行 row_id）。"""
    append_trade_days = append_trade_days or [1, 2, 3, 5, 10, 20]
    dt = pd.to_datetime(index_rows["datetime"], errors="coerce")
    trade_days = pd.Series(dt.unique()).sort_values().tolist()
    n_rows = len(index_rows)
    points: list[dict[str, Any]] = []

    for k in append_trade_days:
        if k <= 0 or k >= len(trade_days):
            continue
        update_start = pd.Timestamp(trade_days[-k])
        hit = index_rows.loc[dt >= update_start, "row_id"]
        if hit.empty:
            continue
        old_n = int(hit.min())
        if old_n <= 0 or old_n >= n_rows:
            continue
        points.append(
            {
                "old_n": old_n,
                "new_n": n_rows,
                "append_rows": n_rows - old_n,
                "append_trade_days": k,
                "update_start": update_start.strftime("%Y-%m-%d"),
            }
        )
    return points


def _probe_factors_at_old_n(
    zoo: FactorZoo,
    panel: pd.DataFrame,
    saved: list[FactorMeta],
    old_values_map: dict[str, np.ndarray],
    *,
    old_n: int,
    pol: IngestPolicy,
    warmup_days: int,
    warmup_retry_days: int,
    overlap_verify_days: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    incremental_factors: list[str] = []
    fallback_factors: list[str] = []
    factor_reports: dict[str, Any] = {}

    for meta in sorted(saved, key=lambda m: m.col_idx):
        fid = meta.factor_id
        stored_old = old_values_map[fid]
        if len(stored_old) < old_n:
            factor_reports[fid] = {"strategy": "skip", "reason": "stored_shorter_than_old_n"}
            continue

        ok = False
        used_warmup = warmup_days
        last_report: dict[str, Any] = {}

        for attempt_warmup in (warmup_days, warmup_retry_days):
            computed, win_info = _prepare_computed_values(
                meta,
                panel,
                zoo,
                pol,
                old_n=old_n,
                warmup_days=attempt_warmup,
                overlap_verify_days=overlap_verify_days,
            )
            overlap_ids = win_info.pop("_overlap_ids")
            passed, overlap_report = verify_overlap_exact(
                stored_old[:old_n],
                computed,
                overlap_ids,
                index_rows=zoo.index.rows,
            )
            last_report = {**win_info, **overlap_report, "passed": passed}
            if passed:
                ok = True
                used_warmup = attempt_warmup
                break

        if ok:
            factor_reports[fid] = {
                "strategy": "incremental",
                "warmup_days": used_warmup,
                **last_report,
            }
            incremental_factors.append(fid)
        else:
            factor_reports[fid] = {
                "strategy": "would_fallback",
                "warmup_days_tried": [warmup_days, warmup_retry_days],
                **last_report,
            }
            fallback_factors.append(fid)

    return incremental_factors, fallback_factors, factor_reports


def probe_incremental_realign_at_old_n(
    zoo: FactorZoo,
    panel: pd.DataFrame,
    *,
    old_n: int,
    policy: IngestPolicy | None = None,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    warmup_retry_days: int = DEFAULT_WARMUP_RETRY_DAYS,
    overlap_verify_days: int = DEFAULT_OVERLAP_VERIFY_DAYS,
) -> dict[str, Any]:
    """模拟从 old_n 增至 len(panel) 的增量 realign（只读，不写盘）。"""
    pol = policy or DEFAULT_INGEST_POLICY
    panel = panel.sort_index()
    new_n = len(panel)
    if old_n <= 0 or old_n >= new_n:
        raise ValueError(f"old_n 须满足 0 < old_n < new_n，得到 old_n={old_n} new_n={new_n}")

    if new_n != zoo.manifest.n_rows:
        raise ValueError(
            f"probe 要求 len(panel)={new_n} == manifest.n_rows={zoo.manifest.n_rows}"
        )

    saved: list[FactorMeta] = []
    for fid in zoo.catalog.list_factor_ids():
        meta = zoo.catalog.get(fid)
        if meta is not None:
            saved.append(meta)

    old_values_map = {m.factor_id: zoo.read_factor(m.factor_id) for m in saved}
    inc, fb, reports = _probe_factors_at_old_n(
        zoo,
        panel,
        saved,
        old_values_map,
        old_n=old_n,
        pol=pol,
        warmup_days=warmup_days,
        warmup_retry_days=warmup_retry_days,
        overlap_verify_days=overlap_verify_days,
    )

    dt_col = pd.to_datetime(zoo.index.rows["datetime"], errors="coerce")
    update_start = str(dt_col.iloc[old_n])

    return {
        "simulated": True,
        "old_n_rows": old_n,
        "new_n_rows": new_n,
        "append_rows": new_n - old_n,
        "update_start": update_start,
        "n_factors": len(saved),
        "incremental_factors": inc,
        "fallback_factors": fb,
        "factor_reports": reports,
        "overlap_verify_days": overlap_verify_days,
        "warmup_days": warmup_days,
        "warmup_retry_days": warmup_retry_days,
    }


def rolling_probe_incremental_realign(
    lib_root: Path,
    *,
    panel: pd.DataFrame,
    append_trade_days: list[int] | None = None,
    policy: IngestPolicy | None = None,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    warmup_retry_days: int = DEFAULT_WARMUP_RETRY_DAYS,
    overlap_verify_days: int = DEFAULT_OVERLAP_VERIFY_DAYS,
    factor_ids: list[str] | None = None,
) -> dict[str, Any]:
    """在多个 append 边界上滚动 probe（只读）。"""
    pol = policy or DEFAULT_INGEST_POLICY
    panel = panel.sort_index()
    zoo = FactorZoo.open(lib_root, verify_hash=False)

    if len(panel) != zoo.manifest.n_rows:
        raise ValueError(
            f"panel 行数 {len(panel)} != manifest.n_rows {zoo.manifest.n_rows}；"
            "滚动 probe 需在 panel 与库已对齐后进行"
        )

    if factor_ids:
        keep = set(factor_ids)
        saved_ids = [fid for fid in zoo.catalog.list_factor_ids() if fid in keep]
        if not saved_ids:
            raise ValueError(f"库中无指定因子: {factor_ids}")
    else:
        saved_ids = None

    points = list_append_boundary_old_n(zoo.index.rows, append_trade_days=append_trade_days)
    windows: list[dict[str, Any]] = []

    for pt in points:
        result = probe_incremental_realign_at_old_n(
            zoo,
            panel,
            old_n=int(pt["old_n"]),
            policy=pol,
            warmup_days=warmup_days,
            warmup_retry_days=warmup_retry_days,
            overlap_verify_days=overlap_verify_days,
        )
        if saved_ids is not None:
            result["incremental_factors"] = [
                f for f in result["incremental_factors"] if f in saved_ids
            ]
            result["fallback_factors"] = [
                f for f in result["fallback_factors"] if f in saved_ids
            ]
            result["factor_reports"] = {
                k: v for k, v in result["factor_reports"].items() if k in saved_ids
            }
            result["n_factors"] = len(result["factor_reports"])

        windows.append({**pt, **result})

    return {
        "lib": str(Path(lib_root).resolve()),
        "n_rows": zoo.manifest.n_rows,
        "n_factors": zoo.n_factors,
        "append_trade_days": append_trade_days or [1, 2, 3, 5, 10, 20],
        "overlap_verify_days": overlap_verify_days,
        "warmup_days": warmup_days,
        "warmup_retry_days": warmup_retry_days,
        "windows": windows,
    }
