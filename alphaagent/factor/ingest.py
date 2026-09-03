"""因子物化、评估与入库编排。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.data.panel import load_panel, slice_panel
from alphaagent.dsl import eval_factor
from alphaagent.dsl.eval import collect_aux_intervals_from_expr
from alphaagent.factor.align import align_series_to_panel, canonical_align
from alphaagent.factor.metrics import coverage as metric_coverage
from alphaagent.factor.metrics import (
    annualized_long_group_excess_return,
    cross_sectional_winsorize_values,
    evaluate_on_panel,
    factor_skew_kurtosis,
)
from alphaagent.factor.types import DEFAULT_INGEST_POLICY, IngestPolicy, IngestResult, MaterializeResult
from alphaagent.factor.zoo import FactorStatus, FactorZoo, SimilarityMatrix


def clip_values(
    values: np.ndarray,
    *,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """按分位数 clip，返回 clip 后向量与 extra 元数据。"""
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.sum() <= 1:
        return arr.copy(), {
            "clip_p01": None,
            "clip_p99": None,
            "clip_lower_pct": lower_pct,
            "clip_upper_pct": upper_pct,
        }
    raw = arr[finite]
    p01, p99 = np.nanpercentile(raw, [lower_pct, upper_pct])
    out = arr.copy()
    out[finite] = np.clip(raw, p01, p99).astype(np.float32, copy=False)
    return out, {
        "clip_p01": float(p01) if np.isfinite(p01) else None,
        "clip_p99": float(p99) if np.isfinite(p99) else None,
        "clip_lower_pct": lower_pct,
        "clip_upper_pct": upper_pct,
    }



def materialize_factor(expr: str, panel: pd.DataFrame, *, cache=None) -> MaterializeResult:
    """DSL 求值并对齐 panel 行序。"""
    panel = panel.sort_index()

    def _eval():
        return eval_factor(expr, panel)

    if cache is not None:
        out = cache.evaluate(expr, panel, _eval)
    else:
        out = _eval()
    if not isinstance(out, pd.Series):
        raise TypeError(f"因子输出须为 Series，得到 {type(out)!r}")
    values = align_series_to_panel(out, panel)
    tags = collect_aux_intervals_from_expr(expr)
    return MaterializeResult(values=values, expr=expr.strip(), aux_tags=tags)


def align_values_to_rows(values_by_key: pd.Series, rows: pd.DataFrame) -> np.ndarray:
    """按 (datetime, instrument) 把会话域因子值重排到目标行序（缺失 → NaN）。

    用于 submit 的会话域复用：挖掘 panel 与因子库 canonical 行集不必相同，
    相似度抽样与入库前对齐都通过本函数完成键映射。
    """
    src_dt = pd.to_datetime(values_by_key.index.get_level_values("datetime"))
    src_inst = values_by_key.index.get_level_values("instrument").astype(str)
    src = pd.Series(
        np.asarray(values_by_key, dtype=np.float32),
        index=pd.MultiIndex.from_arrays([src_dt, src_inst], names=["datetime", "instrument"]),
    )
    tgt_dt = pd.to_datetime(rows["datetime"], errors="coerce")
    tgt_inst = rows["instrument"].astype(str)
    target = pd.MultiIndex.from_arrays([tgt_dt, tgt_inst], names=["datetime", "instrument"])
    aligned = src.reindex(target)
    return aligned.to_numpy(dtype=np.float32)


def materialize_to_canonical(
    expr: str,
    panel: pd.DataFrame,
    zoo: FactorZoo,
) -> MaterializeResult:
    """求值并对齐到因子库 canonical row_id。"""
    panel = panel.sort_index()
    out = eval_factor(expr, panel)
    if not isinstance(out, pd.Series):
        raise TypeError(f"因子输出须为 Series，得到 {type(out)!r}")
    aligned = canonical_align(
        out,
        row_index=zoo.index,
        n_rows=zoo.manifest.n_rows,
    )
    tags = collect_aux_intervals_from_expr(expr)
    return MaterializeResult(values=aligned, expr=expr.strip(), aux_tags=tags)


def materialize_slice_to_canonical(
    expr: str,
    panel: pd.DataFrame,
    zoo: FactorZoo,
    *,
    start: str,
    end: str,
) -> MaterializeResult:
    """在 panel 日期子集上求值，并对齐到完整 canonical index（窗口外为 NaN）。"""
    panel = panel.sort_index()
    panel_slice = slice_panel(panel, start=start, end=end)
    if panel_slice.empty:
        raise ValueError(f"panel 切片为空: start={start!r} end={end!r}")
    out = eval_factor(expr, panel_slice)
    if not isinstance(out, pd.Series):
        raise TypeError(f"因子输出须为 Series，得到 {type(out)!r}")
    aligned = canonical_align(
        out,
        row_index=zoo.index,
        n_rows=zoo.manifest.n_rows,
    )
    tags = collect_aux_intervals_from_expr(expr)
    return MaterializeResult(values=aligned, expr=expr.strip(), aux_tags=tags)


def mask_values_before_start(values: np.ndarray, zoo: FactorZoo, start: str) -> np.ndarray:
    """将 datetime < start 的行置为 NaN，从 start（含）起保留入库值。"""
    out = np.array(values, dtype=np.float32, copy=True)
    rows = zoo.index.rows
    dt = pd.to_datetime(rows["datetime"], errors="coerce")
    cutoff = pd.Timestamp(start)
    out[dt < cutoff] = np.nan
    return out


def compute_ingest_metrics(
    stored_values: np.ndarray,
    panel: pd.DataFrame,
    policy: IngestPolicy,
) -> dict[str, Any]:
    """按 policy 在 eval 区间计算入库指标；coverage 为 eval 区间口径。"""
    panel_sorted = panel.sort_index()
    eval_values, eval_panel = _slice_values_with_panel(
        stored_values,
        panel_sorted,
        start=policy.eval_start,
        end=policy.eval_end,
    )
    # label 名义持有期（label_20d → 20）：ICIR 按持有期重采样去重叠、超额年化
    # 按持有期缩放，避免长持有期因子指标虚高。label_1d 时 hold=1 退化为原行为。
    label_digits = "".join(ch for ch in str(policy.label_col) if ch.isdigit())
    holding_days = max(1, int(label_digits) if label_digits else 1)
    metrics = evaluate_on_panel(
        eval_values, eval_panel, label_col=policy.label_col, holding_days=holding_days
    )
    winsorized_values = cross_sectional_winsorize_values(eval_values, eval_panel)
    winsorized_metrics = evaluate_on_panel(
        winsorized_values, eval_panel, label_col=policy.label_col, holding_days=holding_days
    )
    raw_abs_ic = abs(float(metrics["ic"]))
    winsorized_abs_ic = abs(float(winsorized_metrics["ic"]))
    metrics["winsorized_ic"] = winsorized_metrics["ic"]
    metrics["winsorized_abs_ic_decay"] = (
        max(0.0, (raw_abs_ic - winsorized_abs_ic) / raw_abs_ic)
        if np.isfinite(raw_abs_ic) and raw_abs_ic > 0.0 and np.isfinite(winsorized_abs_ic)
        else float("nan")
    )
    factor_series = pd.Series(eval_values, index=eval_panel.index, dtype=np.float32)
    metrics["long_group_annual_excess_return"] = annualized_long_group_excess_return(
        factor_series,
        eval_panel[policy.label_col],
        direction=1 if float(metrics["ic"]) >= 0.0 else -1,
        holding_days=holding_days,
    )
    skew, kurt = factor_skew_kurtosis(eval_values)
    metrics["skew"] = skew
    metrics["kurt"] = kurt
    metrics["eval_start"] = policy.eval_start
    metrics["eval_end"] = policy.eval_end
    metrics["finite_ratio"] = metric_coverage(stored_values)
    return metrics


def prepare_stored_values(
    expr: str,
    panel: pd.DataFrame,
    zoo: FactorZoo,
    policy: IngestPolicy,
) -> tuple[np.ndarray, str, list[str], dict[str, Any]]:
    """物化 → 可选 clip → mask → 返回 (stored_values, expr, aux_tags, clip_extra)。"""
    mat = materialize_to_canonical(expr, panel, zoo)
    if policy.clip_pct is not None:
        stored_values, clip_extra = clip_values(
            mat.values, lower_pct=policy.clip_pct[0], upper_pct=policy.clip_pct[1]
        )
    else:
        stored_values = mat.values
        clip_extra = {"clip_lower_pct": None, "clip_upper_pct": None}
    stored_values = mask_values_before_start(stored_values, zoo, policy.mask_before_start)
    return stored_values, mat.expr, mat.aux_tags, clip_extra


def _slice_values_with_panel(
    values: np.ndarray,
    panel: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """按日期切片 panel，并取对应位置的因子值。"""
    panel = panel.sort_index()
    sub = slice_panel(panel, start=start, end=end)
    if sub.empty:
        raise ValueError(f"panel 切片为空: start={start!r} end={end!r}")
    pos = panel.index.isin(sub.index)
    return values[pos], sub


def ingest_factor(
    zoo: FactorZoo,
    *,
    factor_id: str,
    name: str,
    expr: str,
    panel: pd.DataFrame | None = None,
    policy: IngestPolicy | None = None,
    stored_values: np.ndarray | None = None,
    metrics_override: dict[str, Any] | None = None,
    label_col: str = DEFAULT_INGEST_POLICY.label_col,
    clip_pct: tuple[float, float] | None = None,
    mask_before_start: str | None = None,
    eval_start: str | None = None,
    eval_end: str | None = None,
    max_cs_corr: float = DEFAULT_INGEST_POLICY.max_cs_corr,
    similar_top_k: int = DEFAULT_INGEST_POLICY.similar_top_k,
    similar_min_pairs: int = 10,
    overwrite: bool = False,
    dry_run: bool = False,
    update_similarity: bool = True,
    compute_similarity: bool = True,
) -> IngestResult:
    """单因子：物化 → [可选 clip] → mask → eval 指标 → 查重 → 入库。

    ``compute_similarity=False`` 跳过与因子库的相似度对比（仅允许配合
    ``dry_run=True`` 使用）：调用方可在统计门槛前置检查通过后再单独计算，
    避免 IC 不达标的候选白算最贵的相似度循环。

    两种调用形态：
    - 常规：传 ``panel``，内部物化并计算指标；
    - 会话域复用：传 canonical 长度的 ``stored_values`` + ``metrics_override``
      （panel 可为 None），跳过重复物化与指标计算。
    """
    if not compute_similarity and not dry_run:
        raise ValueError("compute_similarity=False 仅支持 dry_run=True")
    if stored_values is not None and panel is None and metrics_override is None:
        raise ValueError("stored_values 不带 panel 时必须提供 metrics_override")
    if policy is not None:
        pol = policy
    else:
        train = mask_before_start or eval_start or DEFAULT_INGEST_POLICY.train_start
        pol = IngestPolicy(
            train_start=train,
            val_end=eval_end or DEFAULT_INGEST_POLICY.val_end,
            label_col=label_col,
            max_cs_corr=max_cs_corr,
            similar_top_k=similar_top_k,
            clip_pct=clip_pct,
        )

    if stored_values is None:
        stored_values, mat_expr, aux_tags, clip_extra = prepare_stored_values(expr, panel, zoo, pol)
    else:
        stored_values = np.asarray(stored_values, dtype=np.float32)
        mat_expr = expr
        aux_tags = collect_aux_intervals_from_expr(expr)
        clip_extra = {"clip_lower_pct": None, "clip_upper_pct": None}
    if metrics_override is not None:
        metrics = dict(metrics_override)
    else:
        metrics = compute_ingest_metrics(stored_values, panel, pol)
    extra: dict[str, Any] = {
        **clip_extra,
        "aux_tags": aux_tags,
        "metrics": metrics,
    }

    sim: SimilarityMatrix | None = None
    sim_info: dict[str, Any] | None = None
    existing = zoo.catalog.get(factor_id)

    if existing is None:
        max_corr = 0.0
        neighbor_report: dict[str, Any] | None = None
        if compute_similarity and zoo.n_factors > 0:
            sim = SimilarityMatrix(zoo.paths, zoo.manifest.max_factors)
            neighbor_report = sim.cross_sectional_neighbor_report(
                zoo, stored_values, top_k=pol.similar_top_k, min_pairs=similar_min_pairs
            )
            max_corr = float(neighbor_report["max_abs_corr"])
        if max_corr >= pol.max_cs_corr:
            return IngestResult(
                factor_id=factor_id,
                col_idx=None,
                stored=False,
                skipped_reason=f"cs_corr={max_corr:.4f} >= {pol.max_cs_corr}",
                metrics=metrics,
                similarity=neighbor_report or {"max_abs_corr": 0.0, "top_neighbors": []},
                extra=extra,
            )
        if dry_run:
            return IngestResult(
                factor_id=factor_id,
                col_idx=None,
                stored=False,
                skipped_reason="dry_run",
                metrics=metrics,
                similarity={"max_abs_corr": max_corr},
                extra=extra,
            )
        col_idx = zoo.append_factor(
            factor_id=factor_id,
            name=name,
            expr=mat_expr,
            values=stored_values,
            status=FactorStatus.full,
            extra=extra,
        )
        if update_similarity:
            sim = sim or SimilarityMatrix(zoo.paths, zoo.manifest.max_factors)
            sim_info = sim.append_factor_correlations(
                zoo,
                factor_id=factor_id,
                col_idx=col_idx,
                values=stored_values,
                top_k=pol.similar_top_k,
            )
        return IngestResult(
            factor_id=factor_id,
            col_idx=col_idx,
            stored=True,
            skipped_reason=None,
            metrics=metrics,
            similarity=sim_info,
            extra=extra,
        )

    if not overwrite:
        return IngestResult(
            factor_id=factor_id,
            col_idx=existing.col_idx,
            stored=False,
            skipped_reason="already_exists",
            metrics=metrics,
            similarity=None,
            extra=extra,
        )

    if dry_run:
        return IngestResult(
            factor_id=factor_id,
            col_idx=existing.col_idx,
            stored=False,
            skipped_reason="dry_run",
            metrics=metrics,
            similarity=None,
            extra=extra,
        )

    col_idx = zoo.overwrite_factor(
        factor_id=factor_id,
        name=name,
        expr=mat_expr,
        values=stored_values,
        status=FactorStatus.full,
        extra=extra,
    )
    if update_similarity:
        sim = SimilarityMatrix(zoo.paths, zoo.manifest.max_factors)
        sim_info = sim.append_factor_correlations(
            zoo,
            factor_id=factor_id,
            col_idx=col_idx,
            values=stored_values,
            top_k=pol.similar_top_k,
        )
    return IngestResult(
        factor_id=factor_id,
        col_idx=col_idx,
        stored=True,
        skipped_reason=None,
        metrics=metrics,
        similarity=sim_info,
        extra=extra,
    )


