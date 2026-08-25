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
    topn_portfolio_summary,
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



def materialize_factor(expr: str, panel: pd.DataFrame) -> MaterializeResult:
    """DSL 求值并对齐 panel 行序。"""
    panel = panel.sort_index()
    out = eval_factor(expr, panel)
    if not isinstance(out, pd.Series):
        raise TypeError(f"因子输出须为 Series，得到 {type(out)!r}")
    values = align_series_to_panel(out, panel)
    tags = collect_aux_intervals_from_expr(expr)
    return MaterializeResult(values=values, expr=expr.strip(), aux_tags=tags)


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
    metrics = evaluate_on_panel(eval_values, eval_panel, label_col=policy.label_col)
    winsorized_values = cross_sectional_winsorize_values(eval_values, eval_panel)
    winsorized_metrics = evaluate_on_panel(
        winsorized_values, eval_panel, label_col=policy.label_col
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
    direction = 1 if float(metrics["ic"]) >= 0.0 else -1
    metrics["long_group_annual_excess_return"] = annualized_long_group_excess_return(
        factor_series,
        eval_panel[policy.label_col],
        direction=direction,
    )
    topn = topn_portfolio_summary(
        factor_series * direction,
        eval_panel[policy.label_col],
    )
    for key in ("annualized_return", "annualized_excess_return", "sharpe", "max_drawdown", "annual_turnover", "n_days"):
        metrics[f"topn_{key}"] = topn.get(key)
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
    panel: pd.DataFrame,
    policy: IngestPolicy | None = None,
    stored_values: np.ndarray | None = None,
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
) -> IngestResult:
    """单因子：物化 → [可选 clip] → mask → eval 指标 → 查重 → 入库。"""
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
        if zoo.n_factors > 0:
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


def load_panel_for_zoo(
    zoo: FactorZoo,
    *,
    panel_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """加载与因子库对齐的 panel（可选日期切片）。"""
    from alphaagent.data.adapters.cnequity import CNE_SOURCE, is_cne_source, load_panel_from_cne
    if is_cne_source(panel_path):
        panel = load_panel_from_cne(start=start, end=end, universe_mask=False)
    else:
        path = panel_path or Path(zoo.manifest.panel_path)
        panel = load_panel(path)
    panel = slice_panel(panel, start=start, end=end)
    if len(panel) != zoo.manifest.n_rows and start is None and end is None:
        raise ValueError(
            f"panel 行数 {len(panel)} != 库 n_rows {zoo.manifest.n_rows}；"
            "请用相同 panel 初始化库，或仅用于调试切片"
        )
    return panel.sort_index()
