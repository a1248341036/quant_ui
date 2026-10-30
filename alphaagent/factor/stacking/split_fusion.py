"""分频多模型融合（方案 B，派二）：按因子 label 周期分组 → 每组独立建模 → 组合层融合。

业界对照（必读：artifacts/alphaagent/mine_campaigns/20261029_night/mixed_freq_ml_research_20261030.md）：
- 派二 = 每个频率独立建模，预测分数在组合层融合（MIDAS encoder-decoder / 华泰
  多任务 GRU 组合层融合 / Zhang et al. 2025 MLF 可学习加权集成 LWI）。
- 对治的老毛病：把 1d/5d/20d 不同 label 的因子拉平到同一日频面板 + 统一 label
  训练（混频混训）——train/val 好盲测崩的结构性原因之一。

泛化契约（N 组，不按组名写死任何行为）：
- 分组键：factor entry 的 label_col 经 parse_label_days 解析出的 label_days；
  无法解析的因子归入目标调仓节拍组（blend_label_days）。组名 = f"{days}d"。
- 每组独立 build（组内冗余过滤；compute_quality_scores 已按各因子自身 label
  horizon 计 IC，语义不变）→ 独立 fit_predict_walkforward，组内 label =
  组成员 label_days（组内天然一致）。
- 折结构统一：purge 放宽到覆盖最慢组的前向标签（防慢周期 label 跨折泄漏，
  且保证各组 OOS 行集一致，融合可逐行对齐）。
- 融合：各组 OOS 分数在折内逐日 zscore（daily_zscore）后加权合成，
  某行缺失部分组时按在场权重重归一；
  权重：equal = 1/N；icir = 仅在**已产出的历史 OOS 行**上滚动估计
  （首折回退等权，时序诚实无未来信息，同 csi500-multifactor combiner.py
  "IC_IR 窗口严格 < T" 先例）。
- 退化性质：单组 = 组内模型结果（zscore 是无害单调变换）；某组全缺席（训练
  样本不足/特征全被剔）时融合按在场组重归一。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import warnings

from .dataset import (
    FactorEntry,
    StackingDataset,
    build_stacking_dataset,
    daily_spearman_ic,
)
from .groups import daily_zscore, parse_label_days
from .model import WalkForwardFold, fit_predict_walkforward, walk_forward_splits


def split_factor_groups(
    entries: Sequence[FactorEntry],
    blend_label_days: int = 5,
) -> tuple[dict[str, list[FactorEntry]], dict]:
    """按 entry.label_col 解析 label_days 分组（1d / 5d / 20d …）。

    返回 ({组名 f"{days}d": [FactorEntry, …]}, 诊断)。组序按 days 升序稳定；
    label_col 缺失/无法解析的因子归入 blend_label_days 组（对齐目标调仓节拍），
    诊断里记录 unparsed 名单。
    """
    groups: dict[str, list[FactorEntry]] = {}
    unparsed: list[str] = []
    for e in entries:
        days = parse_label_days(getattr(e, "label_col", None))
        if days is None:
            days = int(blend_label_days)
            unparsed.append(e.name)
        groups.setdefault(f"{days}d", []).append(e)
    ordered = {k: groups[k] for k in sorted(groups, key=lambda k: int(k[:-1]))}
    return ordered, {
        "unparsed": unparsed,
        "group_sizes": {k: len(v) for k, v in ordered.items()},
    }


def build_group_datasets(
    panel: pd.DataFrame,
    groups: dict[str, list[FactorEntry]],
    *,
    mining_end: pd.Timestamp,
    size_neutral: bool = True,
    max_corr: float = 0.6,
    cache=None,
    decay_months: int = 12,
    min_finite_ratio: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> dict[str, StackingDataset]:
    """每组独立 build_stacking_dataset（组内冗余过滤，组内 label = 组周期）。

    物化走共享 cache（跨组命中磁盘缓存，不重复 DSL 求值）；组与组之间不做
    跨频率相关性剔除——不同频率因子物化周期/语义天然不同，互剔会系统性
    挤掉慢周期组（与 dataset._entry_group 的"同组比较"设计一致）。
    """
    out: dict[str, StackingDataset] = {}
    for key, group_entries in groups.items():
        label_days = int(key[:-1])
        if progress:
            progress(f"构建频段组 {key}（{len(group_entries)} 个因子，label={label_days}d）")
        out[key] = build_stacking_dataset(
            panel,
            group_entries,
            label_days=label_days,
            mining_end=mining_end,
            size_neutral=size_neutral,
            max_corr=max_corr,
            cache=cache,
            decay_months=decay_months,
            min_finite_ratio=min_finite_ratio,
            progress=progress,
        )
    return out


def build_frequency_folds(
    dts: pd.DatetimeIndex,
    *,
    isolation: str,
    panel_start: pd.Timestamp,
    mining_end: pd.Timestamp,
    group_label_days: Sequence[int],
    train_months: int,
    step_months: int,
    purge_days: int,
) -> list[WalkForwardFold]:
    """折生成（strict/holdout 与主链同构），purge 放宽覆盖最慢组前向标签。

    purge = max(purge_days, 组 label_days 最大值)：20d 组的向前标签跨折泄漏
    需要至少 20 个交易日的 purge gap；统一大 purge 保证各组 OOS 行集完全一致
    （融合逐行对齐的前提）。
    """
    wf_train_start = panel_start if isolation == "holdout" else mining_end
    base_months = max(3, int((mining_end - panel_start).days / 30.44))
    purge = max(int(purge_days), *[int(d) for d in group_label_days])
    if isolation == "holdout":
        folds: list[WalkForwardFold] = []
        for bump in (0, 1, 2, 3):
            folds = walk_forward_splits(
                dts, train_start=wf_train_start, train_months=base_months + bump,
                step_months=step_months, purge_days=purge,
            )
            if folds and pd.Timestamp(folds[0].oos_dates.min()) >= mining_end:
                break
        if folds:
            folds = [f for f in folds if pd.Timestamp(f.oos_dates.min()) >= mining_end]
        else:
            folds = []
    else:
        folds = walk_forward_splits(
            dts, train_start=mining_end, train_months=train_months,
            step_months=step_months, purge_days=purge,
        )
    return folds


@dataclass
class FrequencyGroupResult:
    """一个频段组的训练产物：组内模型分数 + 诊断。"""

    key: str
    label_days: int
    member_names: list[str]  # 分组原始成员（含冗余过滤剔除）
    dataset: StackingDataset
    model_scores: dict[str, np.ndarray]  # {kind: [n_rows] 该模型 OOS 分数}
    scores: np.ndarray  # [n_rows] 组内模型 nanmean（折外行 NaN）
    label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # 组内 label（组 label_days 前向收益）
    fold_metrics: dict[str, list] = field(default_factory=dict)
    feature_weights: dict[str, list] = field(default_factory=dict)
    feature_contribution: dict[str, list] = field(default_factory=dict)
    oos_ic: dict | None = None  # 组分数 vs 组 label 全 OOS 指标


def train_frequency_model(
    dataset: StackingDataset,
    dates: pd.Series,
    folds: Sequence[WalkForwardFold],
    *,
    kinds: Sequence[str],
    collect_diagnostics: bool = True,
    first_oos=None,
    group_members: Sequence[FactorEntry] | None = None,
) -> FrequencyGroupResult:
    """组内 walk-forward 训练（Ridge/LGBM，复用 fit_predict_walkforward）。

    组内 label = dataset.label（组成员 label_days 的前向收益），label_horizon
    传组 label_days（置换贡献的风险口径按组周期计算，防 20d 组被当 5d 组评估）。
    ``group_members``：分组原始成员（含冗余过滤剔除），用于报告组规模；
    None 时回退为过滤后的入模特征名单。组内特征全被剔除（feature_matrix
    空）时返回缺席占位（scores 全 NaN，融合按在场组重归一）。
    """
    members = list(group_members) if group_members else dataset.entries
    if dataset.feature_matrix.shape[1] == 0:
        return FrequencyGroupResult(
            key=f"{dataset.label_days}d",
            label_days=dataset.label_days,
            member_names=[e.name for e in members],
            dataset=dataset,
            model_scores={},
            scores=np.full(len(dataset.label), np.nan, dtype=np.float32),
            fold_metrics={},
            oos_ic=None,
        )
    model_scores: dict[str, np.ndarray] = {}
    fold_metrics: dict[str, list] = {}
    feature_weights: dict[str, list] = {}
    feature_contribution: dict[str, list] = {}
    for kind in kinds:
        pred, rep, fw, fc = fit_predict_walkforward(
            dataset.feature_matrix, dataset.label, dates, folds, kind=kind,
            feature_names=dataset.feature_names, label_horizon=dataset.label_days,
            collect_diagnostics=collect_diagnostics,
        )
        model_scores[kind] = pred
        fold_metrics[kind] = rep
        if fw:
            feature_weights[kind] = fw
        if fc:
            feature_contribution[kind] = fc
    usable = [model_scores[k] for k in kinds if np.isfinite(model_scores[k]).any()]
    if usable:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scores = np.nanmean(np.column_stack(usable), axis=1).astype(np.float32)
    else:
        scores = np.full(len(dataset.label), np.nan, dtype=np.float32)
    key = f"{dataset.label_days}d"
    members = list(group_members) if group_members else dataset.entries
    return FrequencyGroupResult(
        key=key,
        label_days=dataset.label_days,
        member_names=[e.name for e in members],
        dataset=dataset,
        model_scores=model_scores,
        scores=scores,
        label=dataset.label,
        fold_metrics=fold_metrics,
        feature_weights=feature_weights,
        feature_contribution=feature_contribution,
        oos_ic=_oos_ic_metrics(scores, dataset.label, dates, first_oos, dataset.label_days),
    )


def _oos_ic_metrics(
    score: np.ndarray,
    label: np.ndarray,
    dates: pd.Series,
    first_oos,
    label_horizon: int,
) -> dict | None:
    """组分数 vs 组内 label 的全 OOS 指标（date ≥ first_oos 且双方有限）。

    口径与 scripts/train_ml_composite.py 的 _blended_oos_ic 一致（<100 行 None）。
    """
    date_np = pd.to_datetime(pd.Series(dates)).to_numpy()
    mask = (
        (date_np >= pd.Timestamp(first_oos))
        & np.isfinite(score) & np.isfinite(label)
    )
    if mask.sum() < 100:
        return None
    ic = daily_spearman_ic(score[mask], label[mask], pd.Series(date_np[mask]))
    if not len(ic):
        return None
    return {
        "ic_mean": round(float(ic.mean()), 6),
        "ic_ir": round(float(ic.mean() / ic.std()), 4)
        if len(ic) > 2 and ic.std() > 1e-12 else None,
        "n_days": int(len(ic)),
        "label_horizon_days": int(label_horizon),
    }


def blend_group_scores(
    group_results: Sequence[FrequencyGroupResult],
    folds: Sequence[WalkForwardFold],
    dates: pd.Series,
    *,
    first_oos,
    blend_weight: str = "icir",
    blend_label_days: int = 5,
    min_hist_rows: int = 100,
) -> tuple[np.ndarray, dict]:
    """频段分数融合到统一调仓节拍：折内逐日 zscore + 加权在场归一合成。

    权重方案：
    - equal：1/N 等权。
    - icir：第 i 折用前 i 折**已产出**的 OOS 行（≥ first_oos 且早于本折 OOS
      起点）对每组估计逐日 Spearman IC → 均值/标准差 = ICIR → 非负归一；
      历史不足（< min_hist_rows）或无正 ICIR 时回退等权；首折必然等权。

    已知偏置（判读时注意）：慢周期组（20d）label 是逐日滚动重叠标签，逐日
    IC 序列自相关使 ICIR 系统性偏高，icir 权重会偏向慢周期组。组间权重是
    相对次序而非绝对定价，且权重只来自已产出 OOS（无未来信息），可作为
    一期对照基线；矫正（horizon 折扣 / 简单回归配权 LWI）留二期。

    返回 (blended [n_rows], 诊断)。blended 非折 OOS 行 = NaN（与主链 stacked 同构）。
    """
    date_np = pd.to_datetime(pd.Series(dates)).to_numpy()
    n = len(date_np)
    blended = np.full(n, np.nan, dtype=np.float32)
    first_oos_ts = pd.Timestamp(first_oos)
    keys = [g.key for g in group_results]
    fold_weights: list[dict] = []

    for i, fold in enumerate(folds):
        oos_idx = np.flatnonzero(np.isin(date_np, fold.oos_dates.to_numpy()))
        if not len(oos_idx):
            continue
        weights: dict[str, float] | None = None
        fallback: str | None = None
        if blend_weight == "icir" and i > 0:
            hist_mask = (
                (date_np >= first_oos_ts)
                & (date_np < pd.Timestamp(fold.oos_dates.min()))
            )
            icirs: dict[str, float] = {}
            for g in group_results:
                hm = hist_mask & np.isfinite(g.scores) & np.isfinite(g.label)
                if int(hm.sum()) < min_hist_rows:
                    continue
                ic = daily_spearman_ic(g.scores[hm], g.label[hm], pd.Series(date_np[hm]))
                if len(ic) > 2 and ic.std() > 1e-12:
                    icirs[g.key] = float(ic.mean() / ic.std())
                else:
                    icirs[g.key] = 0.0
            pos = {k: max(v, 0.0) for k, v in icirs.items()}
            total = sum(pos.values())
            if total > 1e-12:
                weights = {k: pos[k] / total for k in pos}
            else:
                fallback = "all_icir_nonpositive"
        if weights is None:
            weights = {k: 1.0 / len(keys) for k in keys}
            fallback = fallback or ("equal_first_fold" if i == 0 else "equal_no_hist")
        num = np.zeros(len(oos_idx), dtype=np.float64)
        den = np.zeros(len(oos_idx), dtype=np.float64)
        for g in group_results:
            w = weights.get(g.key, 0.0)
            if w <= 0:
                continue
            seg = np.asarray(g.scores[oos_idx], dtype=np.float64)
            seg_dates = pd.Series(date_np[oos_idx])
            z = daily_zscore(seg, seg_dates.to_numpy())
            finite = np.isfinite(z)
            num += np.where(finite, z * w, 0.0)
            den += np.where(finite, w, 0.0)
        ok = den > 1e-12
        blended[oos_idx] = np.where(ok, num / np.maximum(den, 1e-12), np.nan).astype(np.float32)
        fold_weights.append({
            "oos_start": str(pd.Timestamp(fold.oos_dates.min()).date()),
            "oos_end": str(pd.Timestamp(fold.oos_dates.max()).date()),
            "weights": {k: round(float(weights[k]), 6) for k in weights},
            "fallback": fallback,
        })

    # 全 OOS ICIR——仅供判读（含未来信息，绝不回填为权重）
    oos_mask_all = date_np >= first_oos_ts
    all_icir: dict[str, float] = {}
    for g in group_results:
        m = oos_mask_all & np.isfinite(g.scores) & np.isfinite(g.label)
        if int(m.sum()) < min_hist_rows:
            continue
        ic = daily_spearman_ic(g.scores[m], g.label[m], pd.Series(date_np[m]))
        if len(ic) > 2 and ic.std() > 1e-12:
            all_icir[g.key] = round(float(ic.mean() / ic.std()), 4)
    diag = {
        "blend_label_days": int(blend_label_days),
        "blend_weight": blend_weight,
        "fold_weights": fold_weights,
        "full_oos_icir": all_icir,
        "note": (
            "icir 只在已产出的历史 OOS 行上滚动估计（首折等权，时序诚实）；"
            "full_oos_icir 为全 OOS 段 ICIR，仅判读、不回填权重。"
            "慢周期组（20d）重叠标签使逐日 IC 自相关、ICIR 系统性偏高，"
            "icir 组间权重会偏向慢周期组——矫正（horizon 折扣/简单回归配权）留二期。"
        ),
    }
    return blended, diag


def per_frequency_payload(
    group_results: Sequence[FrequencyGroupResult],
    blend_diag: dict,
    *,
    blend_label_days: int,
    blend_weight: str,
) -> dict:
    """report.json 的 per_frequency 字段：每组 OOS IC/ICIR + 融合权重与诊断。"""
    groups: dict[str, dict] = {}
    for g in group_results:
        groups[g.key] = {
            "label_days": g.label_days,
            "n_entries": len(g.member_names),
            "n_features": len(g.dataset.feature_names),
            "members": g.member_names,
            "dropped": g.dataset.dropped,
            "fold_metrics": g.fold_metrics,
            "feature_weights": g.feature_weights,
            "feature_contribution": g.feature_contribution,
            "oos_ic": g.oos_ic,
        }
    return {
        "blend_label_days": int(blend_label_days),
        "blend_weight": blend_weight,
        "groups": groups,
        "fold_weights": blend_diag.get("fold_weights"),
        "full_oos_icir": blend_diag.get("full_oos_icir"),
        "note": blend_diag.get("note"),
    }