"""Stacking 数据集构建：因子枚举 → 物化 → 截面预处理 → 前向收益标签。

行粒度：全部矩阵按 panel 行序（datetime, instrument）对齐，训练样本即
panel 行。标签 = T+1 收盘进、T+1+hold 收盘出的 N 日收益（与
``alphaagent/data/panel.py`` 的 N 日标签口径一致）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from alphaagent.factor.zoo.catalog import FactorCatalog
from core.factor_categories import candidate_dir, production_dir


@dataclass(frozen=True)
class FactorEntry:
    """一个入库因子的最小描述（跨库去重后）。"""

    factor_id: str
    name: str
    expr: str
    library: str  # 如 production_technical / candidate_fundamental
    created_at: str | None = None  # 入库时间（mining_end auto 推断用）


def _to_utc_naive(ts) -> pd.Timestamp | None:
    """created_at 可能是 tz-aware 或 date；统一转成 tz-naive Timestamp。"""
    try:
        ts = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _coerce_naive(ts) -> pd.Timestamp:
    """强制 tz-naive Timestamp（panel datetime 层为 naive，比较前必须归一）。"""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


@dataclass
class StackingDataset:
    panel: pd.DataFrame
    feature_matrix: np.ndarray  # [n_rows, K] float32，已截面 rank+zscore+中性化
    feature_names: list[str]
    entries: list[FactorEntry]  # 与 feature_names 一一对应（冗余剔除后）
    label: np.ndarray  # [n_rows] 前向收益
    mining_end: pd.Timestamp  # 时间隔离边界：仅此日期之后的行可进入训练/OOS
    label_days: int
    dropped: list[dict] = field(default_factory=list)  # 被剔除因子及原因


def collect_factor_entries(
    modes: Sequence[str] = ("technical", "fundamental"),
    *,
    include_candidate: bool = True,
    include_production: bool = True,
) -> list[FactorEntry]:
    """枚举四个因子库（candidate/production × technical/fundamental）的全部因子。

    双数据源合并：catalog（meta/factors.parquet）+ candidate registry
    （mining_candidate_registry.json，挖掘中尚未写 catalog 的候选也在内）。
    同一表达式只保留首个（production 优先于 candidate，modes 顺序优先）。
    """
    entries: list[FactorEntry] = []
    seen_exprs: set[str] = set()

    def _add(entry: FactorEntry) -> None:
        expr = entry.expr.strip()
        if not expr or expr in seen_exprs:
            return
        seen_exprs.add(expr)
        entries.append(entry)

    for mode in modes:
        libs: list[tuple[str, Path]] = []
        if include_production:
            libs.append((f"production_{mode}", production_dir(mode)))
        if include_candidate:
            libs.append((f"candidate_{mode}", candidate_dir(mode)))
        for lib_name, root in libs:
            catalog = FactorCatalog(root / "meta" / "factors.parquet")
            for fid in catalog.list_factor_ids():
                meta = catalog.get(fid)
                if meta is None:
                    continue
                created = getattr(meta, "created_at", None)
                _add(FactorEntry(
                    factor_id=meta.factor_id, name=meta.name, expr=meta.expr.strip(),
                    library=lib_name,
                    created_at=str(created) if created is not None else None,
                ))
            if include_candidate and lib_name.startswith("candidate_"):
                registry_path = root / "mining_candidate_registry.json"
                if registry_path.exists():
                    import json

                    try:
                        registry = json.loads(registry_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    items = registry.values() if isinstance(registry, dict) else registry
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name") or "")
                        expr = str(item.get("expr") or "")
                        if not name or not expr:
                            continue
                        _add(FactorEntry(
                            factor_id=name, name=name, expr=expr.strip(), library=lib_name,
                            created_at=str(item.get("ingested_at")) if item.get("ingested_at") else None,
                        ))
    return entries


def materialize_entries(
    panel: pd.DataFrame,
    entries: Sequence[FactorEntry],
    *,
    cache=None,
    min_finite_ratio: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[tuple[FactorEntry, np.ndarray]], list[dict]]:
    """逐因子 DSL 求值并对齐 panel 行序；覆盖率过低的因子剔除。"""
    from alphaagent.factor.ingest import materialize_factor

    n = len(panel)
    kept: list[tuple[FactorEntry, np.ndarray]] = []
    dropped: list[dict] = []
    for i, entry in enumerate(entries, 1):
        if progress:
            progress(f"[{i}/{len(entries)}] materialize {entry.name} ({entry.library})")
        try:
            result = materialize_factor(entry.expr, panel, cache=cache)
            values = np.asarray(result.values, dtype=np.float32)
        except Exception as exc:  # 单因子失败不拖垮整个数据集
            dropped.append({"name": entry.name, "library": entry.library, "reason": f"eval_error: {exc}"})
            continue
        ratio = float(np.isfinite(values).mean()) if n else 0.0
        if ratio < min_finite_ratio:
            dropped.append({"name": entry.name, "library": entry.library, "reason": f"coverage={ratio:.3f}"})
            continue
        kept.append((entry, values))
    return kept, dropped


def transform_factor_values(
    values: np.ndarray,
    panel: pd.DataFrame,
    *,
    size_neutral: bool = True,
) -> np.ndarray:
    """原始因子值 → 逐日截面 rank → zscore。

    先做市值中性化（逐日对 log(float_cap) 回归取残差，panel 缺 float_cap 时
    跳过），再做行业内 rank pct（panel 缺 industry_sw_l1 时退化为全市场 rank），
    最后逐日 zscore。输出 float32、NaN 保持 NaN。
    """
    if size_neutral and "float_cap" in panel.columns:
        from alphaagent.factor.metrics import cross_sectional_size_neutralize_values

        values = cross_sectional_size_neutralize_values(values, panel)

    df = pd.DataFrame(
        {
            "v": values.astype(np.float64),
            "datetime": panel.index.get_level_values("datetime"),
        }
    )
    industry = panel.index.get_level_values("instrument")  # placeholder，防列缺失
    if "industry_sw_l1" in panel.columns:
        industry = pd.Series(panel["industry_sw_l1"].to_numpy(), index=panel.index)
        df["ind"] = industry.to_numpy()
        rank_keys = ["datetime", "ind"]
    else:
        rank_keys = ["datetime"]

    # 逐组 rank pct（NaN 不参与），再逐日 zscore
    ranked = df.groupby(rank_keys, sort=False, dropna=True)["v"].rank(pct=True)
    df["r"] = ranked
    g = df.groupby("datetime", sort=False)["r"]
    mean = g.transform("mean")
    std = g.transform("std")
    z = (df["r"] - mean) / std.where(std > 1e-12)
    out = z.to_numpy(dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def forward_return_label(panel: pd.DataFrame, hold_days: int) -> np.ndarray:
    """N 日前向收益：T+1 收盘进场、T+1+N 收盘出场（无前视，尾部为 NaN）。"""
    close = pd.Series(
        panel["adj_close"].to_numpy(dtype=np.float64), index=panel.index
    )
    entry = close.groupby(level="instrument", sort=False).shift(-1)
    exit_ = close.groupby(level="instrument", sort=False).shift(-(hold_days + 1))
    label = (exit_ / entry - 1.0).to_numpy(dtype=np.float32)
    return label


def daily_spearman_ic(
    pred: np.ndarray,
    label: np.ndarray,
    dates: pd.DatetimeIndex | pd.Series,
) -> pd.Series:
    """逐日截面 Spearman IC（rank 化后逐日 Pearson，全向量化）。"""
    df = pd.DataFrame(
        {
            "p": pred.astype(np.float64),
            "y": label.astype(np.float64),
            "d": pd.Series(dates).to_numpy() if not isinstance(dates, pd.Series) else dates.to_numpy(),
        }
    ).dropna()
    if df.empty:
        return pd.Series(dtype=np.float64)
    df["pr"] = df.groupby("d")["p"].rank()
    df["yr"] = df.groupby("d")["y"].rank()
    d = df["d"]
    n = df.groupby("d")["pr"].count().astype(np.float64)
    e_pr = df.groupby("d")["pr"].mean()
    e_yr = df.groupby("d")["yr"].mean()
    e_xy = (df["pr"] * df["yr"]).groupby(d).mean()
    sd_pr = df.groupby("d")["pr"].std()
    sd_yr = df.groupby("d")["yr"].std()
    # 样本协方差 = (E[xy] - Ex·Ey) · n/(n-1)，与 std 的 ddof=1 一致
    cov = (e_xy - e_pr * e_yr) * n / (n - 1)
    ic = cov / (sd_pr * sd_yr)
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
    return ic


def build_dataset_from_values(
    panel: pd.DataFrame,
    factor_values: Sequence[tuple[FactorEntry, np.ndarray]],
    *,
    label_days: int,
    mining_end: pd.Timestamp,
    size_neutral: bool = True,
    max_corr: float = 0.9,
    ics_for_quality: dict[str, float] | None = None,
) -> StackingDataset:
    """从已物化的因子值数组组装数据集（含冗余过滤；供脚本与测试共用）。

    ``ics_for_quality``：{因子名: |mining 窗口日均 IC|}。冗余过滤为贪心：
    质量分高者先保留，后续因子与任一已保留因子 |corr| > max_corr 则剔除。
    """
    quality = ics_for_quality or {}

    # 质量分降序贪心保留；最终按原始传入顺序输出，保证结果可复现
    names_in_order = [e.name for e, _ in factor_values]
    order = sorted(
        range(len(factor_values)),
        key=lambda i: quality.get(factor_values[i][0].name, 0.0),
        reverse=True,
    )
    kept: list[tuple[FactorEntry, np.ndarray, np.ndarray]] = []  # (entry, raw, transformed)
    dropped: list[dict] = []
    for i in order:
        entry, raw = factor_values[i]
        transformed = transform_factor_values(raw, panel, size_neutral=size_neutral)
        redundant_with: str | None = None
        for kept_entry, _, kept_arr in kept:
            corr = _sampled_corr(transformed, kept_arr, panel)
            if corr is not None and abs(corr) > max_corr:
                redundant_with = kept_entry.name
                break
        if redundant_with is not None:
            dropped.append({"name": entry.name, "library": entry.library,
                            "reason": f"redundant_with={redundant_with}"})
            continue
        kept.append((entry, raw, transformed))
    kept.sort(key=lambda t: names_in_order.index(t[0].name))

    feature_matrix = (
        np.column_stack([t for _, _, t in kept]).astype(np.float32) if kept else np.empty((len(panel), 0), dtype=np.float32)
    )
    label = forward_return_label(panel, label_days)
    return StackingDataset(
        panel=panel,
        feature_matrix=feature_matrix,
        feature_names=[e.name for e, _, _ in kept],
        entries=[e for e, _, _ in kept],
        label=label,
        mining_end=_coerce_naive(mining_end),
        label_days=label_days,
        dropped=dropped,
    )


def _sampled_corr(a: np.ndarray, b: np.ndarray, panel: pd.DataFrame, stride: int = 5) -> float | None:
    """按日抽样的截面 Pearson 均值（两因子间；全 NaN 或样本不足返回 None）。"""
    dts = panel.index.get_level_values("datetime")
    unique_days = dts.unique()
    pairs_a, pairs_b = [], []
    for day in unique_days[::stride]:
        mask = dts == day
        xa, xb = a[mask], b[mask]
        ok = np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 5:
            continue
        pairs_a.append(xa[ok])
        pairs_b.append(xb[ok])
    if len(pairs_a) < 3:
        return None
    a_all = np.concatenate(pairs_a)
    b_all = np.concatenate(pairs_b)
    if a_all.std() < 1e-12 or b_all.std() < 1e-12:
        return None
    return float(np.corrcoef(a_all, b_all)[0, 1])


def build_stacking_dataset(
    panel: pd.DataFrame,
    entries: Sequence[FactorEntry],
    *,
    label_days: int,
    mining_end: pd.Timestamp,
    size_neutral: bool = True,
    max_corr: float = 0.9,
    cache=None,
    decay_months: int = 12,
    min_finite_ratio: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> StackingDataset:
    """完整流水线：物化 → 冗余过滤（以 mining 窗口 IC 为质量分）→ 组装。"""
    materialized, dropped_eval = materialize_entries(
        panel, entries, cache=cache, min_finite_ratio=min_finite_ratio, progress=progress
    )
    dropped: list[dict] = list(dropped_eval)

    mining_start = _coerce_naive(mining_end) - pd.DateOffset(months=decay_months)
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    label = forward_return_label(panel, label_days)
    quality: dict[str, float] = {}
    ics: dict[str, pd.Series] = {}
    for entry, raw in materialized:
        mask = (dts <= mining_end) & (dts >= mining_start)
        if mask.sum() < 20:
            quality[entry.name] = 0.0
            continue
        ic = daily_spearman_ic(raw[mask], label[mask], dts[mask])
        ics[entry.name] = ic
        quality[entry.name] = float(ic.abs().mean()) if len(ic) else 0.0

    dataset = build_dataset_from_values(
        panel,
        materialized,
        label_days=label_days,
        mining_end=mining_end,
        size_neutral=size_neutral,
        max_corr=max_corr,
        ics_for_quality=quality,
    )
    dataset.dropped = dropped + dataset.dropped
    return dataset


def decay_table(
    factor_values: Iterable[tuple[FactorEntry, np.ndarray]],
    panel: pd.DataFrame,
    label: np.ndarray,
    *,
    mining_end: pd.Timestamp,
    decay_months: int = 12,
) -> list[dict]:
    """幸存者偏差量化：每因子 mining 窗口日均 IC vs OOS（mining_end 后）日均 IC。"""
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    mining_start = _coerce_naive(mining_end) - pd.DateOffset(months=decay_months)
    mining_end = _coerce_naive(mining_end)
    rows: list[dict] = []
    for entry, raw in factor_values:
        m_mask = (dts <= mining_end) & (dts >= mining_start)
        o_mask = dts > mining_end
        ic_m = daily_spearman_ic(raw[m_mask], label[m_mask], dts[m_mask]) if m_mask.sum() >= 20 else pd.Series(dtype=float)
        ic_o = daily_spearman_ic(raw[o_mask], label[o_mask], dts[o_mask]) if o_mask.sum() >= 20 else pd.Series(dtype=float)
        ic_m_mean = float(ic_m.mean()) if len(ic_m) else None
        ic_o_mean = float(ic_o.mean()) if len(ic_o) else None
        decay = (ic_o_mean / ic_m_mean) if (ic_m_mean and ic_o_mean is not None and abs(ic_m_mean) > 1e-12) else None
        rows.append(
            {
                "name": entry.name,
                "library": entry.library,
                "ic_mining": ic_m_mean,
                "ic_oos": ic_o_mean,
                "decay_ratio": decay,
            }
        )
    return rows
