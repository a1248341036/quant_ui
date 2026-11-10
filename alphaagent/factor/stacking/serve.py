"""ML 组合服务化：训练产物固化（served model）与离线推理。

背景：训练脚本此前只把「训练过程的 OOS 预测」写成分数，模型对象用完即丢。
后果是每次更新分数都必须重跑完整训练（~13 分钟），训练一停分数就断供
（2026-11 模拟盘因此空仓一个月）。本模块把训练与推理拆开：

    训练（低频，重）：全量有效行重新拟合每组模型 → ``served_model.joblib``
    推理（每日，轻）：加载模型 → 物化 → 变换 → 预测 → 融合 → 平滑 → pred 通道

两条通道的分工：

- 报告里的 OOS 指标仍来自 walk-forward（每折只用过去训练，时序诚实）；
- served model 走「用截至训练日的全部历史重新拟合」，是上线口径，**不用于评估**。

推理口径必须与训练期预测逐位一致，否则新分数与历史分数不可比：

1. 每组独立物化 → ``transform_factor_values``（逐日 rank / 市值中性化 / zscore）
2. 组内多 kind 等权平均（ridge 只预测特征全有限行；lgbm 原生吃 NaN）
3. 组间按固定权重逐日 zscore + 在场归一
4. 组合分数尾随 WMA 平滑
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .dataset import (
    FactorEntry,
    StackingDataset,
    materialize_entries,
    transform_factor_values,
)
from .groups import daily_zscore
from .model import make_model

SERVED_MODEL_NAME = "served_model.joblib"
MODEL_CARD_NAME = "model_card.json"
SERVED_SCHEMA = 1


# ── 分数后处理（与 scripts/train_ml_composite.py 同源，此处为唯一实现）──────


def wma_smooth_scores(values: np.ndarray, panel: pd.DataFrame, window: int) -> np.ndarray:
    """组合分数尾随线性 WMA（按股票、按交易日），治换手/持仓重叠。

    挖掘侧单因子靠表达式内 WMA 平滑过换手门（ovdiv：WMA20 日换手 1.07→0.50），
    组合分数此前是全链路唯一未平滑的分数。卷积实现（快）：最近一日权重最大；
    窗口内非有限值不参与加权（den 归一）；原始 NaN 行保持 NaN。
    """
    win = int(window)
    if win <= 1:
        return np.asarray(values, dtype=np.float32)
    ser = pd.Series(np.asarray(values, dtype=np.float64), index=panel.index)
    wide = ser.unstack("instrument")
    # 截尾核（降序）：当前日权重最大、往前线性衰减；convolve(a, w)[t] = Σ_j a[t-j]·w[j]
    wd = np.arange(win, 0, -1, dtype=np.float64)
    wd /= wd.sum()
    out: dict[object, np.ndarray] = {}
    for col in wide.columns:
        v = wide[col].to_numpy(dtype=np.float64)
        m = np.isfinite(v)
        if not m.any():
            out[col] = v
            continue
        v0 = np.where(m, v, 0.0)
        num = np.convolve(v0, wd, mode="full")[: len(v)]
        den = np.convolve(m.astype(np.float64), wd, mode="full")[: len(v)]
        sm = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
        sm[~m] = np.nan
        out[col] = sm
    sm_wide = pd.DataFrame(out, index=wide.index, columns=wide.columns)
    long = (
        sm_wide.rename_axis("datetime")
        .reset_index()
        .melt(id_vars="datetime", var_name="instrument", value_name="score")
        .set_index(["datetime", "instrument"])["score"]
    )
    return long.reindex(panel.index).to_numpy(dtype=np.float32)


def _norm_code(value: object) -> str:
    """CNE instrument（000001.SZ）→ 6 位裸码；引擎侧 merge 依赖该格式。"""
    m = re.search(r"(\d{6})", str(value))
    return (m.group(1) if m else str(value)).zfill(6)


def to_pred_frame(values: np.ndarray, panel: pd.DataFrame) -> pd.DataFrame:
    """组合分数 → ``date, code, score`` 长表（``core.data.load_pred_scores`` 格式）。"""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    ser = pd.Series(values[valid], index=panel.index[valid])
    if ser.empty:
        return pd.DataFrame(columns=["date", "code", "score"])
    wide = ser.unstack("instrument")
    wide.columns = [_norm_code(c) for c in wide.columns]
    long = (
        wide.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="code", value_name="score")
        .dropna(subset=["score"])
    )
    long["date"] = pd.to_datetime(long["date"])
    return long.sort_values(["date", "code"]).reset_index(drop=True)


def blend_group_scores_fixed(
    scores_by_group: Mapping[str, np.ndarray],
    weights: dict[str, float],
    dates: Sequence,
) -> np.ndarray:
    """按固定权重融合各组分数（推理侧；训练侧的 ``blend_group_scores`` 逐折算权重）。

    逐组逐日 zscore 后加权、按在场权重归一——与训练侧同一折内合成口径一致。
    """
    date_np = pd.to_datetime(pd.Series(dates)).to_numpy()
    n = len(date_np)
    num = np.zeros(n, dtype=np.float64)
    den = np.zeros(n, dtype=np.float64)
    for key, scores in scores_by_group.items():
        w = float(weights.get(key, 0.0))
        if w <= 0:
            continue
        z = daily_zscore(np.asarray(scores, dtype=np.float64), date_np)
        finite = np.isfinite(z)
        num += np.where(finite, z * w, 0.0)
        den += np.where(finite, w, 0.0)
    ok = den > 1e-12
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan).astype(np.float32)


# ── 服务模型：结构 / 拟合 / 存取 ───────────────────────────────────────────


@dataclass
class GroupModel:
    """单个频段组的固化模型：入模特征顺序 + 各 kind 的 estimator + 训练来源。"""

    key: str
    label_days: int
    feature_names: list[str]
    entries: list[FactorEntry]
    models: dict[str, Any] = field(default_factory=dict)
    n_train_rows: int = 0
    train_start: str | None = None
    train_end: str | None = None

    def summary(self) -> dict:
        return {
            "key": self.key,
            "label_days": int(self.label_days),
            "n_features": len(self.feature_names),
            "kinds": sorted(self.models),
            "n_train_rows": int(self.n_train_rows),
            "train_start": self.train_start,
            "train_end": self.train_end,
        }


@dataclass
class ServedModel:
    """训练产物：按组存放 estimator，加融合权重与后处理参数。"""

    groups: dict[str, GroupModel]
    kinds: list[str]
    blend_weights: dict[str, float]
    blend_label_days: int
    smooth_window: int
    size_neutral: bool
    trained_until: str
    source_run: str = ""
    schema: int = SERVED_SCHEMA
    meta: dict = field(default_factory=dict)

    def group_keys(self) -> list[str]:
        return sorted(self.groups)

    def model_card(self) -> dict:
        """可 JSON 序列化的模型卡（不含 estimator 权重，供运营页/审计）。"""
        return {
            "schema": int(self.schema),
            "source_run": self.source_run,
            "trained_until": self.trained_until,
            "kinds": list(self.kinds),
            "blend_weights": {k: round(float(v), 6) for k, v in self.blend_weights.items()},
            "blend_label_days": int(self.blend_label_days),
            "smooth_window": int(self.smooth_window),
            "size_neutral": bool(self.size_neutral),
            "groups": {k: g.summary() for k, g in self.groups.items()},
            "meta": self.meta,
        }


def _fit_one(
    dataset: StackingDataset,
    kind: str,
    mask: np.ndarray,
) -> Any | None:
    """单 kind 在全量有效行上拟合；有效行不足 100 返回 None（与训练侧同阈值）。"""
    valid = mask.copy()
    if kind == "ridge":
        if dataset.feature_matrix.shape[1]:
            valid &= np.isfinite(dataset.feature_matrix).all(axis=1)
    if int(valid.sum()) < 100:
        return None
    model = make_model(kind)  # type: ignore[arg-type]
    model.fit(dataset.feature_matrix[valid], dataset.label[valid])
    return model


def fit_group_model(
    dataset: StackingDataset,
    kinds: Sequence[str],
    *,
    train_day_mask: np.ndarray | None = None,
    train_end: str | pd.Timestamp | None = None,
) -> GroupModel | None:
    """用组数据集的全量有效行拟合上线模型（不受 walk-forward 折限制）。

    与评估口径的区别：walk-forward 每折只用过去训练（用于报告指标），这里用
    截至 ``train_end`` 的全部有效行（上线模型），两者互不替代。
    """
    if dataset.feature_matrix.shape[1] == 0:
        return None
    dates = pd.DatetimeIndex(dataset.panel.index.get_level_values("datetime"))
    mask = np.isfinite(dataset.label)
    if train_day_mask is not None:
        mask = mask & train_day_mask
    if train_end is not None:
        mask = mask & (dates <= pd.Timestamp(train_end))
    if int(mask.sum()) < 100:
        return None

    models: dict[str, Any] = {}
    for kind in kinds:
        model = _fit_one(dataset, kind, mask)
        if model is not None:
            models[kind] = model
    if not models:
        return None

    used_dates = dates[mask]
    return GroupModel(
        key=f"{int(dataset.label_days)}d",
        label_days=int(dataset.label_days),
        feature_names=list(dataset.feature_names),
        entries=list(dataset.entries),
        models=models,
        n_train_rows=int(mask.sum()),
        train_start=str(used_dates.min().date()) if len(used_dates) else None,
        train_end=str(used_dates.max().date()) if len(used_dates) else None,
    )


def fit_served_model(
    group_datasets: dict[str, StackingDataset],
    kinds: Sequence[str],
    *,
    blend_weights: dict[str, float] | None = None,
    blend_label_days: int = 5,
    smooth_window: int = 0,
    size_neutral: bool = True,
    source_run: str = "",
    train_end: str | pd.Timestamp | None = None,
    progress: Callable[[str], None] | None = None,
    meta: dict | None = None,
) -> ServedModel:
    """各组全量重拟合 → ServedModel（融合权重缺省等权）。"""
    groups: dict[str, GroupModel] = {}
    for key, dataset in group_datasets.items():
        if progress:
            progress(f"固化频段组 {key}（{dataset.feature_matrix.shape[1]} 个入模特征）")
        gm = fit_group_model(dataset, kinds, train_end=train_end)
        if gm is not None:
            groups[key] = gm
    if not groups:
        raise ValueError("没有任何频段组可固化（入模特征全为空或有效行不足）")

    weights = {k: max(float((blend_weights or {}).get(k, 0.0)), 0.0) for k in groups}
    total = sum(weights.values())
    if total <= 1e-12:
        weights = {k: 1.0 / len(groups) for k in groups}
    else:
        weights = {k: v / total for k, v in weights.items()}

    ends = [g.train_end for g in groups.values() if g.train_end]
    return ServedModel(
        groups=groups,
        kinds=[k for k in kinds if any(k in g.models for g in groups.values())],
        blend_weights=weights,
        blend_label_days=int(blend_label_days),
        smooth_window=int(smooth_window),
        size_neutral=bool(size_neutral),
        trained_until=max(ends) if ends else "",
        source_run=source_run,
        meta=dict(meta or {}),
    )


def save_served_model(model: ServedModel, directory: str | Path) -> Path:
    """落盘 ``served_model.joblib`` + ``model_card.json``，返回 joblib 路径。"""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SERVED_MODEL_NAME
    joblib.dump(model, path)
    (out / MODEL_CARD_NAME).write_text(
        json.dumps(model.model_card(), ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )
    return path


def load_served_model(path: str | Path) -> ServedModel:
    """加载 served model；schema 不匹配直接报错（防旧产物静默错配）。"""
    obj = joblib.load(Path(path))
    if not isinstance(obj, ServedModel):
        raise TypeError(f"{path} 不是 ServedModel（{type(obj).__name__}）")
    if int(obj.schema) != SERVED_SCHEMA:
        raise ValueError(f"served model schema={obj.schema} 与当前 {SERVED_SCHEMA} 不兼容")
    return obj


def resolve_latest_served_model(root: str | Path) -> Path:
    """在 stacking 产物目录下找最新 ``served_model.joblib``（按 mtime）。"""
    base = Path(root)
    candidates = sorted(
        base.glob(f"*/{SERVED_MODEL_NAME}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"{base} 下没有任何 {SERVED_MODEL_NAME}")
    return candidates[0]


# ── 推理 ───────────────────────────────────────────────────────────────────


def _group_matrix(
    served: ServedModel,
    group: GroupModel,
    panel: pd.DataFrame,
    *,
    cache=None,
) -> np.ndarray:
    """物化该组入模因子并按 ``feature_names`` 顺序组装已变换特征矩阵。"""
    materialized, _dropped = materialize_entries(
        panel, group.entries, cache=cache, min_finite_ratio=0.0
    )
    by_name = {entry.name: values for entry, values in materialized}
    cols: list[np.ndarray] = []
    n_rows = len(panel)
    for name in group.feature_names:
        values = by_name.get(name)
        if values is None:
            cols.append(np.full(n_rows, np.nan, dtype=np.float32))
        else:
            cols.append(np.asarray(values, dtype=np.float32))
    if not cols:
        return np.empty((n_rows, 0), dtype=np.float32)
    # transform_factor_values 是逐列 API（1D），与训练期 build_dataset_from_values
    # 的调用粒度一致（那边也是逐因子 raw 调一次），保证推理/训练变换口径同源。
    transformed = [
        transform_factor_values(col, panel, size_neutral=served.size_neutral)
        for col in cols
    ]
    return np.column_stack(transformed).astype(np.float32)


def _predict_group(group: GroupModel, features: np.ndarray) -> np.ndarray:
    """组内多 kind 等权平均；ridge 保守（特征全有限行才预测），lgbm 吃 NaN。"""
    n = features.shape[0]
    preds: list[np.ndarray] = []
    for kind, model in group.models.items():
        if kind == "ridge":
            ok = np.isfinite(features).all(axis=1) if features.shape[1] else np.zeros(n, dtype=bool)
            pred = np.full(n, np.nan, dtype=np.float32)
            if ok.any():
                pred[ok] = np.asarray(model.predict(features[ok]), dtype=np.float32)
        else:
            pred = np.asarray(model.predict(features), dtype=np.float32)
        preds.append(pred)
    if not preds:
        return np.full(n, np.nan, dtype=np.float32)
    if len(preds) == 1:
        return preds[0]
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.column_stack(preds), axis=1).astype(np.float32)


def predict_served(
    served: ServedModel,
    panel: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    cache=None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """在给定面板上推理，返回 ``date, code, score`` 长表（可裁剪到 [start, end]）。

    幂等：同模型同数据 → 同分数（无随机性）。分数已按 ``smooth_window`` 平滑，
    与训练脚本写 pred 的口径一致。
    """
    dates = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    scores_by_group: dict[str, np.ndarray] = {}
    for key in served.group_keys():
        group = served.groups[key]
        if progress:
            progress(f"推理频段组 {key}（{len(group.feature_names)} 个入模特征）")
        features = _group_matrix(served, group, panel, cache=cache)
        scores_by_group[key] = _predict_group(group, features)

    blended = blend_group_scores_fixed(scores_by_group, served.blend_weights, dates)
    if served.smooth_window and int(served.smooth_window) > 1:
        blended = wma_smooth_scores(blended, panel, int(served.smooth_window))
    if progress:
        progress(
            "融合完成："
            + "、".join(f"{k}={served.blend_weights.get(k, 0.0):.3f}" for k in served.group_keys())
        )

    frame = to_pred_frame(blended, panel)
    if frame.empty:
        return frame
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    return frame.reset_index(drop=True)


def latest_score_date(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty:
        return None
    return str(pd.Timestamp(frame["date"].max()).date())
