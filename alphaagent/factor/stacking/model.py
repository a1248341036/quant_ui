"""Stacking 模型：Ridge / LightGBM 统一接口 + walk-forward 训练与 OOS 预测。

时间隔离：所有 fold 的训练样本起点严格晚于 ``mining_end``；每折 train 与
OOS 之间留 ``purge_days`` 个交易日的 purge gap（≥ 标签期），消除前向标签
跨折泄漏。OOS 预测拼接后即组合分数（不用 in-sample 拟合值出分）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .dataset import daily_spearman_ic

ModelKind = Literal["ridge", "lgbm"]

_MODEL_KINDS = ("ridge", "lgbm")


def make_model(kind: ModelKind):
    """构建模型实例。特征已截面 zscore，Ridge 用强正则。"""
    if kind == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=10.0)
    if kind == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=200,
            num_leaves=15,
            max_depth=4,
            learning_rate=0.05,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"unknown model kind: {kind}（可选 {_MODEL_KINDS}）")


@dataclass(frozen=True)
class WalkForwardFold:
    train_dates: pd.DatetimeIndex
    oos_dates: pd.DatetimeIndex


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    *,
    train_start: pd.Timestamp,
    train_months: int = 18,
    step_months: int = 6,
    purge_days: int = 5,
) -> list[WalkForwardFold]:
    """expanding 窗口 walk-forward 折。

    训练样本仅取 ``>= train_start``（时间隔离边界）；第 i 折 OOS 为
    ``[train_start + (train_months + i*step_months), +step_months)``，
    train 侧剔除 OOS 前最后 ``purge_days`` 个交易日（前向标签跨折泄漏）。
    OOS 不足一个交易日的不成折。
    """
    trading_days = dates.unique().sort_values()
    start = pd.Timestamp(train_start)
    folds: list[WalkForwardFold] = []
    i = 0
    while True:
        oos_start = start + pd.DateOffset(months=train_months + i * step_months)
        oos_end = oos_start + pd.DateOffset(months=step_months)
        oos = trading_days[(trading_days >= oos_start) & (trading_days < oos_end)]
        if oos.empty:
            break
        train = trading_days[(trading_days >= start) & (trading_days < oos[0])]
        if purge_days > 0 and len(train) > purge_days:
            train = train[:-purge_days]
        if len(train) >= 20:
            folds.append(WalkForwardFold(train_dates=pd.DatetimeIndex(train), oos_dates=pd.DatetimeIndex(oos)))
        i += 1
    return folds


def _fold_metrics(
    pred_oos: np.ndarray,
    label_oos: np.ndarray,
    dates_oos: pd.Series,
) -> dict:
    ic = daily_spearman_ic(pred_oos, label_oos, dates_oos)
    df = pd.DataFrame({"p": pred_oos, "y": label_oos, "d": dates_oos.to_numpy()}).dropna()
    spread = None
    if not df.empty:
        # 逐日 top20% - bottom20% 平均前向收益（多头-空头价差，日频口径）
        pr = df.groupby("d")["p"].rank(pct=True)
        top = df[pr >= 0.8].groupby("d")["y"].mean()
        bot = df[pr <= 0.2].groupby("d")["y"].mean()
        spread_series = (top - bot).dropna()
        spread = float(spread_series.mean()) if len(spread_series) else None
    return {
        "ic_mean": float(ic.mean()) if len(ic) else None,
        "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 1e-12 else None,
        "n_days": int(len(ic)),
        "long_short_daily_spread": spread,
    }


def fit_predict_walkforward(
    feature_matrix: np.ndarray,
    label: np.ndarray,
    dates: pd.Series,
    folds: Sequence[WalkForwardFold],
    *,
    kind: ModelKind,
) -> tuple[np.ndarray, list[dict]]:
    """逐折训练 + OOS 预测；返回 (拼接的 OOS 预测 [n_rows]（折外为 NaN）, 逐折指标)。

    训练行过滤：日期在折 train_dates 内、label 与特征均有限。预测原样写回
    对应行，不做截面再标准化（模型输出经 OOS 评估，engine_gate 直接消费）。
    """
    n_rows = feature_matrix.shape[0]
    oos_pred = np.full(n_rows, np.nan, dtype=np.float32)
    date_np = pd.to_datetime(pd.Series(dates)).to_numpy()
    report: list[dict] = []
    for fold in folds:
        train_mask = np.isin(date_np, fold.train_dates.to_numpy())
        valid = train_mask & np.isfinite(label)
        if feature_matrix.shape[1]:
            valid &= np.isfinite(feature_matrix).all(axis=1)
        oos_mask = np.isin(date_np, fold.oos_dates.to_numpy())
        oos_ok = oos_mask & np.isfinite(label)
        if feature_matrix.shape[1]:
            oos_ok &= np.isfinite(feature_matrix).all(axis=1)
        if valid.sum() < 100 or oos_ok.sum() == 0:
            report.append({
                "oos_start": str(fold.oos_dates.min().date()),
                "oos_end": str(fold.oos_dates.max().date()),
                "n_train": int(valid.sum()),
                "skipped": True,
            })
            continue
        model = make_model(kind)
        model.fit(feature_matrix[valid], label[valid])
        pred = np.asarray(model.predict(feature_matrix[oos_ok]), dtype=np.float32)
        oos_pred[oos_ok] = pred
        metrics = _fold_metrics(
            pred, label[oos_ok], pd.Series(date_np[oos_ok])
        )
        report.append({
            "oos_start": str(fold.oos_dates.min().date()),
            "oos_end": str(fold.oos_dates.max().date()),
            "n_train": int(valid.sum()),
            "n_oos": int(oos_ok.sum()),
            **metrics,
        })
    return oos_pred, report
