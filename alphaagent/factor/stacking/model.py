"""Stacking 模型：Ridge / LightGBM 统一接口 + walk-forward 训练与 OOS 预测。

时间隔离：所有 fold 的训练样本起点严格晚于 ``mining_end``；每折 train 与
OOS 之间留 ``purge_days`` 个交易日的 purge gap（≥ 标签期），消除前向标签
跨折泄漏。OOS 预测拼接后即组合分数（不用 in-sample 拟合值出分）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

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


def _portfolio_risk_metrics(
    pred_oos: np.ndarray,
    label_oos: np.ndarray,
    dates_oos: pd.Series,
    *,
    horizon_days: int = 5,
    top_frac: float = 0.2,
    min_samples: int = 8,
) -> dict:
    """轻量组合风险口径（相对比较用，非 engine_gate 净超额口径）。

    OOS 段逐日按预测取前 ``top_frac`` 等权多头，组合收益 = 该层前向标签收益；
    label 为 ``horizon_days`` 个交易日的前向收益，逐日序列相互重叠会高估
    Sharpe，故每 ``horizon_days`` 个交易日取一个非重叠样本再计算年化 Sharpe
    与最大回撤。样本不足（< min_samples 个非重叠期）返回 None 字段。
    """
    empty = {"oos_sharpe": None, "oos_max_drawdown": None}
    df = pd.DataFrame({"p": pred_oos, "y": label_oos, "d": pd.Series(dates_oos).to_numpy()}).dropna()
    if df.empty:
        return empty
    rank_pct = df.groupby("d")["p"].rank(pct=True)
    day_ret = df[rank_pct >= 1 - top_frac].groupby("d")["y"].mean().sort_index()
    if len(day_ret) < horizon_days * min_samples:
        return empty
    r = day_ret.to_numpy()[::horizon_days]
    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(np.mean(r) / std * np.sqrt(252.0 / horizon_days)) if std > 1e-12 else None
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    mdd = float(np.min(equity / peak - 1.0))  # ≤0，与 engine_gate 回撤同号
    return {
        "oos_sharpe": None if sharpe is None else round(sharpe, 4),
        "oos_max_drawdown": round(mdd, 5),
    }


def fit_predict_walkforward(
    feature_matrix: np.ndarray,
    label: np.ndarray,
    dates: pd.Series,
    folds: Sequence[WalkForwardFold],
    *,
    kind: ModelKind,
    feature_names: Sequence[str] | None = None,
    collect_diagnostics: bool = True,
    label_horizon: int = 5,
) -> tuple[np.ndarray, list[dict], list[dict] | None, list[dict] | None]:
    """逐折训练 + OOS 预测；返回 (拼接 OOS 预测 [n_rows]（折外 NaN）, 逐折指标, 特征权重, 置换贡献)。

    训练行过滤：日期在折 train_dates 内、label 有限；ridge 另要求特征全有限
    （无缺失处理），lgbm 原生支持 NaN 特征 → 放行，保住基本面/事件等稀疏面
    的样本行（原先整行丢弃会造成训练集缩水与宇宙偏化）。预测原样写回对应行，
    不做截面再标准化（模型输出经 OOS 评估，engine_gate 直接消费）。

    特征权重：每折提取重要性并归一为份额后跨折平均——ridge 取 |coef|（特征
    已截面 zscore，系数可比），lgbm 取 gain 重要性；返回按权重降序的
    [{name, weight}]（weight 合计 1），feature_names 缺失时为 None。

    置换贡献：逐折在 OOS 段把单个特征行内打乱后重预测，IC 相对未打乱的
    下降量跨折平均——"没有这个因子组合损失多少 IC"。同一份打乱预测上顺便
    计算组合风险口径（_portfolio_risk_metrics）：sharpe_drop = 基线 Sharpe −
    打乱后 Sharpe（正 = 该因子在贡献收益稳定性）；dd_impact = 打乱后回撤
    幅度 − 基线回撤幅度（正 = 打乱后回撤更深 ⇒ 该因子在压回撤；负 = 该
    因子在放大回撤）。返回
    [{name, ic_drop, ic_drop_rel, sharpe_drop, dd_impact}] 按 ic_drop 降序；
    ic_drop_rel = ic_drop / 组合平均 IC（≈ 该因子驱动的组合 IC 份额）；负值
    = 打乱后反而更好（该因子在拖后腿）。lgbm 侧该值近似线性化贡献，ridge
    侧精确。

    collect_diagnostics=False（累积曲线等批量重训场景）时跳过权重与置换
    采集——置换是 O(n_features) 次重预测，批量场景下是平方级浪费。
    """
    n_rows = feature_matrix.shape[0]
    oos_pred = np.full(n_rows, np.nan, dtype=np.float32)
    date_np = pd.to_datetime(pd.Series(dates)).to_numpy()
    report: list[dict] = []
    require_finite = kind == "ridge"
    weight_acc: list[np.ndarray] = []
    contrib_acc: list[dict] = []
    for fold in folds:
        train_mask = np.isin(date_np, fold.train_dates.to_numpy())
        valid = train_mask & np.isfinite(label)
        if require_finite and feature_matrix.shape[1]:
            valid &= np.isfinite(feature_matrix).all(axis=1)
        oos_mask = np.isin(date_np, fold.oos_dates.to_numpy())
        oos_ok = oos_mask & np.isfinite(label)
        if require_finite and feature_matrix.shape[1]:
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
        if collect_diagnostics and feature_names and feature_matrix.shape[1]:
            try:
                if kind == "ridge":
                    raw_w = np.abs(np.asarray(model.coef_, dtype=np.float64))
                else:
                    raw_w = np.asarray(
                        model.booster_.feature_importance(importance_type="gain"), dtype=np.float64
                    )
                total = float(raw_w.sum())
                if total > 0:
                    weight_acc.append(raw_w / total)
            except Exception:  # noqa: BLE001 — 权重采集失败不影响训练主流程
                pass
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

        # ── 置换贡献：逐特征打乱 OOS 行后重预测，IC / Sharpe / 回撤 三口径下降量 ──
        if collect_diagnostics and feature_names and feature_matrix.shape[1] and metrics.get("ic_mean") is not None:
            rng = np.random.default_rng(20260907)
            Xo = feature_matrix[oos_ok]
            yo = label[oos_ok]
            do = pd.Series(date_np[oos_ok])
            base_ic = float(metrics["ic_mean"])
            base_risk = _portfolio_risk_metrics(pred, yo, do, horizon_days=label_horizon)
            base_sharpe = base_risk["oos_sharpe"]
            base_mdd = abs(base_risk["oos_max_drawdown"]) if base_risk["oos_max_drawdown"] is not None else None
            drops: dict[str, float] = {}
            sharpe_drops: dict[str, float | None] = {}
            dd_impacts: dict[str, float | None] = {}
            for j, name in enumerate(feature_names):
                Xp = Xo.copy()
                rng.shuffle(Xp[:, j])
                pred_p = np.asarray(model.predict(Xp), dtype=np.float32)
                ic_p = daily_spearman_ic(pred_p, yo, do)
                drops[str(name)] = base_ic - (float(ic_p.mean()) if len(ic_p) else 0.0)
                risk_p = _portfolio_risk_metrics(pred_p, yo, do, horizon_days=label_horizon)
                sharpe_drops[str(name)] = (
                    base_sharpe - risk_p["oos_sharpe"]
                    if base_sharpe is not None and risk_p["oos_sharpe"] is not None
                    else None
                )
                dd_impacts[str(name)] = (
                    abs(risk_p["oos_max_drawdown"]) - base_mdd
                    if base_mdd is not None and risk_p["oos_max_drawdown"] is not None
                    else None
                )
            contrib_acc.append({
                "base_ic": base_ic,
                "drops": drops,
                "sharpe_drops": sharpe_drops,
                "dd_impacts": dd_impacts,
            })

    feature_weights: list[dict] | None = None
    if feature_names and weight_acc:
        mean_w = np.mean(np.asarray(weight_acc, dtype=np.float64), axis=0)
        feature_weights = sorted(
            ({"name": str(n), "weight": round(float(w), 6)} for n, w in zip(feature_names, mean_w)),
            key=lambda x: -x["weight"],
        )

    feature_contribution: list[dict] | None = None
    if feature_names and contrib_acc:
        base_ic = float(np.mean([c["base_ic"] for c in contrib_acc]))

        def _mean_nonull(key: str, name: str) -> float | None:
            vals = [c[key][str(name)] for c in contrib_acc if c[key].get(str(name)) is not None]
            return float(np.mean(vals)) if vals else None

        feature_contribution = sorted(
            (
                {
                    "name": n,
                    "ic_drop": round(ic_drop, 6) if (ic_drop := _mean_nonull("drops", n)) is not None else 0.0,
                    "ic_drop_rel": round(ic_drop / base_ic, 4) if (ic_drop is not None and abs(base_ic) > 1e-12) else None,
                    "sharpe_drop": round(v, 4) if (v := _mean_nonull("sharpe_drops", n)) is not None else None,
                    "dd_impact": round(v, 5) if (v := _mean_nonull("dd_impacts", n)) is not None else None,
                }
                for n in feature_names
            ),
            key=lambda x: -x["ic_drop"],
        )
    return oos_pred, report, feature_weights, feature_contribution


def cumulative_subset_curve(
    feature_matrix: np.ndarray,
    label: np.ndarray,
    dates: pd.Series,
    folds: Sequence[WalkForwardFold],
    *,
    ranked_names: Sequence[str],
    feature_names: Sequence[str],
    kinds: Sequence[ModelKind],
    first_oos=None,
    label_horizon: int = 5,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """贡献排序累积子集曲线：按贡献降序取 Top-k 特征逐级重训，返回每 k 的 OOS IC。

    ranked_names：完整特征名按置换贡献降序排列（来自 fit_predict_walkforward
    的第 4 返回值）。每 k 对每个 kind 重训一遍（collect_diagnostics=False，
    避免 O(n²) 次置换重预测）。IC 统一口径 = 主训练 _blended_oos_ic：date ≥
    first_oos（给了 first_oos 时）且 pred/label 有限值的行。blended 预测上
    顺便计算组合 OOS Sharpe 与最大回撤（_portfolio_risk_metrics，轻量口径）。
    返回 [{"k", per-kind IC..., "blended", "oos_sharpe", "oos_max_drawdown"}]。
    """
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    order = [name_to_idx[str(n)] for n in ranked_names if str(n) in name_to_idx]
    date_np = pd.to_datetime(pd.Series(dates)).to_numpy()

    def _mask(pred: np.ndarray) -> np.ndarray:
        mask = np.isfinite(pred) & np.isfinite(label)
        if first_oos is not None:
            mask &= date_np >= pd.Timestamp(first_oos)
        return mask

    def _ic(pred: np.ndarray) -> float | None:
        mask = _mask(pred)
        if mask.sum() < 20:
            return None
        ic = daily_spearman_ic(pred[mask], label[mask], pd.Series(date_np[mask]))
        return round(float(ic.mean()), 5) if len(ic) else None

    def _risk(pred: np.ndarray) -> dict:
        mask = _mask(pred)
        if mask.sum() < 20:
            return {"oos_sharpe": None, "oos_max_drawdown": None}
        return _portfolio_risk_metrics(
            pred[mask], label[mask], pd.Series(date_np[mask]), horizon_days=label_horizon
        )

    curve: list[dict] = []
    for k in range(1, len(order) + 1):
        idx = order[:k]
        row: dict = {"k": k, "names": [str(feature_names[i]) for i in idx]}
        preds: list[np.ndarray] = []
        for kind in kinds:
            p, _rep, _w, _c = fit_predict_walkforward(
                feature_matrix[:, idx], label, dates, folds, kind=kind, collect_diagnostics=False
            )
            row[kind] = _ic(p)
            preds.append(p)
        usable = [p for p in preds if np.isfinite(p).any()]
        if usable:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                blended = np.nanmean(np.column_stack(usable), axis=1).astype(np.float32)
            row["blended"] = _ic(blended)
            row.update(_risk(blended))
        else:
            row["blended"] = None
            row["oos_sharpe"] = None
            row["oos_max_drawdown"] = None
        curve.append(row)
        if progress:
            risk_txt = ""
            if row.get("oos_sharpe") is not None:
                risk_txt = f"，Sharpe={row['oos_sharpe']}，回撤={row['oos_max_drawdown']}"
            progress(f"[{k}/{len(order)}] Top{k} 子集 blended OOS IC = {row['blended']}{risk_txt}")
    return curve
