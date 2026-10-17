# -*- coding: utf-8 -*-
"""冷门赛道残差化评估（spec: alphaagent_mining_anti_convergence_spec.md M1 / §3.3）。

只在 evaluation 层的数值回归实现，不用 DSL CS_RESIDUALIZE（DSL 限 2 个控制变量，
骨架族最多取到 N 个）。核心纪律（评审 P1 #1 / P2 #8）：

- 只在 train 段逐截面 OLS 记录每日系数 beta_hat，池化平均（与逐日 IC 口径一致）；
- 用同一 beta_hat 分别残差化 train 与 val 两段 -> train_resid_ic（轻度内样）与
  val_resid_ic（真样本外）双披露，过门以 val_resid_ic 为主键；
- 披露三键：resid_ic(=val_resid_ic) / crowd_proxy_corr / family_novelty。

本模块纯数值 + 纯函数（不碰 session/registry），输入输出均为可序列化 dict，
便于 dispatch 层短路调用与单测。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.factor.metrics import (
    cs_ic_summary,
    cross_sectional_ic,
    cross_sectional_rank_ic,
)
from alphaagent.factor.metrics._core import coverage as _coverage

logger = logging.getLogger(__name__)


# 残差化控制列覆盖过低（相对于当日存续股票数）时跳过该日的安全下限。
_MIN_CTRL_COVER = 0.5


def _slice_bounds(dt_level: pd.Index) -> list[tuple[int, int]]:
    """单调 datetime index -> 连续等值段的 [start, end) 边界列表。"""
    vals = dt_level.to_numpy()
    bounds: list[tuple[int, int]] = []
    if len(vals) == 0:
        return bounds
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            bounds.append((start, i))
            start = i
    return bounds


def pool_residual_betas(
    y: np.ndarray,
    x_ctrl: np.ndarray,
    bounds: list[tuple[int, int]],
    *,
    min_rows: int = 12,
    min_ctrl_cover: float = _MIN_CTRL_COVER,
) -> tuple[np.ndarray, int]:
    """逐截面 OLS（y ~ 1 + 控制列），系数按有效日平均（spec §9 Q1）。

    返回 (beta_pooled, n_used_days)。beta_pooled 形状 (k+1,)：首项为截距。
    某日控制矩阵覆盖低于 min_ctrl_cover 或有效样本不足 min_rows 时跳过该日
    （不贡献池化系数，避免小样本截面把 beta_hat 拉偏）。
    """
    n, k = x_ctrl.shape
    if n == 0 or k == 0:
        return np.zeros(k + 1), 0
    yf = y.astype(np.float64)
    xf = x_ctrl.astype(np.float64)
    betas: list[np.ndarray] = []
    n_used = 0
    for st, en in bounds:
        y_day = yf[st:en]
        x_day = xf[st:en]
        good = np.isfinite(y_day) & np.all(np.isfinite(x_day), axis=1)
        m = int(np.count_nonzero(good))
        if m < min_rows:
            continue
        if m / max(1, len(y_day)) < min_ctrl_cover:
            continue
        y_ok = y_day[good]
        x_ok = x_day[good]
        if k + 1 > m:
            continue
        try:
            beta, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(m), x_ok]), y_ok, rcond=None
            )
        except np.linalg.LinAlgError:  # pragma: no cover - 数值保护
            continue
        if not np.all(np.isfinite(beta)):
            continue
        betas.append(beta)
        n_used += 1
    if not betas:
        return np.zeros(k + 1), 0
    return np.mean(np.vstack(betas), axis=0), n_used


def apply_residual(
    y: np.ndarray,
    x_ctrl: np.ndarray,
    beta_pooled: np.ndarray,
) -> np.ndarray:
    """用固定 beta_pooled 残差化：y - (beta0 + X @ beta_rest)。"""
    yf = y.astype(np.float64)
    xf = x_ctrl.astype(np.float64)
    out = np.full(len(yf), np.nan, dtype=np.float64)
    if len(yf) == 0 or xf.shape[1] == 0 or beta_pooled is None:
        return out
    good = np.all(np.isfinite(xf), axis=1)
    if not np.any(good):
        return out
    design = np.column_stack([np.ones(int(np.count_nonzero(good))), xf[good]])
    fitted = design @ beta_pooled.astype(np.float64)
    out[good] = yf[good] - fitted
    return out


def _daily_stats(
    resid: pd.Series,
    label: pd.Series,
    *,
    min_pairs: int,
    time_level: str = "datetime",
) -> dict[str, Any]:
    """对残差序列算逐日 IC / ICIR / 覆盖（复用既有 metrics 口径）。"""
    daily = cross_sectional_ic(resid, label, time_level=time_level, min_pairs=min_pairs)
    daily_rank = cross_sectional_rank_ic(
        resid, label, time_level=time_level, min_pairs=min_pairs
    )
    summary = cs_ic_summary(daily, daily_rank)
    return {
        "ic": float(summary["ic"]),
        "icir": float(summary["icir"]),
        "rank_ic": float(summary["rank_ic"]),
        "n_days": int(summary["n_days"]),
        "coverage": float(_coverage(resid.to_numpy(dtype=np.float64, copy=False))),
    }


def crowd_proxy_corr(
    factor: pd.Series,
    ctrl_series: dict[str, pd.Series],
    *,
    min_pairs: int = 10,
    time_level: str = "datetime",
) -> tuple[float, str | None]:
    """候选原始信号与各族骨架代表的最大截面 Spearman 相关。

    对每族算逐日 rank_ic（复用 cross_sectional_rank_ic 口径），取绝对最大均值；
    返回 (max_corr, family)。label 不参与相似度度量。
    """
    best: float = 0.0
    best_family: str | None = None
    fl = factor.astype(np.float64)
    for fam, ctrl in ctrl_series.items():
        if ctrl is None or len(ctrl) == 0:
            continue
        try:
            daily = cross_sectional_rank_ic(
                fl, ctrl.astype(np.float64), time_level=time_level, min_pairs=min_pairs
            )
        except Exception:  # noqa: BLE001 - 某族数据异常不影响其他族
            logger.debug("crowd_proxy_corr skip family %r (data anomaly)", fam)
            continue
        vals = daily.dropna()
        if vals.empty:
            continue
        corr = float(abs(np.mean(vals.to_numpy(dtype=np.float64))))
        if corr > best:
            best = corr
            best_family = fam
    return best, best_family


def classify_missing(
    factor_values: np.ndarray,
    ctrl_values: dict[str, np.ndarray],
    *,
    min_n_datasets: int = 2,
) -> str | None:
    """原料缺失判定：候选或骨架覆盖过低 -> "data_gap"；否则 None。"""
    if factor_values is None or len(factor_values) == 0:
        return "data_gap"
    if float(_coverage(factor_values.astype(np.float64))) < 0.2:
        return "data_gap"
    present = [v for v in ctrl_values.values() if v is not None and len(v)]
    if len(present) < min_n_datasets:
        return "data_gap"
    return None


def evaluate_cold_track(
    factor_values: np.ndarray,
    panel: pd.DataFrame,
    ctrl_values: dict[str, np.ndarray],
    ctx: Any,
    *,
    label_col: str | None = None,
    policy: dict[str, Any] | None = None,
    min_pairs: int = 10,
    family_novelty: bool = True,
) -> dict[str, Any]:
    """冷门赛道核心评估：拟合 beta -> 残差化 -> 双段 IC -> proxy corr。

    输入约定（由 dispatch 组装）：
      factor_values：与 panel 同序的候选因子值（float ndarray 或 None）；
      panel        ：会话面板（MultiIndex datetime x instrument，含 label_col）；
      ctrl_values  ：{family: ndarray} 每族骨架代表信号（与 panel 同序）；
      ctx          ：提供 train_start/end、val_start/end；
      policy       ：cold_track 配置 dict（min_val_resid_ic / max_crowd_proxy_corr）。

    返回可序列化 dict（三键披露 + fail_detail）。不抛异常：细节缺失时返回
    fail_detail 标记而不是炸断 dispatch。
    """
    policy = policy or {}
    if factor_values is None or len(panel) == 0:
        return {
            "ok": False,
            "reason": "no_panel",
            "resid_ic": float("nan"),
            "crowd_proxy_corr": float("nan"),
            "family_novelty": bool(family_novelty),
        }
    miss = classify_missing(factor_values, ctrl_values)
    if miss:
        return {
            "ok": False,
            "reason": "data_gap",
            "fail_detail": miss,
            "resid_ic": float("nan"),
            "crowd_proxy_corr": float("nan"),
            "family_novelty": bool(family_novelty),
        }

    label_col = label_col or getattr(ctx, "label_col", None) or "label_1d_open_to_open"
    if label_col not in panel.columns:
        return {
            "ok": False,
            "reason": "missing_label",
            "fail_detail": "data_gap",
            "resid_ic": float("nan"),
            "crowd_proxy_corr": float("nan"),
            "family_novelty": bool(family_novelty),
        }

    dt_np = pd.to_datetime(panel.index.get_level_values("datetime"))
    train_start = str(getattr(ctx, "train_start", "") or "")
    train_end = str(getattr(ctx, "train_end", "") or "")
    val_start = str(getattr(ctx, "val_start", "") or "")
    val_end = str(getattr(ctx, "val_end", "") or "")
    train_mask = np.ones(len(panel), dtype=bool)
    val_mask = np.ones(len(panel), dtype=bool)
    if train_start and train_end:
        train_mask = (dt_np >= pd.Timestamp(train_start)) & (dt_np <= pd.Timestamp(train_end))
    if val_start and val_end:
        val_mask = (dt_np >= pd.Timestamp(val_start)) & (dt_np <= pd.Timestamp(val_end))

    fams: list[str] = []
    cols: list[np.ndarray] = []
    for fam, arr in ctrl_values.items():
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float64)
        if len(a) != len(panel):
            continue
        fams.append(fam)
        cols.append(a)
    if not cols:
        return {
            "ok": False,
            "reason": "no_controls",
            "fail_detail": "data_gap",
            "resid_ic": float("nan"),
            "crowd_proxy_corr": float("nan"),
            "family_novelty": bool(family_novelty),
        }
    x_ctrl = np.column_stack(cols)

    full_bounds = _slice_bounds(dt_np)
    train_bounds = [
        (st, en) for st, en in full_bounds
        if train_mask[st] and train_mask[en - 1]
    ]
    beta, n_fit_days = pool_residual_betas(
        factor_values.astype(np.float64),
        x_ctrl,
        train_bounds,
        min_rows=max(min_pairs, 6),
    )
    if n_fit_days < max(2, int(0.1 * max(1, len(train_bounds)))):
        return {
            "ok": False,
            "reason": "underpowered_fit",
            "fail_detail": "underpowered",
            "resid_ic": float("nan"),
            "crowd_proxy_corr": float("nan"),
            "family_novelty": bool(family_novelty),
        }

    resid_full = apply_residual(factor_values.astype(np.float64), x_ctrl, beta)
    resid_series = pd.Series(resid_full, index=panel.index, dtype=np.float64)
    label_series = panel[label_col].astype(np.float64)

    train_stats: dict[str, Any] = {"ic": float("nan"), "n_days": 0}
    val_stats: dict[str, Any] = {"ic": float("nan"), "n_days": 0}
    if np.any(train_mask):
        train_stats = _daily_stats(resid_series[train_mask], label_series[train_mask], min_pairs=min_pairs)
    if np.any(val_mask):
        val_stats = _daily_stats(resid_series[val_mask], label_series[val_mask], min_pairs=min_pairs)

    ctrl_series = {
        fam: pd.Series(c, index=panel.index, dtype=np.float64) for fam, c in zip(fams, cols)
    }
    visible = train_mask | val_mask
    proxy, proxy_family = crowd_proxy_corr(
        pd.Series(
            factor_values.astype(np.float64), index=panel.index, dtype=np.float64
        )[visible],
        {f: s[visible] for f, s in ctrl_series.items()},
        min_pairs=min_pairs,
    )

    val_ic = float(val_stats.get("ic", float("nan")))
    min_val_ic = float(policy.get("min_val_resid_ic", 0.02))
    max_corr = float(policy.get("max_crowd_proxy_corr", 0.3))
    passed = (
        np.isfinite(val_ic)
        and abs(val_ic) >= min_val_ic
        and np.isfinite(proxy)
        and proxy < max_corr
    )

    fail_detail: str | None = None
    if not passed:
        train_ic = float(train_stats.get("ic", float("nan")))
        if not (np.isfinite(train_ic) and np.isfinite(val_ic)):
            fail_detail = "underpowered"
        elif np.sign(train_ic) != np.sign(val_ic) or abs(val_ic) < abs(train_ic) * 0.5:
            fail_detail = "underpowered"
        else:
            fail_detail = "no_alpha"

    return {
        "ok": True,
        "passed": bool(passed),
        "resid_ic": val_ic,
        "train_resid_ic": float(train_stats.get("ic", float("nan"))),
        "train_resid_icir": float(train_stats.get("icir", float("nan"))),
        "val_resid_icir": float(val_stats.get("icir", float("nan"))),
        "val_resid_rank_ic": float(val_stats.get("rank_ic", float("nan"))),
        "train_n_days": int(train_stats.get("n_days", 0)),
        "val_n_days": int(val_stats.get("n_days", 0)),
        "coverage": float(val_stats.get("coverage", float("nan"))),
        "crowd_proxy_corr": float(proxy if np.isfinite(proxy) else float("nan")),
        "crowd_proxy_family": proxy_family,
        "family_novelty": bool(family_novelty),
        "n_fit_days": int(n_fit_days),
        "n_controls": int(len(fams)),
        "min_val_resid_ic": min_val_ic,
        "max_crowd_proxy_corr": max_corr,
        "fail_detail": fail_detail,
    }
