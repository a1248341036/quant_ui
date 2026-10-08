"""Null Importance 因子预筛选（Stage 0：训练前辨真伪）。

原理与动机
----------
树模型的 gain importance 天生被特征自身分布属性污染：连续值/高基数因子
可切分点多，即使与标签毫无关系也能刷出很高的分裂增益。直接按 importance
排名选因子，会把"分布属性好"误判成"有真信息"。

Null Importance 给每个因子配一个专属噪声对照组：

1. 真实跑：X, y（真标签）→ 训练 → 记录每个因子的 gain importance（actual）；
2. 噪声跑：X 原封不动、y 行内随机打乱 → 训练 → 记录 importance（null）；
3. 重复 2 共 ``NULL_IMPORTANCE_RUNS`` 次 → 每个因子得到 null 分布；
4. 打分 ``score = log(1e-10 + actual / (1 + P75(null)))``：
   score > 0 ⇔ 真实重要性超过噪声本底的 75 分位。

打乱标签后特征与标签的真实关系被彻底切断，但特征自身的分布属性原样
保留——null 跑出来的重要性**全部**来自分布属性，正是要扣除的本底。
本质是对每个因子单独做置换检验（H0：该因子与标签无关），天然抗
"多重检验幸存因子"与"分布属性凑出来的假 IC"。

时间隔离契约（与 stacking 包一致）
----------------------------------
筛选窗口只能落在训练段内（调用方传 ``window_start/window_end``），
验证段与盲测段绝不参与筛选决策。推荐窗口 = [panel 起点, mining_end)。

模型选择
--------
LightGBM Random Forest 模式（boosting_type='rf'）：多树平均让重要性比
GBDT 稳定，且不受 boosting 轮数过拟合污染。筛选回答"有没有真信息"，
与最终组合训练模型（Ridge/GBDT）刻意解耦——筛选模型的职责是公平地
给每个因子一个"against 噪声本底"的检验，不是预测。
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

# ── 配置中心（阈值收口；train_ml_composite CLI 可覆盖）─────────────────────
# 含义：打乱标签重跑的次数，构成每个因子的 null importance 分布。
# 为什么 80：Kaggle 实战与文献（Boruta 同类法）验证的充分量——75 分位在
# n≥40 后趋于稳定；调小（如 20）提速但分位数抖动、临界因子判定不稳；
# 调大（如 200）只轻微收窄置信区间，成本线性增加。影响所有 Null Importance
# 筛选（ML 组合 Stage 0）。
NULL_IMPORTANCE_RUNS = 80

# 含义：score = log(actual / (1 + null_75分位)) 的通过线。
# 为什么 0：score>0 数学上等价于"真实重要性超过噪声本底 75 分位"——这是
# "该因子与标签的关系强于纯分布属性噪声"的最低证据；调大（如 0.5）更严，
# 剔除更多临界因子（真实 IC 低但非零的因子可能被误杀）；调负（如 -0.5）
# 放宽，接近只剔"完全无信号"因子。影响 ML 组合 Stage 0 的白名单宽度。
SCREEN_MIN_SCORE = 0.0

# LightGBM RF 模式参数（筛选专用，与 make_model('lgbm') 的 GBDT 刻意解耦）：
# boosting_type='rf' 多树平均使 importance 稳定；bagging_fraction<1 为 rf
# 模式硬性要求；num_leaves/max_depth 与组合训练 GBDT 同量级（容量足够表达
# 非线性，但不至于让单棵树过拟合稀释因子间差异）。
_RF_PARAMS = {
    "objective": "regression",
    "boosting_type": "rf",
    "num_leaves": 31,
    "max_depth": 8,
    "bagging_fraction": 0.623,
    "bagging_freq": 1,
    "feature_fraction": 0.7,
    "min_child_samples": 100,
    "learning_rate": 1.0,  # rf 模式下树独立拟合，lr 仅形式收缩，取 1
    "verbose": -1,
}

# 真实跑与 null 跑的随机种子基数（固定值保证筛选结果可复现）
_NULL_SEED_BASE = 20261008


def _rf_importance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_boost_round: int,
    seed: int,
) -> np.ndarray:
    """一次 RF 训练的逐因子 gain importance（长度 = K）。"""
    import lightgbm as lgb

    params = {**_RF_PARAMS, "seed": int(seed), "feature_fraction_seed": int(seed),
              "bagging_seed": int(seed)}
    dtrain = lgb.Dataset(X, label=y)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    return np.asarray(
        booster.feature_importance(importance_type="gain"), dtype=np.float64
    )


def null_importance_scores(
    feature_matrix: np.ndarray,
    label: np.ndarray,
    dates: pd.Series,
    feature_names: Sequence[str],
    *,
    window_start,
    window_end,
    n_runs: int = NULL_IMPORTANCE_RUNS,
    min_score: float = SCREEN_MIN_SCORE,
    num_boost_round: int = 200,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """在训练段窗口 [window_start, window_end) 上做 Null Importance 筛选。

    返回按 score 降序的
    ``[{name, actual, null_p75, score, passed}, ...]``；
    ``passed = score > min_score``。样本行 = 窗口内且 label 有限
    （LightGBM 原生支持 NaN 特征，稀疏面因子不丢行）。
    """
    names = [str(n) for n in feature_names]
    K = feature_matrix.shape[1]
    if len(names) != K:
        raise ValueError(f"feature_names 数量({len(names)})与特征列数({K})不一致")
    if n_runs < 5:
        raise ValueError(f"n_runs 至少 5（75 分位需足够样本），收到 {n_runs}")

    date_np = pd.to_datetime(pd.Series(dates)).to_numpy()
    start = np.datetime64(pd.Timestamp(window_start))
    end = np.datetime64(pd.Timestamp(window_end))
    win = (date_np >= start) & (date_np < end) & np.isfinite(label)
    n_valid = int(win.sum())
    if n_valid < 200 or K == 0:
        # 样本不足：无法区分真信号与本底，全部判 fail（宁缺毋滥）
        return [
            {"name": n, "actual": None, "null_p75": None, "score": None,
             "passed": False, "reason": f"insufficient_window_samples={n_valid}"}
            for n in names
        ]

    Xw = feature_matrix[win]
    yw = label[win].astype(np.float64)

    # ① 真实重要性（1 次）
    actual = _rf_importance(Xw, yw, num_boost_round=num_boost_round, seed=_NULL_SEED_BASE)

    # ②③ null 分布：只打乱 y，X 一动不动
    null_mat = np.empty((int(n_runs), K), dtype=np.float64)
    for i in range(int(n_runs)):
        rng = np.random.default_rng(_NULL_SEED_BASE + 1 + i)
        y_null = yw[rng.permutation(yw.shape[0])]
        null_mat[i] = _rf_importance(
            Xw, y_null, num_boost_round=num_boost_round, seed=_NULL_SEED_BASE + 1 + i
        )
        if progress and (i + 1) % 10 == 0:
            progress(f"null importance {i + 1}/{n_runs}")

    # ④ 打分：真实重要性相对自身噪声本底的偏离度
    null_p75 = np.percentile(null_mat, 75, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.log(1e-10 + actual / (1.0 + null_p75))
    score = np.where(np.isfinite(score), score, -np.inf)

    rows = [
        {
            "name": names[j],
            "actual": round(float(actual[j]), 4),
            "null_p75": round(float(null_p75[j]), 4),
            "score": round(float(score[j]), 4),
            "passed": bool(score[j] > float(min_score)),
        }
        for j in range(K)
    ]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def screen_dataset(
    dataset,
    *,
    n_runs: int = NULL_IMPORTANCE_RUNS,
    min_score: float = SCREEN_MIN_SCORE,
    num_boost_round: int = 200,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[str], list[dict]]:
    """对 ``build_stacking_dataset`` 产出做筛选，返回 (白名单 names, 逐因子 rows)。

    筛选窗口 = [panel 最早日期, mining_end)——与 holdout 模式的训练段
    起点一致；验证段/盲测段（mining_end 之后）的行不参与。
    """
    dts = pd.DatetimeIndex(dataset.panel.index.get_level_values("datetime"))
    window_start = dts.min()
    rows = null_importance_scores(
        dataset.feature_matrix,
        dataset.label,
        pd.Series(dts),
        dataset.feature_names,
        window_start=window_start,
        window_end=dataset.mining_end,
        n_runs=n_runs,
        min_score=min_score,
        num_boost_round=num_boost_round,
        progress=progress,
    )
    passed = [r["name"] for r in rows if r["passed"]]
    return passed, rows


def apply_screening(dataset, rows: Sequence[dict]) -> None:
    """把筛选结果原地应用到 dataset：列裁剪 + entries/names 同步 + dropped 追加。

    rows 为 ``null_importance_scores`` 的输出；passed 名单之外的因子列被剔除，
    剔除原因写入 ``dataset.dropped``（reason 带 score 便于审计）。
    """
    keep = {r["name"] for r in rows if r["passed"]}
    names = dataset.feature_names
    rows_by_name = {r["name"]: r for r in rows}
    # 先收集原 library 映射再裁剪（裁剪后 entries 索引错位，不能再用原 i 查）
    lib_by_name = {e.name: e.library for e in dataset.entries}
    keep_idx = [i for i, n in enumerate(names) if n in keep]
    dataset.feature_matrix = dataset.feature_matrix[:, keep_idx]
    dataset.feature_names = [names[i] for i in keep_idx]
    dataset.entries = [dataset.entries[i] for i in keep_idx]
    dataset.dropped = dataset.dropped + [
        {
            "name": n,
            "library": lib_by_name.get(n, ""),
            "reason": f"null_importance_score={rows_by_name[n]['score']}"
                      f" (actual={rows_by_name[n]['actual']}, null_p75={rows_by_name[n]['null_p75']})",
        }
        for n in names
        if n not in keep and n in rows_by_name
    ]
