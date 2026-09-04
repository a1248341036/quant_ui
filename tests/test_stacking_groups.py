"""面感知分组与组合测试：组派生、horizon 派生、组分数、组间权重、退化性质。

覆盖退化契约：N=1 等价单组组合；N=2 双组快慢方案；空组/单成员组合法；
lgbm 放行 NaN 特征而 ridge 保持全有限过滤。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.factor.stacking.dataset import FactorEntry, forward_return_label  # noqa: E402
from alphaagent.factor.stacking.groups import (  # noqa: E402
    UNFACETED_GROUP,
    FacetGroupPolicy,
    apply_weights,
    assign_groups,
    blend_weights,
    daily_zscore,
    derive_blend_horizon,
    derive_group,
    group_horizon,
    group_scores,
    parse_label_days,
)
from alphaagent.factor.stacking.model import fit_predict_walkforward, walk_forward_splits  # noqa: E402


def _entry(name: str, *, expr: str = "TS_MEAN($close,5)", facets: tuple = ()) -> FactorEntry:
    return FactorEntry(factor_id=name, name=name, expr=expr, library="test", facets=tuple(facets))


def _panel_dts(n_days: int = 80, n_inst: int = 10) -> pd.Series:
    days = pd.bdate_range("2024-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    return pd.Series(idx.get_level_values("datetime"))


# ── 组派生 ──────────────────────────────────────────────────────

def test_parse_label_days() -> None:
    assert parse_label_days("label_1d_open_to_open") == 1
    assert parse_label_days("label_10d_close_to_close") == 10
    assert parse_label_days("label_20d_close_to_close") == 20
    assert parse_label_days("label_5d") == 5
    assert parse_label_days("weird_col") is None
    assert parse_label_days(None) is None
    assert parse_label_days("") is None


def test_derive_group_stable_order() -> None:
    assert derive_group({"价量面"}) == "行情组"
    assert derive_group({"量能面", "筹码面"}) == "行情组"
    assert derive_group({"股东面"}) == "基本面组"
    assert derive_group({"基本面"}) == "基本面组"
    assert derive_group({"事件面"}) == "事件资金组"
    assert derive_group({"资金面"}) == "事件资金组"
    # 跨组融合因子：按 FACET_GROUPS 插入序（行情组在前）归主组
    assert derive_group({"价量面", "基本面"}) == "行情组"
    assert derive_group({"资金面", "股东面"}) == "基本面组"
    # 无面归属
    assert derive_group(set()) == UNFACETED_GROUP
    assert derive_group(None) == UNFACETED_GROUP


def test_assign_groups_expr_fallback() -> None:
    members = [
        _entry("a", facets=("价量面",)),           # 显式 facets
        _entry("b", expr="RANK($vwap - TS_MEAN($close,5))"),  # 空 facets → 表达式识别 → 行情组
        _entry("c", expr="funda_gross_margin"),     # → 基本面组
        _entry("d", expr="MYSTERY_OP($x, 3)"),      # 无可识别面 → 未分面
    ]
    groups = assign_groups(members)
    names = {g: [e.name for e in es] for g, es in groups.items()}
    assert names["行情组"] == ["a", "b"]
    assert names["基本面组"] == ["c"]
    assert names[UNFACETED_GROUP] == ["d"]


# ── horizon 派生 ────────────────────────────────────────────────

def test_group_horizon_policy_override_and_mode() -> None:
    policy = FacetGroupPolicy(label_days={"基本面组": 15})
    # 覆盖优先
    assert group_horizon([1, 1, 5], "基本面组", policy) == 15
    # 无覆盖 → 成员众数
    assert group_horizon([1, 1, 5], "行情组", policy) == 1
    assert group_horizon([10, 20, 20, 10], "事件资金组", policy) in (10, 20)
    # 全 None → None
    assert group_horizon([None, None], "行情组", policy) is None


def test_derive_blend_horizon() -> None:
    policy = FacetGroupPolicy(blend_label_days=7)
    assert derive_blend_horizon({"行情组": 1, "基本面组": 20}, policy) == 7
    # 中位数：偶数取更低
    assert derive_blend_horizon({"行情组": 1, "基本面组": 20}, None) == 1
    assert derive_blend_horizon({"a": 1, "b": 5, "c": 20}, None) == 5
    # 全 None → None
    assert derive_blend_horizon({"a": None}, None) is None
    assert derive_blend_horizon({}, None) is None


# ── 组分数 ──────────────────────────────────────────────────────

def test_group_scores_nan_mean_and_absent() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 100).astype(np.float32)
    b = a + rng.normal(0, 1, 100).astype(np.float32)
    nan_heavy = np.full(100, np.nan, dtype=np.float32)
    nan_heavy[:10] = rng.normal(0, 1, 10)

    scores, meta = group_scores(
        {
            "单成员组": [a],
            "双成员组": [a, b],
            "稀疏组": [nan_heavy],
            "全缺组": [np.full(100, np.nan, dtype=np.float32)],
        },
        min_coverage=0.30,
    )
    # 单成员组 = 成员本身（nan 位置一致）
    np.testing.assert_allclose(scores["单成员组"], a, atol=1e-6)
    # 双成员组 = nanmean
    expect = np.nanmean(np.vstack([a, b]), axis=0).astype(np.float32)
    np.testing.assert_allclose(scores["双成员组"], expect, atol=1e-6)
    # 覆盖 10% < 30% → 缺席；全 NaN 成员 → 覆盖 0% 同样缺席
    assert "稀疏组" not in scores and "absent" in meta["稀疏组"]
    assert "全缺组" not in scores and meta["全缺组"]["absent"].startswith("coverage<")
    assert meta["单成员组"]["coverage"] == 1.0


def test_group_scores_fully_nan_member_absent() -> None:
    scores, meta = group_scores(
        {"全缺组": [np.full(50, np.nan, dtype=np.float32)]}, min_coverage=0.30
    )
    assert "全缺组" not in scores
    assert meta["全缺组"]["members"] == 1


# ── 组间权重 ────────────────────────────────────────────────────

def _synthetic_blend_data(n_days: int = 80, n_inst: int = 10, seed: int = 13):
    """g1 = 真信号（与标签同向），g2 = 纯噪声。"""
    rng = np.random.default_rng(seed)
    n = n_days * n_inst
    dts = _panel_dts(n_days, n_inst).to_numpy()
    # 每日截面构造：g1 与未来 1 期截面收益相关
    g1 = np.empty(n, dtype=np.float64)
    label = np.empty(n, dtype=np.float64)
    for t in range(n_days):
        cross = rng.normal(0, 1, n_inst)
        ret = 0.8 * cross + rng.normal(0, 0.5, n_inst)  # 标签由 g1 截面驱动
        g1[t * n_inst:(t + 1) * n_inst] = cross
        label[t * n_inst:(t + 1) * n_inst] = ret
    g2 = rng.normal(0, 1, n)
    mining_end = pd.Timestamp(sorted(set(dts))[n_days // 2])
    return {"行情组": g1.astype(np.float32), "基本面组": g2.astype(np.float32)}, label, dts, mining_end


def test_blend_equal_weights_sum_to_one() -> None:
    scores, label, dts, mining_end = _synthetic_blend_data()
    for method in ("equal", "icir", "ridge_nn"):
        w, diag = blend_weights(scores, label, dts, mining_end=mining_end, method=method)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert set(w) == set(scores)
        assert "fallback" not in diag or method == "equal"


def test_blend_icir_upweights_informative_group() -> None:
    scores, label, dts, mining_end = _synthetic_blend_data()
    w, diag = blend_weights(scores, label, dts, mining_end=mining_end, method="icir")
    assert w["行情组"] > w["基本面组"]
    assert diag["icir"]["行情组"] > diag["icir"]["基本面组"]


def test_blend_ridge_nn_nonnegative_informative_dominant() -> None:
    scores, label, dts, mining_end = _synthetic_blend_data()
    w, diag = blend_weights(scores, label, dts, mining_end=mining_end, method="ridge_nn")
    assert all(v >= 0 for v in w.values())
    assert w["行情组"] >= w["基本面组"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_blend_weights_learn_only_on_mining_window() -> None:
    """mining_end 之后的标签/分数被调换也不影响权重 → 证明学习只用挖掘窗口。"""
    scores, label, dts, mining_end = _synthetic_blend_data()
    scores2 = {g: s.copy() for g, s in scores.items()}
    label2 = label.copy()
    future = pd.Series(dts) > mining_end
    for g in scores2:
        scores2[g][future.to_numpy()] = np.random.default_rng(1).normal(size=int(future.sum()))
    label2[future.to_numpy()] = np.random.default_rng(2).normal(size=int(future.sum()))
    for method in ("icir", "ridge_nn"):
        w1, _ = blend_weights(scores, label, dts, mining_end=mining_end, method=method)
        w2, _ = blend_weights(scores2, label2, dts, mining_end=mining_end, method=method)
        assert w1 == pytest.approx(w2, abs=1e-9)


def test_blend_icir_all_negative_falls_back_to_equal() -> None:
    scores, label, dts, mining_end = _synthetic_blend_data()
    # 两组都取负真信号 → IC 全负 → 回退等权
    anti = {g: (-scores["行情组"]).astype(np.float32) for g in scores}
    w, diag = blend_weights(anti, label, dts, mining_end=mining_end, method="icir")
    assert diag.get("fallback") == "all_icir_nonpositive"
    assert w == pytest.approx({g: 0.5 for g in anti}, abs=1e-9)


def test_blend_degenerate_single_group() -> None:
    scores, label, dts, mining_end = _synthetic_blend_data()
    single = {"行情组": scores["行情组"]}
    for method in ("equal", "icir", "ridge_nn"):
        w, diag = blend_weights(single, label, dts, mining_end=mining_end, method=method)
        assert w == {"行情组": pytest.approx(1.0)}
        assert "fallback" not in diag


def test_blend_unknown_method_raises() -> None:
    scores, label, dts, mining_end = _synthetic_blend_data()
    with pytest.raises(ValueError):
        blend_weights(scores, label, dts, mining_end=mining_end, method="xgboost")


def test_blend_too_few_mining_rows_falls_back() -> None:
    scores, label, dts, _ = _synthetic_blend_data()
    w, diag = blend_weights(scores, label, dts, mining_end=pd.Timestamp("2000-01-01"), method="ridge_nn")
    assert diag.get("fallback") == "mining_rows<100"
    assert w == pytest.approx({g: 0.5 for g in scores}, abs=1e-9)


# ── 合成与退化 ──────────────────────────────────────────────────

def test_apply_weights_renormalizes_missing_group_rows() -> None:
    n_days, n_inst = 30, 6
    dts = _panel_dts(n_days, n_inst).to_numpy()
    rng = np.random.default_rng(9)
    a = rng.normal(0, 1, n_days * n_inst)
    b = a.copy()
    b[n_inst * 10: n_inst * 20] = np.nan  # 中间 10 天基本面组整段缺失
    weights = {"行情组": 0.7, "基本面组": 0.3}
    blended = apply_weights(weights, {"行情组": a, "基本面组": b}, dts)
    za = daily_zscore(a, dts)
    zb = daily_zscore(b, dts)
    # 双组齐全的行（前 10 天）：标准加权和（权重和=1）
    both = slice(0, n_inst * 10)
    expect_both = 0.7 * za[both] + 0.3 * zb[both]
    np.testing.assert_allclose(blended[both], expect_both, atol=1e-6)
    # 缺失组的行（中间 10 天）：按可用组重归一 → 纯行情组 zscore
    only_a = slice(n_inst * 10, n_inst * 20)
    np.testing.assert_allclose(blended[only_a], za[only_a], atol=1e-6)
    assert np.isfinite(blended[only_a]).all()


def test_apply_weights_all_groups_missing_row_is_nan() -> None:
    dts = _panel_dts(5, 4).to_numpy()
    a = np.full(20, np.nan, dtype=np.float32)
    blended = apply_weights({"行情组": 1.0}, {"行情组": a}, dts)
    assert np.isnan(blended).all()


def test_daily_zscore_cross_sectional() -> None:
    n_days, n_inst = 20, 8
    dts = _panel_dts(n_days, n_inst).to_numpy()
    rng = np.random.default_rng(4)
    vals = rng.normal(0, 10, n_days * n_inst).astype(np.float32)
    z = daily_zscore(vals, dts)
    day_scores = z[:n_inst]
    assert abs(np.nanmean(day_scores)) < 1e-6  # rank pct 均值 0
    # transform('std') 为 ddof=1 样本标准差 → z 的方差剩 (n-1)/n（与
    # transform_factor_values 同语义）
    assert abs(np.nanstd(day_scores) - np.sqrt((n_inst - 1) / n_inst)) < 1e-5


# ── model.py：finite 过滤按 kind 分 ────────────────────────────

def _nan_feature_panel(seed: int = 21):
    rng = np.random.default_rng(seed)
    n_days, n_inst = 130, 10
    dts_full = _panel_dts(n_days, n_inst)
    n = n_days * n_inst
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    nan_mask = rng.random(n) < 0.25
    x1[nan_mask] = np.nan  # 25% 行缺失第二特征（模拟基本面稀疏面）
    feats = np.column_stack([x0, x1]).astype(np.float32)
    label = (0.5 * x0 + rng.normal(0, 0.5, n)).astype(np.float32)
    label[-n_inst:] = np.nan  # 尾部标签缺失（无未来数据）
    return feats, label, dts_full, nan_mask


def _simple_folds(dts_full: pd.Series):
    days = pd.DatetimeIndex(np.unique(dts_full.to_numpy()))
    return walk_forward_splits(
        days, train_start=days[0], train_months=2, step_months=2, purge_days=3
    )


def test_ridge_keeps_all_finite_feature_filter() -> None:
    feats, label, dts_full, nan_mask = _nan_feature_panel()
    folds = _simple_folds(dts_full)
    pred, report = fit_predict_walkforward(feats, label, dts_full, folds, kind="ridge")
    assert any(not r.get("skipped") for r in report)
    oos_mask = np.isin(pd.to_datetime(dts_full).to_numpy(), folds[0].oos_dates.to_numpy())
    # ridge：OOS 中 NaN 特征行不产预测
    assert np.all(np.isnan(pred[oos_mask & nan_mask & np.isfinite(label)]))


def test_lgbm_tolerates_nan_features() -> None:
    feats, label, dts_full, nan_mask = _nan_feature_panel()
    folds = _simple_folds(dts_full)
    pred, report = fit_predict_walkforward(feats, label, dts_full, folds, kind="lgbm")
    assert any(not r.get("skipped") for r in report)
    oos_mask = np.isin(pd.to_datetime(dts_full).to_numpy(), folds[0].oos_dates.to_numpy())
    target = oos_mask & nan_mask & np.isfinite(label)
    assert target.sum() > 0
    # lgbm：NaN 特征行照样有预测（放行稀疏面样本行）
    assert np.isfinite(pred[target]).all()


# ── 数据集侧：FactorEntry 新字段 ────────────────────────────────

def test_factor_entry_new_fields_default() -> None:
    e = FactorEntry(factor_id="x", name="x", expr="$close", library="t")
    assert e.facets == ()
    assert e.label_col is None
    e2 = FactorEntry(
        factor_id="x", name="x", expr="$close", library="t",
        facets=("价量面", "基本面"), label_col="label_20d_close_to_close",
    )
    assert derive_group(e2.facets) == "行情组"
    assert parse_label_days(e2.label_col) == 20


def test_forward_label_still_works_after_import_change() -> None:
    """冒烟：dataset 改动后既有标签函数行为不变。"""
    days = pd.bdate_range("2024-01-01", periods=10)
    idx = pd.MultiIndex.from_product([days, ["A"]], names=["datetime", "instrument"])
    panel = pd.DataFrame({"adj_close": np.arange(1.0, 11.0)}, index=idx)
    label = forward_return_label(panel, hold_days=2)
    assert label[0] == pytest.approx(4.0 / 2.0 - 1.0)  # T+3 收盘 / T+1 收盘 - 1
    assert np.isnan(label[-3:]).all()
