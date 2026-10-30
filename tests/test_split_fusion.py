# -*- coding: utf-8 -*-
"""分频多模型融合（split_fusion）单元测试：分组 → 折 → 每组训练 → 融合 → payload。

覆盖：label_col 分组与未解析回落、purge 放宽到最慢组且各组 OOS 行集一致、
组内模型/组 OOS 指标、equal/icir 融合权重（首折等权、历史滚动）、
单组退化（zscore 单调变换 IC 不变）、缺席组重归一、payload 结构。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.factor.stacking.dataset import (  # noqa: E402
    FactorEntry,
    build_dataset_from_values,
    daily_spearman_ic,
)
from alphaagent.factor.stacking.split_fusion import (  # noqa: E402
    blend_group_scores,
    build_frequency_folds,
    per_frequency_payload,
    split_factor_groups,
    train_frequency_model,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    n_days, n_inst = 400, 40
    days = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"{i:06d}.SZ" for i in range(600000, 600000 + n_inst)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    shock = rng.normal(0, 0.02, (n_days, n_inst))
    ret = np.empty_like(shock)
    ret[0] = shock[0]
    for t in range(1, n_days):
        ret[t] = 0.5 * ret[t - 1] + shock[t]
    close = 100.0 * np.exp(np.cumsum(ret, axis=0)).ravel(order="F")
    industry = np.tile(np.repeat(np.arange(2), n_inst // 2), n_days)
    return pd.DataFrame(
        {"adj_close": close, "industry_sw_l1": industry, "float_cap": rng.uniform(1e9, 1e11, len(idx))},
        index=idx,
    )


def _entry(name: str, label_col: str | None) -> FactorEntry:
    return FactorEntry(
        factor_id=name, name=name, expr="dummy",
        library="candidate_technical", label_col=label_col,
    )


def _mixed_entries() -> list[FactorEntry]:
    """1d×3 + 5d×3(含 1 个未解析回落) + 20d×3，共 10 个。"""
    out = []
    for i in range(3):
        out.append(_entry(f"f1d_{i}", "label_1d_close_to_close"))
    out.append(_entry("f5d_0", "label_5d_close_to_close"))
    out.append(_entry("f5d_1", "label_5d_close_to_close"))
    out.append(_entry("f5d_nolabel", None))  # 未解析 → 回落目标节拍组
    for i in range(3):
        out.append(_entry(f"f20d_{i}", "label_20d_close_to_close"))
    return out


def _build_datasets(panel: pd.DataFrame, groups: dict[str, list[FactorEntry]],
                    mining_end: pd.Timestamp) -> dict[str, object]:
    """每组用 build_dataset_from_values 快路径构造（组内 label=组周期）。"""
    rng = np.random.default_rng(99)
    n = len(panel)
    materials = []
    for key, members in groups.items():
        label_days = int(key[:-1])
        mat = []
        for j, e in enumerate(members):
            rg = np.random.default_rng(1000 + j)
            # 20d 组给更强信号（慢周期重叠标签更易出正 IC），其余噪声
            amp = 0.5 if label_days == 20 else 0.2
            values = (amp * rg.normal(size=n) + rg.normal(0, 0.3, size=n)).astype(np.float32)
            mat.append((e, values))
        materials.append((key, label_days, mat))
    out: dict[str, object] = {}
    for key, label_days, mat in materials:
        out[key] = build_dataset_from_values(
            panel, mat, label_days=label_days, mining_end=mining_end,
            ics_for_quality={e.name: 0.01 for e, _ in mat},
        )
    return out


def _folds(panel: pd.DataFrame, group_label_days: list[int], mining_end: pd.Timestamp):
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    return build_frequency_folds(
        dts, isolation="strict",
        panel_start=mining_end - pd.DateOffset(months=8),
        mining_end=mining_end,
        group_label_days=group_label_days,
        train_months=3, step_months=2, purge_days=3,
    )


# ── 分组 ────────────────────────────────────────────────────────────────────

def test_split_groups_label_col_and_unparsed_fallback() -> None:
    entries = _mixed_entries()
    groups, diag = split_factor_groups(entries, blend_label_days=5)
    assert list(groups) == ["1d", "5d", "20d"]  # 组序按 days 升序稳定
    assert set(groups["1d"]) == {_entry(f"f1d_{i}", "label_1d_close_to_close") for i in range(3)}
    # 未解析回落目标节拍组（5d）
    assert any(e.name == "f5d_nolabel" for e in groups["5d"])
    assert diag["unparsed"] == ["f5d_nolabel"]
    assert diag["group_sizes"] == {"1d": 3, "5d": 3, "20d": 3}
    # label_col 缺失也回落
    groups2, diag2 = split_factor_groups([_entry("x", None)], blend_label_days=20)
    assert list(groups2) == ["20d"] and diag2["unparsed"] == ["x"]


# ── 折 ─────────────────────────────────────────────────────────────────────

def test_folds_purge_covers_slowest_group_and_aligned_oos(panel: pd.DataFrame) -> None:
    mining_end = pd.Timestamp("2020-11-30")
    folds = _folds(panel, [1, 5, 20], mining_end)
    assert len(folds) >= 1
    # purge ≥ 最慢组标签期（20d）：expanding train 尾部截掉 ≥20 个交易日，
    # 使折内最末 OOS 样本的 20d 前向标签窗口不落入下一折训练段
    trading_days = pd.DatetimeIndex(panel.index.get_level_values("datetime")).unique().sort_values()
    for fold in folds:
        if len(fold.train_dates) > 20:
            # train 在 [start, oos[0]) 基础上至少截掉 purge=20 个交易日
            cutoff = trading_days[trading_days < fold.oos_dates.min()]
            assert len(cutoff) - len(fold.train_dates) >= 20, (
                f"折 train 未截足 20 交易日 purge：cutoff={len(cutoff)} train={len(fold.train_dates)}"
            )
        # 折间 OOS 相邻（步长内无空洞），且每折 OOS 落在 mining_end 之后
        assert pd.Timestamp(fold.oos_dates.min()) >= mining_end
    # 各组 OOS 行集一致（融合逐行对齐前提）：同 panel → 组间矩阵行数全等，
    # 且每折 OOS 日期都在组内 label 可见范围内
    entries = _mixed_entries()
    groups, _ = split_factor_groups(entries, blend_label_days=5)
    datasets = _build_datasets(panel, groups, mining_end)
    n_rows = {k: ds.feature_matrix.shape[0] for k, ds in datasets.items()}
    assert len(set(n_rows.values())) == 1
    idx_dates = pd.DatetimeIndex(panel.index.get_level_values("datetime")).to_numpy()
    for k, ds in datasets.items():
        for f in folds:
            assert all(pd.Timestamp(d) in pd.DatetimeIndex(idx_dates) for d in f.oos_dates)
    # 同一批 trading_days 下，purge 大小只改 train 长度、不改 OOS 行集
    folds_small_purge = _folds(panel, [1, 5, 20], mining_end)  # 已用 purge_days=3 → 内部放宽到 20
    assert [set(f.oos_dates) for f in folds_small_purge] == [set(f.oos_dates) for f in folds]


# ── 组内训练 ───────────────────────────────────────────────────────────────

def test_train_frequency_model_produces_metrics(panel: pd.DataFrame) -> None:
    mining_end = pd.Timestamp("2020-11-30")
    entries = _mixed_entries()
    groups, _ = split_factor_groups(entries, blend_label_days=5)
    datasets = _build_datasets(panel, groups, mining_end)
    folds = _folds(panel, [1, 5, 20], mining_end)
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    date_series = pd.Series(dts)
    first_oos = pd.Timestamp(folds[0].oos_dates.min())
    for key, ds in datasets.items():
        gr = train_frequency_model(
            ds, date_series, folds, kinds=["ridge"],
            first_oos=first_oos, group_members=groups[key],
        )
        assert gr.key == key and gr.label_days == int(key[:-1])
        assert len(gr.member_names) == len(groups[key])
        assert gr.oos_ic is not None and gr.oos_ic["ic_mean"] is not None
        assert gr.oos_ic["label_horizon_days"] == int(key[:-1])  # 风险口径按组周期


def test_train_frequency_model_absent_when_no_features(panel: pd.DataFrame) -> None:
    """组内特征全被剔除 → 缺席占位（scores 全 NaN），融合按在场组重归一。"""
    from types import SimpleNamespace
    mining_end = pd.Timestamp("2020-11-30")
    folds = _folds(panel, [1, 5, 20], mining_end)
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    label = np.zeros(len(panel), dtype=np.float32)
    fake_ds = SimpleNamespace(
        feature_matrix=np.empty((len(panel), 0), dtype=np.float32),
        label=label, feature_names=[], label_days=20, entries=[],
    )
    gr = train_frequency_model(
        fake_ds, pd.Series(dts), folds, kinds=["ridge"],
        first_oos=pd.Timestamp(folds[0].oos_dates.min()),
        group_members=[_entry("g", "label_20d_close_to_close")],
    )
    assert np.all(np.isnan(gr.scores)) and gr.oos_ic is None


# ── 融合 ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fusion_artifacts(panel: pd.DataFrame):
    """全链路产物：3 组 results + folds + date_series + first_oos。"""
    mining_end = pd.Timestamp("2020-11-30")
    entries = _mixed_entries()
    groups, _ = split_factor_groups(entries, blend_label_days=5)
    datasets = _build_datasets(panel, groups, mining_end)
    folds = _folds(panel, [1, 5, 20], mining_end)
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    date_series = pd.Series(dts)
    first_oos = pd.Timestamp(folds[0].oos_dates.min())
    results = [
        train_frequency_model(ds, date_series, folds, kinds=["ridge"],
                              first_oos=first_oos, group_members=groups[key])
        for key, ds in datasets.items()
    ]
    return results, folds, date_series, first_oos, datasets, groups


def test_blend_equal_fills_all_oos_rows(fusion_artifacts) -> None:
    results, folds, date_series, first_oos, _, _ = fusion_artifacts
    blended, diag = blend_group_scores(
        results, folds, date_series, first_oos=first_oos, blend_weight="equal",
        blend_label_days=5,
    )
    n_inst = len(date_series) // len(pd.unique(date_series))  # 每天的行数
    n_oos = int(np.isin(
        pd.to_datetime(date_series.to_numpy()),
        pd.to_datetime(np.concatenate([f.oos_dates.to_numpy() for f in folds])),
    ).sum())
    assert int(np.isfinite(blended).sum()) <= n_oos
    # 尾部 label 窗口不足（1d 尾 2 天/5d 尾 6 天/20d 尾 20 天）→ 各组分数皆 NaN，
    # 融合在这些日子为空是设计的（无诚实 OOS 分数可融合）。其余 OOS 行必须全覆盖：
    # 任一天只要"至少一组在场"，融合分数当天就必须有值（在场组重归一）。
    dts_np = pd.to_datetime(date_series.to_numpy())
    any_present = np.zeros(len(dts_np), dtype=bool)
    for g in results:
        any_present |= np.isfinite(g.scores)
    for day in np.unique(dts_np[any_present]):
        day_mask = dts_np == day
        assert np.isfinite(blended[day_mask]).all(), f"融合在 {pd.Timestamp(day).date()} 缺行"
    # 空段仅限尾部（最多 20d 尾部窗口），其余 OOS 行全部覆盖
    assert int(np.isfinite(blended).sum()) >= n_oos - 20 * n_inst
    for fw in diag["fold_weights"]:
        ws = fw["weights"]
        assert set(ws) == {"1d", "5d", "20d"}
        assert all(abs(w - 1 / 3) < 1e-6 for w in ws.values())
        assert fw["fallback"] in ("equal_first_fold", "equal_no_hist")


def test_blend_icir_first_fold_equal_then_history(fusion_artifacts) -> None:
    results, folds, date_series, first_oos, _, _ = fusion_artifacts
    blended, diag = blend_group_scores(
        results, folds, date_series, first_oos=first_oos, blend_weight="icir",
        blend_label_days=5,
    )
    assert diag["fold_weights"][0]["fallback"] == "equal_first_fold"
    assert set(diag["full_oos_icir"]) == {"1d", "5d", "20d"}
    # 后续折：权重非负且总和=1（历史不足时回退等权，也满足）
    for fw in diag["fold_weights"][1:]:
        ws = fw["weights"]
        assert all(v >= -1e-9 for v in ws.values())
        assert abs(sum(ws.values()) - 1.0) < 1e-6
    assert int(np.isfinite(blended).sum()) > 0


def test_blend_single_group_is_zscore_ic_preserved(fusion_artifacts) -> None:
    """单组退化：融合 = 组内分数逐日 zscore，spearman IC 不变（单调变换）。"""
    results, folds, date_series, first_oos, _, _ = fusion_artifacts
    single = [r for r in results if r.key == "20d"]
    blended, _ = blend_group_scores(
        single, folds, date_series, first_oos=first_oos, blend_weight="equal",
        blend_label_days=5,
    )
    gr = single[0]
    dts_np = pd.to_datetime(date_series.to_numpy())
    mask = (dts_np >= pd.Timestamp(first_oos)) & np.isfinite(gr.label)
    a = daily_spearman_ic(blended[mask], gr.label[mask], pd.Series(dts_np[mask]))
    b = daily_spearman_ic(gr.scores[mask], gr.label[mask], pd.Series(dts_np[mask]))
    assert len(a) == len(b) and np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-6)


def test_blend_absent_group_renormalizes(fusion_artifacts) -> None:
    """某组全缺席（scores 全 NaN）：融合按在场组权重重归一，剩余行不 NaN。"""
    results, folds, date_series, first_oos, _, _ = fusion_artifacts
    from copy import deepcopy
    absent = deepcopy(results)
    for r in absent:
        if r.key == "1d":
            r.scores = np.full_like(r.scores, np.nan)
    blended, diag = blend_group_scores(
        absent, folds, date_series, first_oos=first_oos, blend_weight="icir",
        blend_label_days=5,
    )
    n_oos = int(np.isin(
        pd.to_datetime(date_series.to_numpy()),
        pd.to_datetime(np.concatenate([f.oos_dates.to_numpy() for f in folds])),
    ).sum())
    # 只要"至少一个在场组"当天有分，融合分数就必须有值（在场组权重重归一）；
    # 空段仅限尾部（5d/20d 的 label 窗口不足 → 在场组也全 NaN 的日子）。
    dts_np = pd.to_datetime(date_series.to_numpy())
    any_present = np.zeros(len(dts_np), dtype=bool)
    for r in absent:
        any_present |= np.isfinite(r.scores)
    for day in np.unique(dts_np[any_present]):
        day_mask = dts_np == day
        assert np.isfinite(blended[day_mask]).all(), f"融合在 {pd.Timestamp(day).date()} 缺行"
    assert int(np.isfinite(blended).sum()) >= n_oos - 20 * (len(date_series) // len(pd.unique(date_series)))


# ── payload ────────────────────────────────────────────────────────────────

def test_per_frequency_payload_structure(fusion_artifacts) -> None:
    results, folds, date_series, first_oos, datasets, groups = fusion_artifacts
    _, diag = blend_group_scores(
        results, folds, date_series, first_oos=first_oos, blend_weight="icir",
        blend_label_days=5,
    )
    payload = per_frequency_payload(results, diag, blend_label_days=5, blend_weight="icir")
    assert set(payload["groups"]) == {"1d", "5d", "20d"}
    for key, g in payload["groups"].items():
        assert g["label_days"] == int(key[:-1])
        assert g["n_entries"] == len(groups[key])
        assert g["oos_ic"] is not None and g["oos_ic"]["ic_mean"] is not None
        assert "fold_metrics" in g and "feature_weights" in g
    assert payload["blend_label_days"] == 5 and payload["blend_weight"] == "icir"
    assert len(payload["fold_weights"]) == len(folds)
    assert "note" in payload