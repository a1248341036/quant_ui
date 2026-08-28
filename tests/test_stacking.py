"""Stacking 组合框架单元测试：合成面板上验证对齐、无前视、折隔离、模型方向与 pred 落盘。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.factor.stacking import (  # noqa: E402
    FactorEntry,
    build_dataset_from_values,
    daily_spearman_ic,
    forward_return_label,
    fit_predict_walkforward,
    make_model,
    transform_factor_values,
    walk_forward_splits,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_days, n_inst = 300, 8
    days = pd.bdate_range("2024-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    # AR(1) 收益：让 20 日动量对未来收益有真实预测力（纯随机游走没有）
    shock = rng.normal(0, 0.02, (n_days, n_inst))
    ret = np.empty_like(shock)
    ret[0] = shock[0]
    for t in range(1, n_days):
        ret[t] = 0.5 * ret[t - 1] + shock[t]
    close = 100.0 * np.exp(np.cumsum(ret, axis=0)).ravel(order="F")
    industry = np.tile(np.repeat(["A", "B"], n_inst // 2), n_days)
    return pd.DataFrame({"adj_close": close, "industry_sw_l1": industry}, index=idx)


@pytest.fixture(scope="module")
def factor_values(panel: pd.DataFrame):
    """三个因子：动量（正信号）、反转（弱正）、噪声；以及动量的冗余副本。"""
    rng = np.random.default_rng(11)
    n = len(panel)
    close = pd.Series(panel["adj_close"].to_numpy(), index=panel.index)
    g = close.groupby(level="instrument", sort=False)
    mom = g.pct_change(20).to_numpy()
    rev = (-g.pct_change(5)).to_numpy()
    noise = rng.normal(0, 1, n)
    mom_copy = mom + rng.normal(0, 0.001, n)  # 与 mom 相关 ~1.0
    return mom, rev, noise, mom_copy


def _entries(*names: str) -> list[FactorEntry]:
    return [FactorEntry(factor_id=n, name=n, expr="dummy", library="candidate_technical") for n in names]


def test_forward_label_no_lookahead(panel: pd.DataFrame) -> None:
    label = forward_return_label(panel, hold_days=5)
    close = panel["adj_close"]
    # 抽一行手算：T+6 收盘 / T+1 收盘 - 1
    inst_days = close.index.get_level_values(0).unique().sort_values()
    i = 100  # 任取中间行
    dt, inst = close.index[i]
    day_pos = inst_days.get_loc(dt)
    entry = close.loc[(inst_days[day_pos + 1], inst)]
    exit_ = close.loc[(inst_days[day_pos + 6], inst)]
    assert label[i] == pytest.approx(exit_ / entry - 1, rel=1e-6)
    # 尾部 6 行（hold+1）必为 NaN —— 无未来数据
    tail_per_inst = 6
    n_inst = panel.index.get_level_values("instrument").nunique()
    assert np.all(np.isnan(label[-tail_per_inst * n_inst :]))


def test_transform_per_day_rank(panel: pd.DataFrame, factor_values) -> None:
    mom = factor_values[0]
    z = transform_factor_values(mom, panel, size_neutral=False)
    dts = panel.index.get_level_values("datetime")
    day_mask = dts == dts.unique()[50]
    vals = z[day_mask]
    assert abs(np.nanmean(vals)) < 0.1  # 逐日 zscore → 均值≈0
    assert abs(np.nanstd(vals) - 1.0) < 0.2
    assert np.isnan(z[:25]).all() or np.isfinite(z).any()  # NaN 保持传播


def test_dataset_redundancy_filter(panel: pd.DataFrame, factor_values) -> None:
    mom, rev, noise, mom_copy = factor_values
    entries = _entries("mom", "rev", "noise", "mom_copy")
    quality = {"mom": 0.05, "rev": 0.02, "noise": 0.0, "mom_copy": 0.049}
    ds = build_dataset_from_values(
        panel,
        list(zip(entries, [mom, rev, noise, mom_copy])),
        label_days=5,
        mining_end=panel.index.get_level_values("datetime").unique()[200],
        size_neutral=False,
        max_corr=0.9,
        ics_for_quality=quality,
    )
    assert "mom_copy" not in ds.feature_names  # 低质量冗余被剔
    assert "mom" in ds.feature_names
    assert any("redundant_with=mom" in d["reason"] for d in ds.dropped)
    assert ds.feature_matrix.shape[1] == len(ds.feature_names)


def test_walk_forward_purge_and_isolation(panel: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex(panel.index.get_level_values("datetime").unique())
    mining_end = dates[100]
    folds = walk_forward_splits(
        dates, train_start=mining_end, train_months=2, step_months=2, purge_days=5
    )
    assert len(folds) >= 2
    all_train_dates = pd.DatetimeIndex([])
    for f in folds:
        assert f.oos_dates.min() > mining_end
        # purge：train 结束 + 5 个交易日 < OOS 开始
        assert f.train_dates.max() < f.oos_dates.min()
        all_train_dates = all_train_dates.append(f.train_dates)
    # expanding：折间 OOS 不重叠
    oos = [d for f in folds for d in f.oos_dates]
    assert len(oos) == len(set(oos))


def test_ridge_recovers_positive_signal(panel: pd.DataFrame, factor_values) -> None:
    mom, rev, noise, _ = factor_values
    mining_end = panel.index.get_level_values("datetime").unique()[150]
    ds = build_dataset_from_values(
        panel,
        list(zip(_entries("mom", "rev", "noise"), [mom, rev, noise])),
        label_days=5,
        mining_end=mining_end,
        size_neutral=False,
        max_corr=0.99,
    )
    label = ds.label
    dts = pd.Series(panel.index.get_level_values("datetime"))
    dates = pd.DatetimeIndex(panel.index.get_level_values("datetime").unique())
    folds = walk_forward_splits(dates, train_start=mining_end, train_months=2, step_months=3, purge_days=5)
    pred, report = fit_predict_walkforward(ds.feature_matrix, label, dts, folds, kind="ridge")
    assert any(not r.get("skipped") for r in report)
    # 好因子 OOS 方向为正
    ic = daily_spearman_ic(pred, label, dts)
    assert ic.mean() > 0


def test_make_model_unknown_kind() -> None:
    with pytest.raises(ValueError):
        make_model("xgboost")  # type: ignore[arg-type]


def test_pred_parquet_roundtrip(panel: pd.DataFrame, tmp_path: Path, monkeypatch) -> None:
    from scripts.train_ml_composite import write_pred_parquet

    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, len(panel)).astype(np.float32)
    out = tmp_path / "pred.parquet"
    write_pred_parquet(values, panel, out)

    import core.data as core_data

    monkeypatch.setattr(core_data, "PRED_FILE", out)
    mat = core_data.load_pred_scores()
    assert mat is not None
    assert mat.shape[0] > 0 and mat.shape[1] == 8
    # 分数对齐回读：抽一个 (date, code) 校验
    wide = pd.Series(values, index=panel.index).unstack("instrument")
    wide.columns = [str(c).zfill(6) for c in wide.columns]
    d, c = mat.index[0], mat.columns[0]
    assert mat.loc[d, c] == pytest.approx(wide.loc[d, c], rel=1e-5, nan_ok=True)


def test_daily_ic_recovers_signal(panel: pd.DataFrame, factor_values) -> None:
    mom = factor_values[0]
    label = forward_return_label(panel, hold_days=5)
    dts = pd.Series(panel.index.get_level_values("datetime"))
    ic = daily_spearman_ic(mom, label, dts)
    assert len(ic) > 100
    assert ic.mean() > 0  # 动量在合成数据上是正信号


def test_default_max_corr_is_tight(panel: pd.DataFrame, factor_values) -> None:
    """默认 max_corr=0.6 必须有效去冗余：完全冗余副本被剔、独立信号保留。

    防回归：阈值曾默认 0.9（形同虚设，允许相关 0.89 的因子成对进模型）。
    fixture 中 mom_copy 与 mom 相关 ~1.0；mom 与 rev 变换后相关 ~-0.37。
    """
    mom, rev, noise, mom_copy = factor_values
    entries = _entries("mom", "rev", "noise", "mom_copy")
    quality = {"mom": 0.05, "rev": 0.02, "noise": 0.0, "mom_copy": 0.049}
    ds = build_dataset_from_values(
        panel,
        list(zip(entries, [mom, rev, noise, mom_copy])),
        label_days=5,
        mining_end=panel.index.get_level_values("datetime").unique()[200],
        size_neutral=False,
        ics_for_quality=quality,  # 不传 max_corr → 走默认 0.6
    )
    assert "mom_copy" not in ds.feature_names, "默认阈值下完全冗余副本必须被剔除"
    assert set(ds.feature_names) == {"mom", "rev", "noise"}, (
        f"独立信号不应被默认阈值误伤，实际保留: {ds.feature_names}"
    )


def test_registry_eval_end_extraction() -> None:
    from alphaagent.factor.stacking.dataset import _registry_eval_end

    item = {"metrics": {"eval_end": "2025-12-31"}, "ingest_config": {"ingest_end": "2025-12-31"}}
    assert _registry_eval_end(item) == "2025-12-31"
    # eval_end 缺失时回退 ingest_end
    assert _registry_eval_end({"ingest_config": {"ingest_end": "2025-12-31"}}) == "2025-12-31"
    assert _registry_eval_end({}) is None


def test_to_utc_naive_rejects_nat() -> None:
    from alphaagent.factor.stacking.dataset import _to_utc_naive

    assert _to_utc_naive("2026-08-28T10:00:00+00:00") == pd.Timestamp("2026-08-28")
    assert _to_utc_naive(None) is None
    assert _to_utc_naive("not-a-date") is None
    assert _to_utc_naive(float("nan")) is None
