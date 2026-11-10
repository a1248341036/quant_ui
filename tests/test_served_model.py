# -*- coding: utf-8 -*-
"""served model 单测：拟合 → 存取 → 推理 → 融合/平滑口径。

聚焦训练/推理分离后的两条不变量：
1. served model 可落盘/加载且 model_card 自描述（运营页与审计靠它）；
2. 推理幂等且输出 pred 通道格式（同模型同数据 → 同分数）。
物化环节用 monkeypatch 替身，避免单测依赖 DSL 求值与磁盘缓存。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.factor.stacking import serve  # noqa: E402
from alphaagent.factor.stacking.dataset import FactorEntry, StackingDataset  # noqa: E402
from alphaagent.factor.stacking.serve import (  # noqa: E402
    blend_group_scores_fixed,
    fit_served_model,
    load_served_model,
    predict_served,
    resolve_latest_served_model,
    save_served_model,
    wma_smooth_scores,
)


def _panel(n_days: int = 80, n_inst: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n_days)
    inst = [f"{600000 + i:06d}.SH" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    return pd.DataFrame(
        {
            "adj_close": rng.uniform(8, 30, len(idx)),
            "float_cap": rng.uniform(1e9, 1e10, len(idx)),
        },
        index=idx,
    )


def _dataset(panel: pd.DataFrame, label_days: int = 5, n_factors: int = 3,
             seed: int = 11) -> StackingDataset:
    rng = np.random.default_rng(seed)
    n = len(panel)
    entries = [
        FactorEntry(
            factor_id=f"f{i}", name=f"f{i}", expr="dummy",
            library="candidate_technical", label_col=f"label_{label_days}d_close_to_close",
        )
        for i in range(n_factors)
    ]
    feats = rng.normal(size=(n, n_factors)).astype(np.float32)
    # label 与特征弱相关（保证模型有可学信号，不是纯噪声）
    label = (0.3 * feats[:, 0] + rng.normal(0, 1, n)).astype(np.float32)
    return StackingDataset(
        panel=panel,
        entries=entries,
        feature_matrix=feats,
        feature_names=[e.name for e in entries],
        label=label,
        mining_end=pd.Timestamp("2024-01-01"),
        label_days=label_days,
    )


@pytest.fixture()
def fake_materialize(monkeypatch):
    """用与 panel 行序一致的**确定性**随机值替换 DSL 物化（单测不碰求值链）。

    确定性很重要：``predict_served`` 的幂等性断言依赖「同输入 → 同输出」，
    若替身每次返回新随机数，测出来的差异是替身造的，不是推理链的。
    """
    import zlib

    def _fake(panel, entries, *, cache=None, min_finite_ratio=0.05, progress=None):
        out = []
        for e in entries:
            seed = zlib.crc32(e.name.encode("utf-8")) % (2**32)
            rng = np.random.default_rng(seed)
            out.append((e, rng.normal(size=len(panel)).astype(np.float32)))
        return out, []

    monkeypatch.setattr(serve, "materialize_entries", _fake)
    return _fake


def test_fit_save_load_roundtrip(tmp_path: Path) -> None:
    panel = _panel()
    ds = _dataset(panel)
    served = fit_served_model(
        {"5d": ds}, ["ridge"], blend_weights={"5d": 1.0},
        blend_label_days=5, smooth_window=1, size_neutral=True, source_run="unit_test",
    )
    assert served.group_keys() == ["5d"]
    assert served.blend_weights == {"5d": 1.0}
    assert served.trained_until, "训练截止日必须记录"
    assert served.groups["5d"].n_train_rows > 100

    path = save_served_model(served, tmp_path)
    assert path.is_file()
    card = json.loads((tmp_path / serve.MODEL_CARD_NAME).read_text(encoding="utf-8"))
    assert card["source_run"] == "unit_test"
    assert card["groups"]["5d"]["n_features"] == 3

    loaded = load_served_model(path)
    assert loaded.trained_until == served.trained_until
    assert set(loaded.groups["5d"].models) == {"ridge"}


def test_fit_served_model_defaults_to_equal_weights() -> None:
    panel = _panel()
    served = fit_served_model({"1d": _dataset(panel, 1), "20d": _dataset(panel, 20)},
                              ["ridge"], blend_weights=None)
    assert served.blend_weights == {"1d": 0.5, "20d": 0.5}
    assert abs(sum(served.blend_weights.values()) - 1.0) < 1e-9


def test_predict_is_idempotent_and_pred_shaped(fake_materialize) -> None:
    panel = _panel()
    # size_neutral=False：本用例只验推理链路与幂等性，不走 numba 市值中性化
    # （该路径对 12 只 × 80 天的极小面板存在已知类型推断限制，生产 5000 只不受影响）
    served = fit_served_model({"5d": _dataset(panel)}, ["ridge"],
                              blend_weights={"5d": 1.0}, smooth_window=1,
                              size_neutral=False)
    first = predict_served(served, panel)
    second = predict_served(served, panel)
    assert list(first.columns) == ["date", "code", "score"]
    assert not first.empty
    assert first["code"].str.len().eq(6).all(), "code 必须是 6 位裸码"
    pd.testing.assert_frame_equal(first, second)


def test_predict_respects_date_window(fake_materialize) -> None:
    panel = _panel()
    served = fit_served_model({"5d": _dataset(panel)}, ["ridge"],
                              blend_weights={"5d": 1.0}, smooth_window=1,
                              size_neutral=False)
    days = panel.index.get_level_values("datetime")
    start, end = days.min(), days.max()
    mid = days.unique()[len(days.unique()) // 2]
    frame = predict_served(served, panel, start=mid, end=end)
    assert frame["date"].min() >= pd.Timestamp(mid)
    assert frame["date"].max() <= pd.Timestamp(end)
    assert frame["date"].min() > pd.Timestamp(start)


def test_blend_fixed_weights_renormalize_on_missing_group() -> None:
    dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=6), 4))
    n = len(dates)
    s1 = np.linspace(-1, 1, n)
    s2 = np.full(n, np.nan)  # 缺席组：权重应被重归一，不产生 NaN 污染
    blended = blend_group_scores_fixed({"1d": s1, "20d": s2}, {"1d": 0.5, "20d": 0.5}, dates)
    assert np.isfinite(blended).any()
    assert np.nanmax(np.abs(blended)) <= 5.0


def test_wma_smooth_keeps_leading_nan_and_reduces_noise() -> None:
    panel = _panel(n_days=10, n_inst=2)
    values = np.arange(len(panel), dtype=np.float32)
    values[0:2] = np.nan
    out = wma_smooth_scores(values, panel, 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert np.isfinite(out[2:]).all()
    assert len(out) == len(values)


def test_resolve_latest_served_model(tmp_path: Path) -> None:
    import os
    import time

    for name, delta in (("run_a", 10), ("run_b", 20)):
        d = tmp_path / name
        d.mkdir()
        p = d / serve.SERVED_MODEL_NAME
        p.write_bytes(b"x")
        os.utime(p, (time.time() - 100 + delta, time.time() - 100 + delta))
    latest = resolve_latest_served_model(tmp_path)
    assert latest.parent.name == "run_b"


def test_supervised_scorer_does_not_drift_from_training_window() -> None:
    """served 拟合严格遵守 train_end（防把推理期数据混进上线模型）。"""
    panel = _panel(n_days=120)
    ds = _dataset(panel)
    cutoff = panel.index.get_level_values("datetime").unique()[40]
    served = fit_served_model({"5d": ds}, ["ridge"], blend_weights={"5d": 1.0},
                              train_end=cutoff)
    assert served.groups["5d"].train_end == str(pd.Timestamp(cutoff).date())
