# -*- coding: utf-8 -*-
"""行业轮动诊断 metrics 与 profile 挂载测试（P1）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaagent.factor.evaluation.context import EvaluationContext
from alphaagent.factor.evaluation.plugins import (
    industry_exposure_r2,
    industry_ic_dispersion,
    industry_within_rank_ic,
)
from alphaagent.factor.evaluation.profile import EvaluationProfile, resolve_profiles


def _synthetic_panel(n_days: int = 40, n_a: int = 20, n_b: int = 20, seed: int = 42) -> pd.DataFrame:
    """两行业合成面板：IND_A 有行业内 alpha，且行业均值偏移相反。

    MultiIndex = from_product([dates, instruments])，日块内行序为
    IND_A 全部再 IND_B 全部；数组按列优先拼回同一行序。
    label = 行业偏移（A +0.8 / B -0.8）+ 行业内 alpha（仅 A 与信号相关）。
    """
    rng = np.random.default_rng(seed)
    instruments = [f"S{i:03d}" for i in range(n_a + n_b)]
    industries = ["IND_A"] * n_a + ["IND_B"] * n_b
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    panel = pd.DataFrame(index=index)
    panel["industry_sw_l1"] = np.tile(np.array(industries), n_days)

    # 每日期块按 [A(20), B(20)] 展开，与面板行序对齐
    a = np.tile(np.linspace(-1, 1, n_a), n_days).reshape(n_days, n_a)
    b = np.tile(np.linspace(-1, 1, n_b), n_days).reshape(n_days, n_b)
    panel["x"] = np.column_stack([a, b]).ravel()
    y_a = 0.8 + 1.0 * a + rng.normal(0, 0.1, (n_days, n_a))
    y_b = -0.8 + rng.normal(0, 0.1, (n_days, n_b))
    panel["label"] = np.column_stack([y_a, y_b]).ravel()
    panel["f_industry_dummy"] = (panel["industry_sw_l1"] == "IND_A").astype(np.float64)
    return panel


def _ctx(panel: pd.DataFrame, factor_col: str, label_col: str = "label") -> EvaluationContext:
    return EvaluationContext(
        panel=panel[[field for field in ("industry_sw_l1", label_col) if field in panel.columns]],
        factor=pd.Series(
            panel[factor_col].to_numpy(dtype=np.float64),
            index=panel.index, name=factor_col,
        ),
        label=pd.Series(
            panel[label_col].to_numpy(dtype=np.float64),
            index=panel.index, name=label_col,
        ),
        profile=EvaluationProfile("test", "val"),
        factor_name=factor_col,
    )


def _params() -> dict:
    return {"field": "industry_sw_l1", "min_cover": 15, "min_days": 20}


def test_within_rank_ic_detects_within_industry_alpha() -> None:
    panel = _synthetic_panel()
    out = industry_within_rank_ic(_ctx(panel, "x"), _params())
    # IND_A 有干净行业内 alpha、IND_B 无 → 行业内加权 IC 显著为正且高于全市场 IC
    # （全市场 IC 被两行业相反偏移的对冲稀释）
    assert out["within_rank_ic"] > 0.3
    assert out["market_rank_ic"] > 0.2
    assert out["within_minus_market"] > 0.1
    assert out["n_days"] == 40
    assert out["n_industries_used"] == 2


def test_exposure_r2_high_for_industry_dummy() -> None:
    panel = _synthetic_panel()
    out = industry_exposure_r2(_ctx(panel, "f_industry_dummy"), _params())
    # 纯行业哑变量：逐日因子值完全由行业组均值解释
    assert out["exposure_r2"] > 0.99
    assert out["n_days"] == 40


def test_exposure_r2_low_for_within_industry_alpha() -> None:
    panel = _synthetic_panel()
    out = industry_exposure_r2(_ctx(panel, "x"), _params())
    assert out["exposure_r2"] < 0.05


def test_industry_dummy_shows_no_within_alpha_and_full_exposure() -> None:
    panel = _synthetic_panel()
    params = _params()
    within = industry_within_rank_ic(_ctx(panel, "f_industry_dummy"), params)
    # 纯行业哑变量在行业内恒等 → 行业内 Rank IC 无定义（NaN），暴露 R²≈1：
    # 正是「全市场高 IC ≠ 行业内 alpha」的诊断证据
    assert np.isnan(within["within_rank_ic"])
    assert within["n_industries_used"] == 0
    expo = industry_exposure_r2(_ctx(panel, "f_industry_dummy"), params)
    assert expo["exposure_r2"] > 0.99


def test_industry_ic_dispersion_reports_sector_spread() -> None:
    panel = _synthetic_panel()
    out = industry_ic_dispersion(_ctx(panel, "x"), _params())
    assert out["n_industries"] == 2
    assert np.isfinite(out["dispersion_std"])
    assert out["dispersion_std"] > 0.0
    assert set(out["top3"]) == {"IND_A", "IND_B"}
    assert set(out["bottom3"]) == {"IND_A", "IND_B"}


def test_missing_industry_column_returns_nan_and_note() -> None:
    panel = _synthetic_panel().drop(columns=["industry_sw_l1"])
    params = dict(_params())
    within = industry_within_rank_ic(_ctx(panel, "x"), params)
    assert np.isnan(within["within_rank_ic"])
    assert "缺少行业字段" in within["note"]
    expo = industry_exposure_r2(_ctx(panel, "x"), params)
    assert np.isnan(expo["exposure_r2"])
    assert "缺少行业字段" in expo["note"]
    disp = industry_ic_dispersion(_ctx(panel, "x"), params)
    assert np.isnan(disp["dispersion_std"])
    assert "缺少行业字段" in disp["note"]


class TestProfileMount:
    def test_disabled_rotation_does_not_mount(self) -> None:
        profiles = resolve_profiles({"rotation_policy": {"enabled": False}})
        for pid in ("train_screen", "validation", "production_delivery"):
            plugins = [m["plugin"] for m in profiles[pid].metrics]
            assert not any("industry" in name for name in plugins)

    def test_enabled_mounts_three_metrics(self) -> None:
        profiles = resolve_profiles({"rotation_policy": {"enabled": True, "report_industry_metrics": True}})
        expected = ["industry_within_rank_ic", "industry_exposure_r2", "industry_ic_dispersion"]
        for pid in ("train_screen", "validation", "production_delivery"):
            plugins = [m["plugin"] for m in profiles[pid].metrics]
            assert plugins[-3:] == expected
        # 吞吐敏感 lite 不挂
        lite_plugins = [m["plugin"] for m in profiles["train_screen_lite"].metrics]
        assert not any("industry" in name for name in lite_plugins)

    def test_user_metric_override_is_appended_not_replaced(self) -> None:
        profiles = resolve_profiles({
            "rotation_policy": {"enabled": True, "report_industry_metrics": True},
            "evaluation_profiles": {
                "validation": {"metrics": [{"plugin": "cross_sectional_core"}]},
            },
        })
        plugins = [m["plugin"] for m in profiles["validation"].metrics]
        assert plugins[0] == "cross_sectional_core"
        assert plugins[-1] == "industry_ic_dispersion"
