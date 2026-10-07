# -*- coding: utf-8 -*-
"""慢档门槛标定（`technical_weekly` / `technical_monthly`，2026-10-07）。

**为什么存在这个标定**：`min_abs_ic` / `min_icir` / `min_val_abs_ic` 按 **label_1d 尺度**
标定；同一批因子换到 label_5d / label_20d 后 |IC|、|ICIR| 会系统性变大。若慢档沿用 1d 线，
门槛会近乎失效（选择性失衡）——实测配对样本（候选池 42 个因子 × 3 个 label，train
2020-2022，n=41 有效配对）中位比值：

    label_5d/1d  = |IC| 1.776 / |ICIR| 1.800
    label_20d/1d = |IC| 2.865 / |ICIR| 3.032

本测试锁定三件不变量：
1. 慢档 IC/ICIR 门槛 == 主档值 × 上述实测比值（数值可复算，防止后人手改漂移）；
2. 与 label 尺度无关的项**不动**（val 保留比、coverage/autocorr/corr、engine_gate 年化门槛）；
3. label↔freq 一致性校验不被破坏（复用 `ensure_label_freq_consistency`）。
"""
from __future__ import annotations

import pytest

from alphaagent.factor.mining.research_spec import (
    default_research_spec,
    effective_research_spec,
    ensure_label_freq_consistency,
)

# 2026-10-07 配对实测中位比值（n=41）—— 标定算法的输入，改标定须先更新实测
R_IC_5D, R_ICIR_5D = 1.776, 1.800
R_IC_20D, R_ICIR_20D = 2.865, 3.032

# 主档（label_1d）标定线 = 标定基准
BASE = {
    ("evaluation_policy", "min_train_abs_ic"): 0.02,
    ("evaluation_policy", "min_train_icir"): 0.28,
    ("evaluation_policy", "min_val_abs_ic"): 0.015,
    ("delivery_policy.candidate", "min_abs_ic"): 0.02,
    ("delivery_policy.candidate", "min_icir"): 0.28,
    ("delivery_policy.candidate", "min_val_abs_ic"): 0.015,
    ("delivery_policy.production", "min_train_abs_ic"): 0.025,
    ("delivery_policy.production", "min_train_icir"): 0.30,
    ("delivery_policy.production", "min_val_abs_ic"): 0.015,
}
# 与 label 尺度无关 → 三个 technical 档必须相同
SCALE_FREE = [
    ("delivery_policy.candidate", "min_val_ic_retention"),
    ("delivery_policy.production", "min_val_ic_retention"),
    ("delivery_policy.candidate", "min_coverage"),
    ("delivery_policy.candidate", "min_cs_autocorr"),
    ("delivery_policy.candidate", "max_abs_corr"),
    ("delivery_policy.candidate", "min_val_long_excess"),
]
TIERS = ["technical", "technical_weekly", "technical_monthly"]


def _get(spec: dict, path: str, key: str):
    d = spec
    for part in path.split("."):
        d = (d or {}).get(part) if isinstance(d, dict) else None
    return (d or {}).get(key) if isinstance(d, dict) else None


@pytest.mark.parametrize("mode, r_ic, r_icir", [
    ("technical_weekly", R_IC_5D, R_ICIR_5D),
    ("technical_monthly", R_IC_20D, R_ICIR_20D),
])
def test_slow_tier_thresholds_are_scaled_from_base(mode: str, r_ic: float, r_icir: float) -> None:
    """慢档 IC/ICIR 门槛 == 主档基准 × 实测比值（四舍五入到 4 位小数）。"""
    spec = effective_research_spec(mode)
    for (path, key), base in BASE.items():
        got = _get(spec, path, key)
        assert got is not None, f"{mode} 缺少 {path}.{key}"
        ratio = r_icir if key.endswith("icir") else r_ic
        assert got == pytest.approx(round(base * ratio, 4), abs=5e-5), (
            f"{mode}.{path}.{key} = {got}，期望 {base} × {ratio} = {round(base * ratio, 4)}"
        )


def test_tier_thresholds_increase_with_holding_period() -> None:
    """不快于 label 的档，门槛必须更严：1d < 5d < 20d（对 IC/ICIR 类）。"""
    specs = {m: effective_research_spec(m) for m in TIERS}
    for path, key in BASE:
        vals = [_get(specs[m], path, key) for m in TIERS]
        assert vals[0] < vals[1] < vals[2], f"{path}.{key} 在三档非严格递增: {vals}"


def test_scale_free_thresholds_untouched() -> None:
    """与 label 尺度无关的项在三个 technical 档间必须完全一致（本次标定刻意不动）。"""
    specs = {m: effective_research_spec(m) for m in TIERS}
    for path, key in SCALE_FREE:
        vals = [_get(specs[m], path, key) for m in TIERS]
        assert len(set(vals)) == 1, f"{path}.{key} 被误改: {vals}"


def test_engine_gate_numeric_thresholds_inherit_main_tier() -> None:
    """engine_gate 年化门槛无该档实测数据 → 暂继承主档（0.03 / 0.5），仅 freq/白名单不同。"""
    tech = ((effective_research_spec("technical").get("delivery_policy") or {})
            .get("production") or {}).get("engine_gate") or {}
    for mode in ("technical_weekly", "technical_monthly"):
        eg = ((effective_research_spec(mode).get("delivery_policy") or {})
              .get("production") or {}).get("engine_gate") or {}
        assert eg.get("min_excess_annual") == tech.get("min_excess_annual")
        assert eg.get("min_excess_sharpe") == tech.get("min_excess_sharpe")


@pytest.mark.parametrize("mode", TIERS + ["fundamental", "report", "technical_daily"])
def test_label_freq_consistency_still_holds(mode: str) -> None:
    """标定不得破坏 2026-10-04 的 label↔freq 强制一致（构建期 fail-closed）。"""
    ensure_label_freq_consistency(effective_research_spec(mode))
    ensure_label_freq_consistency(default_research_spec(mode))
