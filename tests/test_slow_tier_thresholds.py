# -*- coding: utf-8 -*-
"""慢档门槛标定（`technical_weekly` / `technical_monthly`，2026-10-07 v2：文献锚 + 选择性对齐）。

**为什么存在这个标定**：`min_abs_ic` / `min_icir` / `min_val_abs_ic` 按 **label_1d 尺度**
标定；同一批因子换到 label_5d / label_20d 后 |IC|、|ICIR| 会系统性变大，若慢档沿用 1d 线，
门槛近乎失效（选择性失衡）。但**只按等比放大又会把绝对值推到文献罕见区间**——故 v2 用双锚：

* 锚①（绝对量级，外部基准）：日频 Rank-IC ≥0.02 有筛选价值、0.03~0.05 可用；月度 IC
  0.02~0.06 正常有效、0.05~0.10 属"很好"；ICIR 有效线 0.3（0.5 算强）；文献实测月度因子
  PE TTM 0.0529/0.6995、PB 0.0557/0.4888。
* 锚②（相对选择性）：慢档在同一因子池上的过线率与主档持平。
  实测配对池（候选池 42 因子 × label_1d/5d/20d，train 2020-2022，n=41）：
  主档 0.02/0.28 → 36.6%；weekly 0.030/0.450 → **36.6%**；monthly 0.053/0.650 → **36.6%**。

本测试锁定：
1. 慢档 candidate/evaluation 的 IC/ICIR 线 == 上述文献锚值（可复算：主档基准 × 锚定比值）；
2. 三档门槛严格递增（1d < 5d < 20d）；
3. production 按主档"正式库更严"倍数派生（IC ×1.25、ICIR ×0.30/0.28）；min_val_abs_ic 按
   本档 IC 线/主档 0.02 的倍数放大；
4. 与 label 尺度无关的项在三档间完全一致（本次标定刻意不动）；
5. label↔freq 一致性校验不被破坏。
"""
from __future__ import annotations

import pytest

from alphaagent.factor.mining.research_spec import (
    default_research_spec,
    effective_research_spec,
    ensure_label_freq_consistency,
)

# 主档（label_1d）标定线 = 对标基准
BASE_IC, BASE_ICIR, BASE_VAL_IC = 0.02, 0.28, 0.015
BASE_PROD_IC, BASE_PROD_ICIR = 0.025, 0.30

# 慢档锚定线（文献锚 + 选择性对齐的选定值）
TIER_BAR = {
    "technical_weekly": (0.030, 0.450),
    "technical_monthly": (0.053, 0.650),
}
# 由锚定线派生的比值（主档 → 本档）
RATIO = {m: (ic / BASE_IC, icir / BASE_ICIR) for m, (ic, icir) in TIER_BAR.items()}

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


@pytest.mark.parametrize("mode", list(TIER_BAR))
def test_slow_tier_bar_matches_documented_anchor(mode: str) -> None:
    """candidate / evaluation 的 IC、ICIR 线 == 文献锚定值（= 主档基准 × 锚定比值）。"""
    spec = effective_research_spec(mode)
    ic_bar, icir_bar = TIER_BAR[mode]
    r_ic, r_icir = RATIO[mode]
    assert r_ic == pytest.approx(ic_bar / BASE_IC)
    assert r_icir == pytest.approx(icir_bar / BASE_ICIR)
    for path in ("evaluation_policy", "delivery_policy.candidate"):
        assert _get(spec, path, "min_train_abs_ic" if path == "evaluation_policy" else "min_abs_ic") \
            == pytest.approx(ic_bar, abs=1e-4)
        assert _get(spec, path, "min_train_icir" if path == "evaluation_policy" else "min_icir") \
            == pytest.approx(icir_bar, abs=1e-4)
        # min_val_abs_ic 按本档 IC 线/主档 0.02 的倍数放大
        assert _get(spec, path, "min_val_abs_ic") == pytest.approx(BASE_VAL_IC * r_ic, abs=1e-4)


@pytest.mark.parametrize("mode", list(TIER_BAR))
def test_production_derives_from_candidate_bar(mode: str) -> None:
    """production 门槛 == 本档候选线 × 主档"正式库更严"倍数（IC ×1.25、ICIR ×0.30/0.28）。"""
    spec = effective_research_spec(mode)
    ic_bar, icir_bar = TIER_BAR[mode]
    assert _get(spec, "delivery_policy.production", "min_train_abs_ic") \
        == pytest.approx(ic_bar * (BASE_PROD_IC / BASE_IC), abs=1e-4)
    assert _get(spec, "delivery_policy.production", "min_train_icir") \
        == pytest.approx(icir_bar * (BASE_PROD_ICIR / BASE_ICIR), abs=1e-4)


def test_bar_stays_within_literature_range() -> None:
    """慢档线必须落在文献可解释区间：ic ≤ 0.10（>0.10 属罕见/查泄漏），icir ≤ 1.0。"""
    for mode, (ic_bar, icir_bar) in TIER_BAR.items():
        assert ic_bar <= 0.10, f"{mode} 的 IC 线 {ic_bar} 超出文献'很好'上限"
        assert icir_bar <= 1.0, f"{mode} 的 ICIR 线 {icir_bar} 过高（文献强线约 0.5~0.7）"


def test_tier_thresholds_increase_with_holding_period() -> None:
    """不快于 label 的档，门槛必须更严：1d < 5d < 20d（对 IC/ICIR 类）。"""
    specs = {m: effective_research_spec(m) for m in TIERS}
    keys = [
        ("evaluation_policy", "min_train_abs_ic"),
        ("evaluation_policy", "min_train_icir"),
        ("evaluation_policy", "min_val_abs_ic"),
        ("delivery_policy.candidate", "min_abs_ic"),
        ("delivery_policy.candidate", "min_icir"),
        ("delivery_policy.production", "min_train_abs_ic"),
        ("delivery_policy.production", "min_train_icir"),
    ]
    for path, key in keys:
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
    for mode in TIER_BAR:
        eg = ((effective_research_spec(mode).get("delivery_policy") or {})
              .get("production") or {}).get("engine_gate") or {}
        assert eg.get("min_excess_annual") == tech.get("min_excess_annual")
        assert eg.get("min_excess_sharpe") == tech.get("min_excess_sharpe")


@pytest.mark.parametrize("mode", TIERS + ["fundamental", "report", "technical_daily"])
def test_label_freq_consistency_still_holds(mode: str) -> None:
    """标定不得破坏 2026-10-04 的 label↔freq 强制一致（构建期 fail-closed）。"""
    ensure_label_freq_consistency(effective_research_spec(mode))
    ensure_label_freq_consistency(default_research_spec(mode))
