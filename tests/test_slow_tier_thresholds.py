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
from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria

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


# ── fundamental 档（label_20d + monthly）────────────────────────────────
# 2026-10-08 门槛统一：门槛只由 label 持有期唯一决定、与数据面无关。fundamental 与
# technical_monthly 同为 label_20d，**共用同一套 20d 门槛**（_MONTHLY_20D_OVERRIDES，
# 见 core/research_modes.py；数值 = technical_monthly 的双锚标定 0.053/0.65/0.0398）。
# 取代此前 fundamental 独立标定的 0.035/0.45/0.021（"文献月度基本面量级"）——那造成
# 同为 label_20d 却门槛不同（技术月频 0.053/0.65 vs 基本面松 0.035/0.45）的矛盾。
MONTHLY_BAR = (0.053, 0.650)
MONTHLY_VAL_IC = 0.0398
FUNDA_LEVELS = [
    ("evaluation_policy", "min_train_abs_ic", "min_train_icir"),
    ("delivery_policy.candidate", "min_abs_ic", "min_icir"),
    ("delivery_policy.production", "min_train_abs_ic", "min_train_icir"),
]


def test_fundamental_inherits_monthly_20d_scale() -> None:
    """fundamental 与 technical_monthly 同为 label_20d：三层门槛完全一致（同源 _MONTHLY_20D_OVERRIDES）。"""
    fund = effective_research_spec("fundamental")
    m20 = effective_research_spec("technical_monthly")
    for path, ic_key, icir_key in FUNDA_LEVELS:
        # 与 technical_monthly 逐字段一致（唯一真源：shared 20d 常量）
        assert _get(fund, path, ic_key) == _get(m20, path, ic_key), f"{path}.{ic_key}"
        assert _get(fund, path, icir_key) == _get(m20, path, icir_key), f"{path}.{icir_key}"
        assert _get(fund, path, "min_val_abs_ic") == _get(m20, path, "min_val_abs_ic"), f"{path}.min_val_abs_ic"
    # 逐层绝对值（与 technical_monthly 相同）：evaluation/candidate 0.053/0.65，production ×1.25 更严
    assert _get(fund, "evaluation_policy", "min_train_abs_ic") == pytest.approx(0.053, abs=1e-4)
    assert _get(fund, "delivery_policy.candidate", "min_abs_ic") == pytest.approx(0.053, abs=1e-4)
    assert _get(fund, "delivery_policy.production", "min_train_abs_ic") == pytest.approx(0.0663, abs=1e-4)  # 0.053 × 1.25
    assert _get(fund, "delivery_policy.production", "min_train_icir") == pytest.approx(0.6964, abs=1e-4)  # 0.650 × 0.30/0.28


def test_fundamental_bar_within_literature_range() -> None:
    """20d 线必须落在文献实测月度因子有效区间（PE/PB 月度量级），既不高也不形同虚设。"""
    ic_bar, icir_bar = MONTHLY_BAR
    assert 0.02 <= ic_bar <= 0.06, f"IC 线 {ic_bar} 超出月度有效量级 0.02~0.06"
    assert 0.3 <= icir_bar <= 0.70, f"ICIR 线 {icir_bar} 超出月度有效量级 0.3~0.70"


def test_fundamental_scale_free_matches_monthly() -> None:
    """与 label 尺度无关的项：fundamental 与 technical_monthly（及主档）完全一致，
    不再有基本面档独有放宽。"""
    fund = effective_research_spec("fundamental")
    m20 = effective_research_spec("technical_monthly")
    for path, key in SCALE_FREE:
        assert _get(fund, path, key) == _get(m20, path, key), f"{path}.{key} 应同值"
    eg_f = (fund["delivery_policy"]["production"] or {}).get("engine_gate") or {}
    eg_m = (m20["delivery_policy"]["production"] or {}).get("engine_gate") or {}
    assert eg_f["min_excess_annual"] == eg_m["min_excess_annual"] == 0.03
    assert eg_f["min_excess_sharpe"] == eg_m["min_excess_sharpe"] == 0.5
    pd_f = (fund["delivery_policy"]["production"] or {})
    pd_m = (m20["delivery_policy"]["production"] or {})
    assert pd_f["max_winsorized_abs_ic_decay"] == pd_m["max_winsorized_abs_ic_decay"] == 0.10


# ── 盲测绝对门与 val 段一致（2026-10-09 用户定调）───────────────────────
# 盲测是唯一诚实样本外，绝对门与各档 val 段绝对门同一把尺子：1d→0.015、5d→0.0225、
# 20d→0.0398（= 各档 min_val_abs_ic）。旧 0.010 的 t≈2.5 论证依赖 ~410 独立日样本，
# 实测 IC 自相关 0.998+ 新息样本远少，且 20d 档 0.010 仅为候选线 0.053 的 1/5 量纲失配。
BLIND_EXPECTED_ABS_IC = {
    "technical": 0.015,
    "technical_weekly": 0.0225,
    "technical_monthly": 0.0398,
    "fundamental": 0.0398,
}


@pytest.mark.parametrize("mode", sorted(BLIND_EXPECTED_ABS_IC))
def test_blind_test_abs_ic_aligns_with_val_gate(mode: str) -> None:
    """盲测绝对门 == 本档 val 段绝对门（test 段与 val 同一把尺子）。"""
    spec = effective_research_spec(mode)
    blind = spec["delivery_policy"]["blind_test"]
    prod = spec["delivery_policy"]["production"]
    assert blind["min_test_abs_ic"] == pytest.approx(prod["min_val_abs_ic"], abs=1e-4)
    assert blind["min_test_abs_ic"] == pytest.approx(BLIND_EXPECTED_ABS_IC[mode], abs=1e-4)


def test_blind_test_canonical_default_matches_1d_val() -> None:
    """canonical 默认（1d 口径）= 1d val 绝对门 0.015。"""
    assert DeliveryCriteria.defaults().blind_test.min_test_abs_ic == pytest.approx(0.015, abs=1e-4)
