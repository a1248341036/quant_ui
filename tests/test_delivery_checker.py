"""交付门槛收敛的等价性测试。

- 单一来源：DeliveryCriteria.defaults() 与 DEFAULT_RESEARCH_SPEC.delivery_policy
  逐键一致（含 candidate/production/engine_gate 全部数值）。
- 严格等价：DeliveryChecker 的 Stage 判定在 research_spec 默认数值下，
  与旧 submit.py 硬编码判定函数（_check_stage_one/_check_stage_two 等，
  本次重构移除前的语义）逐位一致——边界/缺失/方向反转/换手等场景。
"""

from __future__ import annotations

import numpy as np
import pytest

from alphaagent.factor.mining.delivery_checker import DeliveryChecker
from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria
from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC


# ── 单一来源 ─────────────────────────────────────────────────────


def test_criteria_defaults_match_research_spec():
    """criteria 默认值必须与 research_spec.delivery_policy 逐键一致（唯一真源）。"""
    spec_dp = DEFAULT_RESEARCH_SPEC["delivery_policy"]
    criteria_dp = DeliveryCriteria.defaults().to_spec_dict()
    assert criteria_dp == spec_dp

    # 关键数值抽查（防止整体相等被结构差异掩盖后仍然误报）
    cand = criteria_dp["candidate"]
    assert cand["min_abs_ic"] == 0.015
    assert cand["min_icir"] == 0.25
    assert cand["min_coverage"] == 0.85
    assert cand["max_abs_corr"] == 0.5
    assert cand["min_cs_autocorr"] == 0.18
    assert cand["min_val_ic_retention"] == 0.5

    prod = criteria_dp["production"]
    assert prod["min_train_abs_ic"] == 0.025
    assert prod["min_train_icir"] == 0.30
    assert prod["min_val_abs_ic"] == 0.015
    assert prod["min_val_ic_retention"] == 0.50  # 2026-08-29 从 0.60 下调
    assert prod["min_val_long_excess"] == 0.0
    assert prod["max_winsorized_abs_ic_decay"] == 0.10
    assert prod["max_abs_corr"] == 0.4


def test_criteria_from_spec_fills_missing_keys():
    """from_spec 对缺失键回落 canonical 默认；空 spec 得到默认门槛。"""
    base = DeliveryCriteria.defaults()
    partial = DeliveryCriteria.from_spec({
        "candidate": {"min_abs_ic": 0.02},
        "production": {"min_train_abs_ic": 0.03},
    })
    assert partial.candidate.min_abs_ic == 0.02
    assert partial.candidate.min_icir == 0.25  # 回落默认
    assert partial.production.min_train_abs_ic == 0.03
    assert partial.production.min_train_icir == 0.30

    empty = DeliveryCriteria.from_spec({})
    assert empty == base
    none = DeliveryCriteria.from_spec(None)
    assert none == base


def test_criteria_from_full_research_spec():
    """传入完整 research_spec 时从 delivery_policy 取数。"""
    c = DeliveryCriteria.from_spec(DEFAULT_RESEARCH_SPEC)
    assert c == DeliveryCriteria.defaults()


# ── 严格等价：Stage 判定 == 旧 check 语义（spec 数值下）──


@pytest.fixture()
def checker() -> DeliveryChecker:
    return DeliveryChecker(DeliveryCriteria.defaults())


# 复刻旧 _stage_one_stats_reasons + _stage_one_turnover_reasons 语义
def _old_stage_one_stats(metrics):
    reasons = []
    ic = metrics.get("ic")
    if ic is None or abs(float(ic)) < 0.015:
        reasons.append("ic")
    icir = metrics.get("icir")
    if icir is None or abs(float(icir)) < 0.25:
        reasons.append("icir")
    cov = metrics.get("coverage") or metrics.get("factor_coverage")
    if cov is None or float(cov) <= 0.85:
        reasons.append("coverage")
    ac = metrics.get("cs_pearson_autocorr")
    if ac is None or not np.isfinite(float(ac)) or float(ac) < 0.18:
        reasons.append("cs_autocorr")
    return reasons


# 复刻旧 _stage_one_val_retention_reasons 语义（候选阈值 0.5）
def _old_stage_one_val_retention(train, val):
    n_days = val.get("n_days") or val.get("n_instruments")
    if n_days is not None and int(n_days) == 0:
        return []
    t_ic, v_ic = train.get("ic"), val.get("ic")
    if t_ic is None or v_ic is None:
        return ["val_ic_missing"]
    t, v = float(t_ic), float(v_ic)
    if not np.isfinite(t) or not np.isfinite(v):
        return ["val_ic_missing"]
    if t * v < 0:
        return ["val_sign_flip"]
    if abs(t) > 1e-12 and abs(v) / abs(t) < 0.5:
        return ["val_retention"]
    return []


# 复刻旧 _check_stage_two 语义（production 阈值，保留比 0.50）
def _old_stage_two(train, val, similarity):
    reasons = []
    ic = train.get("ic")
    if ic is None or abs(float(ic)) < 0.025:
        reasons.append("train_ic")
    icir = train.get("icir")
    if icir is None or abs(float(icir)) < 0.30:
        reasons.append("train_icir")
    v_ic = val.get("ic")
    if v_ic is None or abs(float(v_ic)) < 0.015:
        reasons.append("val_ic")
    reasons += _old_stage_one_val_retention_with(train, val, 0.50)  # 2026-08-29 0.60 → 0.50
    thr_vle = 0.0
    vle = val.get("val_long_excess")
    if vle is None or not np.isfinite(float(vle)) or float(vle) <= thr_vle:
        reasons.append("val_long_excess")
    wd = train.get("winsorized_abs_ic_decay")
    if wd is None or float(wd) > 0.10:
        reasons.append("winsorized_abs_ic_decay")
    corr = (similarity or {}).get("max_abs_corr", 0.0)
    if corr is None or float(corr) >= 0.4:
        reasons.append("max_cs_corr")
    return reasons


def _old_stage_one_val_retention_with(train, val, min_ratio):
    n_days = val.get("n_days") or val.get("n_instruments")
    if n_days is not None and int(n_days) == 0:
        return []
    t_ic, v_ic = train.get("ic"), val.get("ic")
    if t_ic is None or v_ic is None:
        return ["val_ic_missing"]
    t, v = float(t_ic), float(v_ic)
    if not np.isfinite(t) or not np.isfinite(v):
        return ["val_ic_missing"]
    if t * v < 0:
        return ["val_sign_flip"]
    if abs(t) > 1e-12 and abs(v) / abs(t) < min_ratio:
        return ["val_retention"]
    return []


GOOD_STAGE1 = {"ic": 0.04, "icir": 0.4, "coverage": 0.90, "cs_pearson_autocorr": 0.6}
GOOD_STAGE2_TRAIN = {"ic": 0.04, "icir": 0.4, "winsorized_abs_ic_decay": 0.02}
GOOD_STAGE2_VAL = {"ic": 0.03, "val_long_excess": 0.02, "n_days": 100}
LOW_CORR = {"max_abs_corr": 0.1}


@pytest.mark.parametrize("mut", [
    {},                       # 全达标
    {"ic": 0.005},            # IC 过低
    {"icir": 0.1},            # ICIR 过低
    {"coverage": 0.8},        # coverage 过低
    {"cs_pearson_autocorr": 0.05},  # 换手过低
    {"ic": None},             # 缺失 IC
    {"icir": None},           # 缺失 ICIR
    {"coverage": None},       # 缺失 coverage
    {"cs_pearson_autocorr": None},  # 缺失换手
    {"ic": 0.014999},         # 恰好低于 IC 门槛
    {"icir": 0.25},           # 恰好等于 ICIR 门槛（< 判定，放行）
    {"coverage": 0.85},       # 恰好等于 coverage 门槛（<= 拒绝）
])
def test_stage_one_stats_parity(checker, mut):
    metrics = {**GOOD_STAGE1, **mut}
    result = checker.stage_one_stats(metrics)
    old = _old_stage_one_stats(metrics)
    assert result.passed == (len(old) == 0)
    assert sorted(result.fail_reasons) == sorted(old)


@pytest.mark.parametrize("train,val", [
    (GOOD_STAGE2_TRAIN, GOOD_STAGE2_VAL),
    (GOOD_STAGE2_TRAIN, {"ic": 0.005, "val_long_excess": 0.02, "n_days": 100}),  # val IC 低
    (GOOD_STAGE2_TRAIN, {"ic": -0.03, "val_long_excess": 0.02, "n_days": 100}),  # 方向反转
    (GOOD_STAGE2_TRAIN, {"ic": 0.03, "val_long_excess": -0.01, "n_days": 100}),  # val 多头超额为负
    (GOOD_STAGE2_TRAIN, {"ic": 0.03, "val_long_excess": 0.02, "n_days": 0}),     # val 空窗
    ({**GOOD_STAGE2_TRAIN, "winsorized_abs_ic_decay": 0.15}, GOOD_STAGE2_VAL),   # 截尾衰减过高
    ({**GOOD_STAGE2_TRAIN, "ic": 0.02}, GOOD_STAGE2_VAL),                        # train IC 低
    ({**GOOD_STAGE2_TRAIN, "icir": 0.2}, GOOD_STAGE2_VAL),                       # train ICIR 低
    (GOOD_STAGE2_TRAIN, {}),                                                      # val 缺失
])
def test_stage_two_parity(checker, train, val):
    result = checker.stage_two(train, val, LOW_CORR)
    old = _old_stage_two(train, val, LOW_CORR)
    assert result.passed == (len(old) == 0)
    assert sorted(result.fail_reasons) == sorted(old)


def test_stage_two_high_corr(checker):
    result = checker.stage_two(GOOD_STAGE2_TRAIN, GOOD_STAGE2_VAL, {"max_abs_corr": 0.5})
    old = _old_stage_two(GOOD_STAGE2_TRAIN, GOOD_STAGE2_VAL, {"max_abs_corr": 0.5})
    assert result.passed == (len(old) == 0)
    assert "max_cs_corr" in result.fail_reasons


def test_stage_one_val_retention_parity(checker):
    cases = [
        (GOOD_STAGE2_TRAIN, GOOD_STAGE2_VAL),
        (GOOD_STAGE2_TRAIN, {"ic": -0.03, "n_days": 100}),  # 方向反转
        (GOOD_STAGE2_TRAIN, {"ic": 0.02, "n_days": 100}),   # 保留比 0.5 正好过
        (GOOD_STAGE2_TRAIN, {"ic": 0.015, "n_days": 100}),  # 保留比 0.375 < 0.5
        (GOOD_STAGE2_TRAIN, {"ic": None, "n_days": 100}),   # val IC 缺失
        (GOOD_STAGE2_TRAIN, {"n_days": 0}),                 # val 空窗跳过
    ]
    for train, val in cases:
        result = checker.stage_one_val_retention(train, val)
        old = _old_stage_one_val_retention(train, val)
        assert result.passed == (len(old) == 0), (train, val, result, old)
        assert sorted(result.fail_reasons) == sorted(old), (train, val, result, old)


@pytest.mark.parametrize("corr", [0.1, 0.5, 0.6, None, 0.0])
def test_stage_one_correlation_parity(checker, corr):
    sim = {"max_abs_corr": corr} if corr is not None else {}
    result = checker.stage_one_correlation(sim)
    old_corr = corr if corr is not None else 0.0
    old_fail = ["max_cs_corr"] if (old_corr is None or float(old_corr) >= 0.5) else []
    assert result.passed == (len(old_fail) == 0)
    assert sorted(result.fail_reasons) == sorted(old_fail)


# ── 模式差异仍生效 ───────────────────────────────────────────────


def test_fundamental_criteria_prompt_reflects_mode():
    from alphaagent.factor.mining.research_spec import default_research_spec

    fund = default_research_spec("fundamental")
    c = DeliveryCriteria.from_spec(fund)
    assert c.candidate.min_abs_ic == 0.012
    assert c.candidate.min_icir == 0.20
    assert c.production.min_train_abs_ic == 0.020
    assert c.production.min_train_icir == 0.28
    assert c.engine_gate.freq == "monthly"
    assert c.engine_gate.min_excess_annual == 0.02

    text = c.to_prompt_text()
    assert "0.012" in text and "0.2" in text and "monthly" in text
